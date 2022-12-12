import argparse
import os
from pathlib import Path
import numpy as np
import wandb
import yaml
import pickle
from tqdm import tqdm
import random

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler
from torch.utils import data

import pytorch3d.transforms as transforms 

from ema_pytorch import EMA
from multiprocessing import cpu_count

from body_model.body_model import BodyModel

from egoego.data.diffusion_amass_dataset_v2 import AMASSDataset, quat_ik_torch, quat_fk_torch
from egoego.model.transformer_cond_diffusion_model import CondGaussianDiffusion
from egoego.model.tcn_cond_diffusion_model import TCNCondGaussianDiffusion

from egoego.vis.blender_vis_mesh_motion import run_blender_rendering_and_save2video, save_verts_faces_to_mesh_file
from egoego.vis.mesh_motion import get_mesh_verts_faces_for_human_only 

from egoego.vis.pose import show3Dpose_animation_smpl22
from egoego.lafan1.utils import rotate_at_frame_smplh

def cycle(dl):
    while True:
        for data in dl:
            yield data

class Trainer(object):
    def __init__(
        self,
        opt,
        diffusion_model,
        *,
        ema_decay = 0.995,
        train_batch_size = 32,
        train_lr = 1e-4,
        train_num_steps = 10000000,
        gradient_accumulate_every = 2,
        amp = False,
        step_start_ema = 2000,
        ema_update_every = 10,
        save_and_sample_every = 200000,
        results_folder = './results',
        use_wandb=True,
    ):
        super().__init__()

        self.w_vel_loss = opt.w_vel_loss 

        self.use_wandb = use_wandb           
        if self.use_wandb:
            # Loggers
            wandb.init(config=opt, project=opt.wandb_pj_name, entity=opt.entity, name=opt.exp_name, dir=opt.save_dir)

        self.model = diffusion_model
        self.ema = EMA(diffusion_model, beta=ema_decay, update_every=ema_update_every)

        self.step_start_ema = step_start_ema
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_num_steps = train_num_steps

        self.optimizer = Adam(diffusion_model.parameters(), lr=train_lr)

        self.step = 0

        self.amp = amp
        self.scaler = GradScaler(enabled=amp)

        self.results_folder = results_folder

        self.vis_folder = results_folder.replace("weights", "vis_res")

        NUM_BETAS = 16
        self.smpl_batch_size = opt.window 
        smplh_path = "/viscam/u/jiamanli/github/hm_interaction/smpl_all_models/smplh_amass"  
        male_bm_path = os.path.join(smplh_path, 'male/model.npz')
        self.male_bm = BodyModel(bm_path=male_bm_path, num_betas=NUM_BETAS, batch_size=self.smpl_batch_size)
        female_bm_path = os.path.join(smplh_path,  'female/model.npz')
        self.female_bm = BodyModel(bm_path=female_bm_path, num_betas=NUM_BETAS, batch_size=self.smpl_batch_size)
        for p in self.male_bm.parameters():
            p.requires_grad = False
        for p in self.female_bm.parameters():
            p.requires_grad = False 

        self.male_bm = self.male_bm.cuda()
        self.female_bm = self.female_bm.cuda()

        self.bm_dict = {'male' : self.male_bm, 'female' : self.female_bm}

        self.train_w_cond_diffusion = opt.train_w_cond_diffusion 
        self.use_head_condition = opt.use_head_condition 
        self.use_first_pose_condition = opt.use_first_pose_condition

        self.add_velocity_rep = opt.add_velocity_rep 

        self.opt = opt 

        self.prep_dataloader(window_size=opt.window)

    def prep_dataloader(self, window_size=None):
        if window_size is None:
            window_size = random.sample([60, 120, 180, 240], 1)[0] 
        # Define dataset
        train_dataset = AMASSDataset(self.opt, train=True, window=window_size, use_subset=False)
        val_dataset = AMASSDataset(self.opt, train=False, window=window_size, use_subset=False)

        self.ds = train_dataset 
        self.val_ds = val_dataset
        self.dl = cycle(data.DataLoader(self.ds, batch_size=self.batch_size, shuffle=True, pin_memory=True, num_workers=0))
        self.val_dl = cycle(data.DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False, pin_memory=True, num_workers=0))

        # self.model.assign_mean_std_for_normalize(train_dataset.global_jpos_mean, train_dataset.global_jpos_std, \
        #     train_dataset.global_jvel_mean, train_dataset.global_jvel_std)

    def save(self, milestone):
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.scaler.state_dict()
        }
        torch.save(data, os.path.join(self.results_folder, 'model-'+str(milestone)+'.pt'))

    def load(self, milestone):
        data = torch.load(os.path.join(self.results_folder, 'model-'+str(milestone)+'.pt'))

        self.step = data['step']
        self.model.load_state_dict(data['model'], strict=False)
        self.ema.load_state_dict(data['ema'], strict=False)
        self.scaler.load_state_dict(data['scaler'])

    def train(self):
        # self.load("nan")
        # torch.autograd.set_detect_anomaly(True)

        init_step = self.step 
        for idx in range(init_step, self.train_num_steps):
            self.optimizer.zero_grad()

            # change_data_steps = 20000
            # if idx > 0:
            #     if idx % change_data_steps == 0:
            #         self.prep_dataloader()

            nan_exists = False # If met nan in loss or gradient, need to skip to next data. 
            for i in range(self.gradient_accumulate_every):
                data = next(self.dl).cuda()

                with autocast(enabled = self.amp):
                    if self.train_w_cond_diffusion:
                        if self.use_head_condition:
                            cond_mask = self.prep_head_condition_mask(data) # BS X T X D 
                            if self.use_first_pose_condition:
                                # Randomly mask first pose
                                if random.random() < 0.5:
                                    frame_mask = self.prep_temporal_condition_mask(data)
                                    cond_mask = cond_mask * frame_mask

                        loss_diffusion, loss_root_v, loss_contact_feet = self.model(data, cond_mask)
                    else:
                        loss_diffusion, loss_root_v, loss_contact_feet = self.model(data)
                    
                    loss = loss_diffusion + self.w_vel_loss * loss_root_v + self.w_vel_loss * loss_contact_feet

                    if torch.isnan(loss).item():
                        print('WARNING: NaN loss. Skipping to next data...')
                        nan_exists = True 
                        torch.cuda.empty_cache()
                        continue

                    self.scaler.scale(loss / self.gradient_accumulate_every).backward()
                    # loss.backward()

                    # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.25)
                    # for param in self.model.parameters():
                    #     if param.grad is not None:
                    #         zeros = torch.zeros_like(param.grad).to(param.grad.device)
                    #         param.grad = torch.where(param.grad.isnan(), zeros, param.grad)

                    # check gradients
                    parameters = [p for p in self.model.parameters() if p.grad is not None]
                    total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2.0).to(data.device) for p in parameters]), 2.0)
                    if torch.isnan(total_norm):
                        print('WARNING: NaN gradients. Skipping to next data...')
                        nan_exists = True 

                        # Save checkpoint 
                        # self.save("nan")
                        # import pdb 
                        # pdb.set_trace() 
                        torch.cuda.empty_cache()
                        continue

                    if self.use_wandb:
                        log_dict = {
                            "Train/Loss/Total Loss": loss.item(),
                            "Train/Loss/Diffusion Loss": loss_diffusion.item(),
                            "Train/Loss/Root velocity Loss": loss_root_v.item(),
                            "Train/Loss/Contact feet velocity Loss": loss_contact_feet.item(),
                        }
                        wandb.log(log_dict)

                    if idx % 50 == 0 and i == 0:
                        print("Step: {0}".format(idx))
                        print("Loss: %.4f" % (loss.item()))

                # pbar.set_description(f'loss: {loss.item():.4f}')

            if nan_exists:
                continue

            self.scaler.step(self.optimizer)
            self.scaler.update()

            # self.opt.step()
            # self.opt.zero_grad()

            self.ema.update()

            if self.step != 0 and self.step % self.save_and_sample_every == 0:
                self.ema.ema_model.eval()
                with torch.no_grad():
                    milestone = self.step // self.save_and_sample_every
                    # batches = num_to_groups(36, self.batch_size)
                    # batches = num_to_groups(36, 1)
                    # all_res_list = list(map(lambda n: self.ema.ema_model.sample(batch_size=n), batches))
                    bs_for_vis = 4
                    if self.train_w_cond_diffusion:
                        if self.use_head_condition:
                            data = next(self.val_dl).cuda() # BS X T X D 
                            cond_mask = self.prep_head_condition_mask(data) # BS X T X D 
                            if self.use_first_pose_condition:
                                # Randomly mask first pose
                                if random.random() < 0.5:
                                    frame_mask = self.prep_temporal_condition_mask(data)
                                    cond_mask = cond_mask * frame_mask
                        all_res_list = self.ema.ema_model.sample(data[:bs_for_vis], \
                        cond_mask[:bs_for_vis])
                    else:
                        all_res_list = self.ema.ema_model.sample(batch_size=bs_for_vis)

                # Visualization
                # self.gen_vis_res(data, self.step) # For debug GT 
                if self.train_w_cond_diffusion:  
                    for_vis_gt_data = data[:bs_for_vis]
                    self.gen_vis_res(for_vis_gt_data, self.step, vis_gt=True)

                self.gen_vis_res(all_res_list, self.step)

                self.save(milestone)

            self.step += 1
            # pbar.update(1)

        print('training complete')

        if self.use_wandb:
            wandb.run.finish()
    
    def prep_head_condition_mask(self, data, joint_idx=15):
        # data: BS X T X D 
        # head_idx = 15 
        # Condition part is zeros, while missing part is ones. 
        mask = torch.ones_like(data).to(data.device)

        if self.opt.add_velocity_rep:
            cond_pos_dim_idx = joint_idx * 3 
            cond_vel_dim_idx = 22 * 3 + joint_idx * 3
            cond_rot_dim_idx = 2 * 22 * 3 + joint_idx * 6
            mask[:, :, cond_pos_dim_idx:cond_pos_dim_idx+3] = torch.zeros(data.shape[0], data.shape[1], 3).to(data.device)
            mask[:, :, cond_vel_dim_idx:cond_vel_dim_idx+3] = torch.zeros(data.shape[0], data.shape[1], 3).to(data.device)
            mask[:, :, cond_rot_dim_idx:cond_rot_dim_idx+6] = torch.zeros(data.shape[0], data.shape[1], 6).to(data.device)
        else:
            cond_pos_dim_idx = joint_idx * 3 
            cond_rot_dim_idx = 22 * 3 + joint_idx * 6
            mask[:, :, cond_pos_dim_idx:cond_pos_dim_idx+3] = torch.zeros(data.shape[0], data.shape[1], 3).to(data.device)
            mask[:, :, cond_rot_dim_idx:cond_rot_dim_idx+6] = torch.zeros(data.shape[0], data.shape[1], 6).to(data.device)

        return mask 

    def prep_temporal_condition_mask(self, data, t_idx=0):
        mask = torch.ones_like(data).to(data.device) # BS X T X D 
        mask[:, t_idx, :] = torch.zeros(data.shape[0], data.shape[2]).to(data.device) # BS X D  

        return mask 

    def cond_sample_res(self):
        weights = os.listdir(self.results_folder)
        weights_paths = [os.path.join(self.results_folder, weight) for weight in weights]
        weight_path = max(weights_paths, key=os.path.getctime)
   
        print(f"Loaded weight: {weight_path}")

        milestone = weight_path.split("/")[-1].split("-")[-1].replace(".pt", "")
        
        self.load(milestone)
        self.ema.ema_model.eval()
        num_sample = 4
        with torch.no_grad():
            for s_idx in range(num_sample):
                if self.use_head_condition:
                    data = next(self.val_dl).cuda() # BS X T X D 
                    cond_mask = self.prep_head_condition_mask(data) # BS X T X D 
                    if self.use_first_pose_condition:
                        frame_mask = self.prep_temporal_condition_mask(data)
                        cond_mask = cond_mask * frame_mask

                    max_num = 1
                    all_res_list = self.ema.ema_model.sample(x_start=data[:max_num], \
                    cond_mask=cond_mask[:max_num])
                
                if self.use_head_condition:
                    vis_tag = "test_head_cond_sample_"+str(s_idx)
                    if self.use_first_pose_condition:
                        vis_tag = "test_init_and_head_cond_sample_"+str(s_idx)
                   
                if self.use_head_condition:
                    # Visualize GT 
                    self.gen_vis_res(data[:max_num], vis_tag, vis_gt=True)

                self.gen_vis_res(all_res_list, vis_tag)

    def full_body_gen_cond_head_pose(self, head_pose, init_root_trans=None, init_pose_aa=None):
        # head_pose: BS X T X 7 
        # init_root_trans: 1 X 3
        # init_pose_aa: 1 X 22 X 3 

        use_pred_global_head_pos = True 
        
        weights = os.listdir(self.results_folder)
        weights_paths = []
        for weight in weights:
            if "nan" not in weight:
                weights_paths.append(os.path.join(self.results_folder, weight))
        # weights_paths = [os.path.join(self.results_folder, weight) for weight in weights]
        weight_path = max(weights_paths, key=os.path.getctime)
   
        print(f"Loaded weight: {weight_path}")

        milestone = weight_path.split("/")[-1].split("-")[-1].replace(".pt", "")
        
        self.load(milestone)
        self.ema.ema_model.eval()

        global_head_jpos = head_pose[:, :, :3] # BS X T X 3 
        global_head_quat = head_pose[:, :, 3:] # BS X T X 4 

        data = torch.zeros(head_pose.shape[0], head_pose.shape[1], 22*3+22*6).to(head_pose.device) # BS X T X D 

        # Canonicalize the first head frame 
        aligned_head_trans, aligned_head_quat, recover_rot_quat = \
        rotate_at_frame_smplh(global_head_jpos.data.cpu().numpy(), \
        global_head_quat.data.cpu().numpy(), cano_t_idx=0)
        # BS(1) X T' X 3, BS(1) X T' X 4, BS(1) X 1 X 1 X 4  

        aligned_head_trans = torch.from_numpy(aligned_head_trans).to(head_pose.device)
        aligned_head_quat = torch.from_numpy(aligned_head_quat).to(head_pose.device)

        move_to_zero_trans = aligned_head_trans[:, 0:1, :].clone() # Move the head joint x, y to 0,  BS X 1 X 3
        move_to_zero_trans[:, :, 2] = 0 

        aligned_head_trans = aligned_head_trans - move_to_zero_trans # BS X T X 3 

        aligned_head_rot_mat = transforms.quaternion_to_matrix(aligned_head_quat) # BS X T X 3 X 3 
        aligned_head_rot_6d = transforms.matrix_to_rotation_6d(aligned_head_rot_mat) # BS X T X 6  

        if init_root_trans is not None and init_pose_aa is not None:
            # Convert the first pose to global jpos and global rotation 6d, also canonicalize based on head orientation
            init_pose_rot_mat = transforms.axis_angle_to_matrix(init_pose_aa) # 1 X 22 X 3 X 3
            init_full_pose = self.ds.process_first_pose(init_root_trans, init_pose_rot_mat)
            # 1 X (22*3+22*6)

            data[0, 0:1, :] = init_full_pose 

        head_idx = 15 
        data[:, :, head_idx*3:head_idx*3+3] = aligned_head_trans
        data[:, :, 22*3+head_idx*6:22*3+head_idx*6+6] = aligned_head_rot_6d 

        # Normalize data to [-1, 1]
        normalized_jpos = self.ds.normalize(data[0, :, :22*3].reshape(data.shape[1], 22, 3)) # T X 22 X 3 
        data[0, :, :22*3] = normalized_jpos.reshape(-1, 22*3) # T X (22*3)

        with torch.no_grad():
            cond_mask = self.prep_head_condition_mask(data) # BS X T X D 
            if self.use_first_pose_condition:
                frame_mask = self.prep_temporal_condition_mask(data)
                cond_mask = cond_mask * frame_mask

            # import pdb 
            # pdb.set_trace() 
            all_res_list = self.ema.ema_model.sample(x_start=data, cond_mask=cond_mask) # BS X T X D

        # De-normalize jpos 
        if self.only_keep_root_trans:
            normalized_global_root_jpos = all_res_list[:, :, :3] # BS X T X 3 
            global_root_jpos = self.model.de_normalize_root_jpos(normalized_global_root_jpos) # BS X T X 3
            
            bs, num_steps, _ = global_root_jpos.shape
            global_rot_6d = all_res_list[:, :, 3:].reshape(bs, num_steps, 22, 6) # BS X T X 22 X 6
        else:
            normalized_global_jpos = all_res_list[0, :, :22*3].reshape(-1, 22, 3) # T X 22 X 3 
            global_jpos = self.ds.de_normalize(normalized_global_jpos) # T X 22 X 3

            global_rot_6d = all_res_list[:, :, 22*3:] # BS X T X (22*6)
            global_jpos = global_jpos[None] # BS X T X 22 X 3
            bs, num_steps, _, _ = global_jpos.shape
            global_rot_6d = global_rot_6d.reshape(bs, num_steps, 22, 6) # BS X T X 22 X 6 
            global_root_jpos = global_jpos[:, :, 0, :]

            head_idx = 15 
            global_head_jpos = global_jpos[:, :, head_idx, :] # BS X T X 3 

        global_rot_mat = transforms.rotation_6d_to_matrix(global_rot_6d) # BS X T X 22 X 3 X 3
        global_quat = transforms.matrix_to_quaternion(global_rot_mat) # BS X T X 22 X 4 
        recover_rot_quat = torch.from_numpy(recover_rot_quat).to(global_quat.device) # BS X 1 X 1 X 4 
        ori_global_quat = transforms.quaternion_multiply(recover_rot_quat, global_quat) # BS X T X 22 X 4 
        ori_global_root_jpos = global_root_jpos # BS X T X 3 
        ori_global_root_jpos = transforms.quaternion_apply(recover_rot_quat.squeeze(1).repeat(1, num_steps, 1), \
                        ori_global_root_jpos) # BS X T X 3 

        if use_pred_global_head_pos:
            ori_global_head_jpos = global_head_jpos # BS X T X 3 
            ori_global_head_jpos = transforms.quaternion_apply(recover_rot_quat.squeeze(1).repeat(1, num_steps, 1), \
                            ori_global_head_jpos) # BS X T X 3 

        # Convert global join rotation to local joint rotation
        ori_global_rot_mat = transforms.quaternion_to_matrix(ori_global_quat) # BS X T X 22 X 4 
        ori_local_rot_mat = quat_ik_torch(ori_global_rot_mat[0]) # T X 22 X 3 X 3 
        ori_local_aa_rep = transforms.matrix_to_axis_angle(ori_local_rot_mat) # T X 22 X 3  
        
        # vis_res = True 
        # if vis_res:
        #     self.gen_vis_res(all_res_list, step="test_vis_res")

            # For debug 
            # gt_data = next(self.val_dl).cuda() # BS X T X D 
            # self.gen_vis_res(gt_data, step="test_vis_gt_res")

            # import pdb 
            # pdb.set_trace() 
    
        if use_pred_global_head_pos:
            zero_root_trans = torch.zeros(ori_local_aa_rep.shape[0], 3).to(ori_local_aa_rep.device).float()
            bs = 1
            betas = torch.zeros(bs, 10).to(zero_root_trans.device).float()
            gender = ["male"] * bs 

            mesh_jnts, _, _ = \
            get_mesh_verts_faces_for_human_only(zero_root_trans[None], \
            ori_local_aa_rep[None].float(), betas, gender, \
            self.bm_dict, self.smpl_batch_size)
            # BS(1) X T' X 22 X 3, BS(1) X T' X Nv X 3

            head_idx = 15 
            wo_root_trans_head_pos = mesh_jnts[0, :, head_idx, :] # T X 3 

            # calculated_root_trans = ori_global_head_jpos[0] - wo_root_trans_head_pos # T X 3 
            calculated_root_trans = head_pose[0, :, :3] - wo_root_trans_head_pos

            return ori_local_aa_rep, calculated_root_trans # T X 22 X 3, T X 3 
        else:
            return ori_local_aa_rep, ori_global_root_jpos[0] # T X 22 X 3, T X 3  
    
    def gen_vis_res(self, all_res_list, step, vis_gt=False):
        # all_res_list: N X T X D 
        num_seq = all_res_list.shape[0]
        normalized_global_jpos = all_res_list[:, :, :22*3].reshape(num_seq, -1, 22, 3)
        global_jpos = self.ds.de_normalize_jpos(normalized_global_jpos.reshape(-1, 22, 3)) # (N*T) X 22 X 3
        global_jpos = global_jpos.reshape(num_seq, -1, 22, 3) # N X T X 22 X 3 
        global_root_jpos = global_jpos[:, :, 0, :].clone() # N X T X 3 

        if self.add_velocity_rep:
            global_rot_6d = all_res_list[:, :, 2*22*3:].reshape(num_seq, -1, 22, 6)
        else:
            global_rot_6d = all_res_list[:, :, 22*3:].reshape(num_seq, -1, 22, 6)
        
        global_rot_mat = transforms.rotation_6d_to_matrix(global_rot_6d) # N X T X 22 X 3 X 3 

        for idx in range(num_seq):
            curr_global_rot_mat = global_rot_mat[idx] # T X 22 X 3 X 3 
            curr_local_rot_mat = quat_ik_torch(curr_global_rot_mat) # T X 22 X 3 X 3 
            curr_local_rot_aa_rep = transforms.matrix_to_axis_angle(curr_local_rot_mat) # T X 22 X 3 
            
            curr_global_root_jpos = global_root_jpos[idx] # T X 3
            move_xy_trans = curr_global_root_jpos.clone()[0:1] # 1 X 3 
            move_xy_trans[:, 2] = 0 
            root_trans = curr_global_root_jpos - move_xy_trans # T X 3 

            # Generate global joint position 
            bs = 1
            betas = torch.zeros(bs, 16).to(root_trans.device)
            gender = ["male"] * bs 

            mesh_jnts, mesh_verts, mesh_faces = \
            get_mesh_verts_faces_for_human_only(root_trans[None], \
            curr_local_rot_aa_rep[None], betas, gender, \
            self.bm_dict, self.smpl_batch_size)
            # BS(1) X T' X 22 X 3, BS(1) X T' X Nv X 3
            
            dest_mesh_vis_folder = os.path.join(self.vis_folder, "blender_mesh_vis", str(step))
            if not os.path.exists(dest_mesh_vis_folder):
                os.makedirs(dest_mesh_vis_folder)

            if vis_gt:
                mesh_save_folder = os.path.join(dest_mesh_vis_folder, \
                                "objs_step_"+str(step)+"_bs_idx_"+str(idx)+"_gt")
                out_rendered_img_folder = os.path.join(dest_mesh_vis_folder, \
                                "imgs_step_"+str(step)+"_bs_idx_"+str(idx)+"_gt")
                out_vid_file_path = os.path.join(dest_mesh_vis_folder, \
                                "vid_step_"+str(step)+"_bs_idx_"+str(idx)+"_gt.mp4")
            else:
                mesh_save_folder = os.path.join(dest_mesh_vis_folder, \
                                "objs_step_"+str(step)+"_bs_idx_"+str(idx))
                out_rendered_img_folder = os.path.join(dest_mesh_vis_folder, \
                                "imgs_step_"+str(step)+"_bs_idx_"+str(idx))
                out_vid_file_path = os.path.join(dest_mesh_vis_folder, \
                                "vid_step_"+str(step)+"_bs_idx_"+str(idx)+".mp4")

            # Visualize the skeleton 
            if vis_gt:
                dest_skeleton_vis_path = os.path.join(dest_mesh_vis_folder, \
                                "vid_step_"+str(step)+"_bs_idx_"+str(idx)+"_skeleton_gt.gif")
            else:
                dest_skeleton_vis_path = os.path.join(dest_mesh_vis_folder, \
                                "vid_step_"+str(step)+"_bs_idx_"+str(idx)+"_skeleton.gif")
            channels = global_jpos[idx:idx+1] # 1 X T X 22 X 3 
            show3Dpose_animation_smpl22(channels.data.cpu().numpy(), dest_skeleton_vis_path) 

            # For visualizing human mesh only 
            save_verts_faces_to_mesh_file(mesh_verts.data.cpu().numpy()[0], mesh_faces.data.cpu().numpy(), mesh_save_folder)
            run_blender_rendering_and_save2video(mesh_save_folder, out_rendered_img_folder, out_vid_file_path, vis_object=False)

def run_train(opt, device):
    # Prepare Directories
    save_dir = Path(opt.save_dir)
    wdir = save_dir / 'weights'
    wdir.mkdir(parents=True, exist_ok=True)

    # Save run settings
    with open(save_dir / 'opt.yaml', 'w') as f:
        yaml.safe_dump(vars(opt), f, sort_keys=True)

    # Define model  
    repr_dim = 22 * 3 + 22 * 6 
    if opt.add_velocity_rep:
        repr_dim += 22 * 3 

    if opt.use_l2_loss:
        loss_type = "l2"
    else:
        loss_type = "l1"
   
    if opt.use_tcn:
        diffusion_model = TCNCondGaussianDiffusion(d_feats=repr_dim,  \
                    max_timesteps=opt.window+1, out_dim=repr_dim, timesteps=1000, \
                    objective="pred_x0", loss_type=loss_type, \
                    add_noisy_head_condition=opt.add_noisy_head_condition, \
                    add_motion_loss=opt.add_motion_loss, \
                    batch_size=opt.batch_size)
    else:
        diffusion_model = CondGaussianDiffusion(d_feats=repr_dim, d_model=opt.d_model, \
                    n_dec_layers=opt.n_dec_layers, n_head=opt.n_head, d_k=opt.d_k, d_v=opt.d_v, \
                    max_timesteps=opt.window+1, out_dim=repr_dim, timesteps=1000, \
                    objective="pred_x0", loss_type=loss_type, \
                    add_noisy_head_condition=opt.add_noisy_head_condition, \
                    add_motion_loss=opt.add_motion_loss, \
                    batch_size=opt.batch_size)
  
    diffusion_model.to(device)

    trainer = Trainer(
        opt,
        diffusion_model,
        train_batch_size=opt.batch_size, # 32
        train_lr=opt.learning_rate, # 1e-4
        train_num_steps=8000000,         # 700000, total training steps
        gradient_accumulate_every=2,    # gradient accumulation steps
        ema_decay=0.995,                # exponential moving average decay
        amp=True,                        # turn on mixed precision
        results_folder=str(wdir),
    )

    trainer.train()

    torch.cuda.empty_cache()

def run_sample(opt, device):
    # Prepare Directories
    save_dir = Path(opt.save_dir)
    wdir = save_dir / 'weights'

    # Define model     
    repr_dim = 22 * 3 + 22 * 6 
    if opt.add_velocity_rep:
        repr_dim += 22 * 3 
   
    if opt.use_l2_loss:
        loss_type = "l2"
    else:
        loss_type = "l1"

    if opt.use_tcn:
        diffusion_model = TCNCondGaussianDiffusion(d_feats=repr_dim,  \
                    max_timesteps=opt.window+1, out_dim=repr_dim, timesteps=1000, \
                    objective="pred_x0", loss_type=loss_type, \
                    add_noisy_head_condition=opt.add_noisy_head_condition, \
                    add_motion_loss=opt.add_motion_loss, \
                    batch_size=opt.batch_size)
    else:
        diffusion_model = CondGaussianDiffusion(d_feats=repr_dim, d_model=opt.d_model, \
                    n_dec_layers=opt.n_dec_layers, n_head=opt.n_head, d_k=opt.d_k, d_v=opt.d_v, \
                    max_timesteps=opt.window+1, out_dim=repr_dim, timesteps=1000, \
                    objective="pred_x0", loss_type=loss_type, \
                    add_noisy_head_condition=opt.add_noisy_head_condition, \
                    add_motion_loss=opt.add_motion_loss, \
                    batch_size=opt.batch_size)

    diffusion_model.to(device)

    trainer = Trainer(
        opt,
        diffusion_model,
        train_batch_size=opt.batch_size, # 32
        train_lr=opt.learning_rate, # 1e-4
        train_num_steps=8000000,         # 700000, total training steps
        gradient_accumulate_every=2,    # gradient accumulation steps
        ema_decay=0.995,                # exponential moving average decay
        amp=True,                        # turn on mixed precision
        results_folder=str(wdir),
        use_wandb=False 
    )
  
    trainer.cond_sample_res()

    torch.cuda.empty_cache()

def get_trainer(opt):
    opt.window = opt.diffusion_window 

    opt.diffusion_save_dir = os.path.join(opt.diffusion_project, opt.diffusion_exp_name)
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

    # Prepare Directories
    save_dir = Path(opt.diffusion_save_dir)
    wdir = save_dir / 'weights'

    # Define model 
    repr_dim = 22 * 3 + 22 * 6 
   
    transformer_diffusion = CondGaussianDiffusion(d_feats=repr_dim, d_model=opt.diffusion_d_model, \
                n_dec_layers=opt.diffusion_n_dec_layers, n_head=opt.diffusion_n_head, \
                d_k=opt.diffusion_d_k, d_v=opt.diffusion_d_v, \
                max_timesteps=opt.diffusion_window+1, out_dim=repr_dim, timesteps=1000, objective="pred_x0", \
                add_noisy_head_condition=opt.add_noisy_head_condition, \
                add_motion_loss=opt.add_motion_loss, \
                batch_size=opt.diffusion_batch_size)

    transformer_diffusion.to(device)

    trainer = Trainer(
        opt,
        transformer_diffusion,
        train_batch_size=opt.diffusion_batch_size, # 32
        train_lr=opt.diffusion_learning_rate, # 1e-4
        train_num_steps=8000000,         # 700000, total training steps
        gradient_accumulate_every=2,    # gradient accumulation steps
        ema_decay=0.995,                # exponential moving average decay
        amp=True,                        # turn on mixed precision
        results_folder=str(wdir),
        use_wandb=False 
    )

    return trainer 

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', default='runs/train', help='project/name')
    parser.add_argument('--wandb_pj_name', type=str, default='', help='project name')
    parser.add_argument('--entity', default='jiamanli', help='W&B entity')
    parser.add_argument('--exp_name', default='', help='save to project/name')

    parser.add_argument('--window', type=int, default=80, help='horizon')

    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--learning_rate', type=float, default=2e-4, help='generator_learning_rate')

    parser.add_argument('--n_dec_layers', type=int, default=4, help='the number of decoder layers')
    parser.add_argument('--n_head', type=int, default=4, help='the number of heads in self-attention')
    parser.add_argument('--d_k', type=int, default=256, help='the dimension of keys in transformer')
    parser.add_argument('--d_v', type=int, default=256, help='the dimension of values in transformer')
    parser.add_argument('--d_model', type=int, default=512, help='the dimension of intermediate representation in transformer')
    
    parser.add_argument('--device', default='0', help='cuda device')
    
    parser.add_argument('--checkpoint', type=str, default="", help='checkpoint')

    parser.add_argument("--use_subset", action="store_true")

    parser.add_argument("--test_sample_res", action="store_true")

    parser.add_argument("--use_head_condition", action="store_true")
    parser.add_argument("--use_first_pose_condition", action="store_true")
    parser.add_argument("--use_resampling", action="store_true")

    parser.add_argument("--train_w_cond_diffusion", action="store_true")

    parser.add_argument("--add_noisy_head_condition", action="store_true")

    parser.add_argument("--add_motion_loss", action="store_true")

    parser.add_argument('--w_vel_loss', type=float, default=1, help='loss weight for root and foot velocity')

    # For data representation
    parser.add_argument("--add_velocity_rep", action="store_true")
    parser.add_argument("--canonicalize_init_head", action="store_true")

    # For moddel architecture 
    parser.add_argument("--use_tcn", action="store_true")

    # For loss type 
    parser.add_argument("--use_l2_loss", action="store_true")

    opt = parser.parse_args()
    return opt

if __name__ == "__main__":
    opt = parse_opt()
    opt.save_dir = os.path.join(opt.project, opt.exp_name)
    opt.exp_name = opt.save_dir.split('/')[-1]
    device = torch.device(f"cuda:{opt.device}" if torch.cuda.is_available() else "cpu")
    if opt.test_sample_res:
        run_sample(opt, device)
    else:
        run_train(opt, device)
