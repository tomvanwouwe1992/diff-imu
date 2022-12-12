import numpy as np
import os
import random
import joblib 
import pickle 
import cv2 

import torch
from torch.utils.data import Dataset
import pytorch3d.transforms as transforms 

from body_model.body_model import BodyModel
from body_model.utils import SMPL_JOINTS, KEYPT_VERTS

from egoego.lafan1.utils import rotate_at_frame_smplh
from egoego.vis.mesh_motion import get_mesh_verts_faces_for_human_only 

def get_smpl_parents():
    smplh_path = "/viscam/u/jiamanli/github/hm_interaction/smpl_all_models/smplh_amass"
    bm_path = os.path.join(smplh_path, 'male/model.npz')
    npz_data = np.load(bm_path)
    ori_kintree_table = npz_data['kintree_table'] # 2 X 52 
    parents = ori_kintree_table[0, :22] # 22 
    parents[0] = -1 # Assign -1 for the root joint's parent idx.

    return parents

def local2global_pose(local_pose):
    # local_pose: T X J X 3 X 3 
    kintree = get_smpl_parents() 

    bs = local_pose.shape[0]

    local_pose = local_pose.view(bs, -1, 3, 3)

    global_pose = local_pose.clone()

    for jId in range(len(kintree)):
        parent_id = kintree[jId]
        if parent_id >= 0:
            global_pose[:, jId] = torch.matmul(global_pose[:, parent_id], global_pose[:, jId])

    return global_pose # T X J X 3 X 3 

def quat_ik_torch(grot_mat):
    # grot: T X J X 3 X 3 
    parents = get_smpl_parents() 

    grot = transforms.matrix_to_quaternion(grot_mat) # T X J X 4 

    res = torch.cat(
            [
                grot[..., :1, :],
                transforms.quaternion_multiply(transforms.quaternion_invert(grot[..., parents[1:], :]), \
                grot[..., 1:, :]),
            ],
            dim=-2) # T X J X 4 

    res_mat = transforms.quaternion_to_matrix(res) # T X J X 3 X 3 

    return res_mat 

def quat_fk_torch(lrot_mat, lpos):
    # lrot: N X J X 3 X 3 (local rotation with reprect to its parent joint)
    # lpos: N X J X 3 (root joint is in global space, the other joints are offsets relative to its parent in rest pose)
    parents = get_smpl_parents() 

    lrot = transforms.matrix_to_quaternion(lrot_mat)

    gp, gr = [lpos[..., :1, :]], [lrot[..., :1, :]]
    for i in range(1, len(parents)):
        gp.append(
            transforms.quaternion_apply(gr[parents[i]], lpos[..., i : i + 1, :]) + gp[parents[i]]
        )
        gr.append(transforms.quaternion_multiply(gr[parents[i]], lrot[..., i : i + 1, :]))

    res = torch.cat(gr, dim=-2), torch.cat(gp, dim=-2)

    return res

class AMASSDataset(Dataset):
    def __init__(
        self,
        opt,
        train,
        window=60,
        for_eval=False,
        use_aposer_split=False,
        use_subset=False 
    ):
        self.opt = opt 

        self.train = train
        
        self.window = window

        self.use_subset = use_subset 

        self.rest_human_offsets = self.get_rest_pose_joints() # 1 X J X 3 

        data_root_folder = "/move/u/jiamanli/datasets/egoego_processed_data/amass_same_shape_egoego_processed"
        if self.train:
            if use_aposer_split:
                self.data_path = os.path.join(data_root_folder, "train_aposer_amass_smplh_motion.p")
            else:
                self.data_path = os.path.join(data_root_folder, "train_amass_smplh_motion.p")
        else:
            if use_aposer_split:
                self.data_path = os.path.join(data_root_folder, "test_aposer_amass_smplh_motion.p")
            else:
                self.data_path = os.path.join(data_root_folder, "test_amass_smplh_motion.p")
        
        ori_data_dict = joblib.load(self.data_path)
        self.data_dict = self.filter_data(ori_data_dict)

        if self.train:
            if self.opt.canonicalize_init_head:
                processed_data_path = os.path.join(data_root_folder, "cano_train_diffusion_amass_window_"+str(self.window)+".p")
            else:
                processed_data_path = os.path.join(data_root_folder, "train_diffusion_amass_window_"+str(self.window)+".p")
        else: 
            if self.opt.canonicalize_init_head:
                processed_data_path = os.path.join(data_root_folder, "cano_test_diffusion_amass_window_"+str(self.window)+".p")
            else:
                processed_data_path = os.path.join(data_root_folder, "test_diffusion_amass_window_"+str(self.window)+".p")

        if self.opt.canonicalize_init_head:
            mean_std_data_path = os.path.join(data_root_folder, "cano_mean_std_data_window_"+str(self.window)+".p")
        else:
            mean_std_data_path = os.path.join(data_root_folder, "mean_std_data_window_"+str(self.window)+".p")
        
        if os.path.exists(processed_data_path):
            self.window_data_dict = joblib.load(processed_data_path)
        else:
            mean_std_dict = self.cal_normalize_data_input()
            joblib.dump(self.window_data_dict, processed_data_path)
            if self.train:
                joblib.dump(mean_std_dict, mean_std_data_path)

        if os.path.exists(mean_std_data_path):
            mean_std_jpos_data = joblib.load(mean_std_data_path)
           
            self.global_jpos_mean = torch.from_numpy(mean_std_jpos_data['global_jpos_mean']).float().reshape(22, 3)[None]
            self.global_jpos_std = torch.from_numpy(mean_std_jpos_data['global_jpos_std']).float().reshape(22, 3)[None]
            self.global_jvel_mean = torch.from_numpy(mean_std_jpos_data['global_jvel_mean']).float().reshape(22, 3)[None]
            self.global_jvel_std = torch.from_numpy(mean_std_jpos_data['global_jvel_std']).float().reshape(22, 3)[None]

        self.for_eval = for_eval

        if self.train:
            print("Total number of windows for training:{0}".format(len(self.window_data_dict)))
        else:
            print("Total number of windows for validation:{0}".format(len(self.window_data_dict)))

    def get_rest_pose_joints(self):
        NUM_BETAS = 16
        smpl_batch_size = 1
        smplh_path = "/viscam/u/jiamanli/github/hm_interaction/smpl_all_models/smplh_amass"  
        male_bm_path = os.path.join(smplh_path, 'male/model.npz')
        male_bm = BodyModel(bm_path=male_bm_path, num_betas=NUM_BETAS, batch_size=smpl_batch_size)
        female_bm_path = os.path.join(smplh_path,  'female/model.npz')
        female_bm = BodyModel(bm_path=female_bm_path, num_betas=NUM_BETAS, batch_size=smpl_batch_size)
       
        male_bm = male_bm.cuda()
        female_bm = female_bm.cuda() 

        bm_dict = {'male' : male_bm, 'female' : female_bm}

        zero_root_trans = torch.zeros(1, 1, 3).cuda()
        zero_rot_aa_rep = torch.zeros(1, 1, 22, 3).cuda()
        bs = 1 
        betas = torch.zeros(1, NUM_BETAS).cuda().float()
        gender = ["male"] * bs 

        rest_human_jnts, rest_human_verts, human_faces = \
        get_mesh_verts_faces_for_human_only(zero_root_trans, \
        zero_rot_aa_rep, betas, gender, \
        bm_dict, smpl_batch_size)
        # 1 X 1 X J X 3 

        parents = get_smpl_parents()
        parents[0] = 0 # Make root joint's parent itself so that after deduction, the root offsets are 0
        rest_human_offsets = rest_human_jnts.squeeze(0) - rest_human_jnts.squeeze(0)[:, parents, :]

        return rest_human_offsets # 1 X J X 3 

    def filter_data(self, ori_data_dict):
        new_cnt = 0
        new_data_dict = {}
        max_len = 0 
        for k in ori_data_dict:
            curr_data = ori_data_dict[k]
            seq_len = curr_data['head_qpos'].shape[0]
           
            seq_name = curr_data['seq_name']
            if self.use_subset:
                if "CMU" in seq_name:
                    if seq_len >= self.window:
                        new_data_dict[new_cnt] = curr_data 
                        new_cnt += 1 

                    if seq_len > max_len:
                        max_len = seq_len 
            else:
                if seq_len >= self.window:
                    new_data_dict[new_cnt] = curr_data 
                    new_cnt += 1 

                if seq_len > max_len:
                    max_len = seq_len 

        print("The numer of sequences in original data:{0}".format(len(ori_data_dict)))
        print("After filtering, remaining sequences:{0}".format(len(new_data_dict)))
        print("Max length:{0}".format(max_len))

        return new_data_dict 

    def __len__(self):
        return len(self.window_data_dict)

    def cal_normalize_data_input(self):
        global_jpos_list = []
        global_jvel_list = []
        global_rot_6d_list = []
        for index in self.data_dict:
            seq_name = self.data_dict[index]['seq_name']

            seq_root_trans = self.data_dict[index]['trans'] # T X 3 
            seq_root_orient = self.data_dict[index]['root_orient'] # T X 3 
            seq_pose_body = self.data_dict[index]['body_pose'].reshape(-1, 21, 3) # T X 21 X 3

            num_steps = seq_root_trans.shape[0]
            for start_t_idx in range(0, num_steps, self.window//2):
                end_t_idx = start_t_idx + self.window - 1
                if end_t_idx <= num_steps - 1:
                    # import time 
                    # start_time = time.time()
                    query = self.process_window_data(seq_root_trans, seq_root_orient, seq_pose_body, start_t_idx, end_t_idx)
                    # end_time = time.time() 
                    # print("Processing window need:{0}".format(end_time-start_time))

                    curr_global_jpos = query['global_jpos']
                    global_jpos_list.append(curr_global_jpos.reshape(-1, 66).cpu())

                    curr_global_jvel = query['global_jvel']
                    global_jvel_list.append(curr_global_jvel.reshape(-1, 66).cpu())

                    curr_global_rot_6d = query['global_rot_6d']
                    global_rot_6d_list.append(curr_global_rot_6d.reshape(-1, 22*6).cpu())

            # break 

        for_save_global_jpos_data = torch.stack(global_jpos_list).data.cpu().numpy() # N X T X 66 
        for_save_global_jvel_data = torch.stack(global_jvel_list).data.cpu().numpy() # N X T X 66 
        for_save_global_rot_6d_data = torch.stack(global_rot_6d_list).data.cpu().numpy() # N X T X (22*6) 
        self.window_data_dict = {}
        num_seq = for_save_global_jpos_data.shape[0]
        for s_idx in range(num_seq):
            self.window_data_dict[s_idx] = np.concatenate((for_save_global_jpos_data[s_idx], \
            for_save_global_jvel_data[s_idx], for_save_global_rot_6d_data[s_idx]), axis=1)

        global_jpos_mean = for_save_global_jpos_data.reshape(-1, 66).mean(axis=0) # 66
        global_jpos_std = for_save_global_jpos_data.reshape(-1, 66).std(axis=0) # 66 
       
        global_jvel_mean = for_save_global_jvel_data.reshape(-1, 66).mean(axis=0) # 66 
        global_jvel_std = for_save_global_jvel_data.reshape(-1, 66).std(axis=0) # 66 

        mean_dict = {}
        mean_dict['global_jpos_mean'] = global_jpos_mean 
        mean_dict['global_jpos_std'] = global_jpos_std 
        mean_dict['global_jvel_mean'] = global_jvel_mean 
        mean_dict['global_jvel_std'] = global_jvel_std 

        return mean_dict  

    def normalize_jpos(self, ori_jpos):
        # ori_jpos: T X 22 X 3 
        normalized_jpos = (ori_jpos - self.global_jpos_mean.to(ori_jpos.device))/self.global_jpos_std.to(ori_jpos.device)

        return normalized_jpos # T X 22 X 3 

    def de_normalize_jpos(self, normalized_jpos):
        de_jpos = normalized_jpos * self.global_jpos_std.to(normalized_jpos.device) + self.global_jpos_mean.to(normalized_jpos.device)

        return de_jpos # T X 22 X 3 

    def normalize_jvel(self, ori_jvel):
        # ori_jpos: T X 22 X 3 
        normalized_jvel = (ori_jvel - self.global_jvel_mean.to(ori_jvel.device))/self.global_jvel_std.to(ori_jvel.device)

        return normalized_jvel # T X 22 X 3 

    def de_normalize_jvel(self, normalized_jvel):
        de_jvel = normalized_jvel * self.global_jvel_std.to(normalized_jvel.device) + self.global_jvel_mean.to(normalized_jvel.device)

        return de_jvel # T X 22 X 3 

    def process_window_data(self, seq_root_trans, seq_root_orient, seq_pose_body, random_t_idx, end_t_idx):
        window_root_trans = torch.from_numpy(seq_root_trans[random_t_idx:end_t_idx+1]).cuda()
        window_root_orient = torch.from_numpy(seq_root_orient[random_t_idx:end_t_idx+1]).float().cuda()
        window_pose_body  = torch.from_numpy(seq_pose_body[random_t_idx:end_t_idx+1]).float().cuda()

        window_root_rot_mat = transforms.axis_angle_to_matrix(window_root_orient) # T' X 3 X 3 
        window_root_quat = transforms.matrix_to_quaternion(window_root_rot_mat)

        window_pose_rot_mat = transforms.axis_angle_to_matrix(window_pose_body) # T' X 21 X 3 X 3 

        # Generate global joint rotation 
        local_joint_rot_mat = torch.cat((window_root_rot_mat[:, None, :, :], window_pose_rot_mat), dim=1) # T' X 22 X 3 X 3 
        global_joint_rot_mat = local2global_pose(local_joint_rot_mat) # T' X 22 X 3 X 3 
        global_joint_rot_quat = transforms.matrix_to_quaternion(global_joint_rot_mat) # T' X 22 X 4 

        if self.opt.canonicalize_init_head:
            # Canonicalize first frame's facing direction based on global head joint rotation. 
            head_idx = 15 
            global_head_joint_rot_quat = global_joint_rot_quat[:, head_idx, :].data.cpu().numpy() # T' X 4 

            aligned_root_trans, aligned_head_quat, recover_rot_quat = \
            rotate_at_frame_smplh(window_root_trans.data.cpu().numpy()[np.newaxis], \
            global_head_joint_rot_quat[np.newaxis], cano_t_idx=0)
            # BS(1) X T' X 3, BS(1) X T' X 4, BS(1) X 1 X 1 X 4  

            # Apply the rotation to the root orientation 
            cano_window_root_quat = transforms.quaternion_multiply( \
            transforms.quaternion_invert(torch.from_numpy(recover_rot_quat[0, 0]).float().to(\
            window_root_quat.device)).repeat(window_root_quat.shape[0], 1), window_root_quat) # T' X 4 
            cano_window_root_rot_mat = transforms.quaternion_to_matrix(cano_window_root_quat) # T' X 3 X 3 

            cano_local_joint_rot_mat = torch.cat((cano_window_root_rot_mat[:, None, :, :], window_pose_rot_mat), dim=1) # T' X 22 X 3 X 3 
            cano_global_joint_rot_mat = local2global_pose(cano_local_joint_rot_mat) # T' X 22 X 3 X 3 
            
            cano_local_rot_aa_rep = transforms.matrix_to_axis_angle(cano_local_joint_rot_mat) # T' X 22 X 3 

            cano_local_rot_6d = transforms.matrix_to_rotation_6d(cano_local_joint_rot_mat)
            cano_global_rot_6d = transforms.matrix_to_rotation_6d(cano_global_joint_rot_mat)

            # Generate global joint position 
            local_jpos = self.rest_human_offsets.repeat(cano_local_rot_aa_rep.shape[0], 1, 1) # T' X 22 X 3 
            _, human_jnts = quat_fk_torch(cano_local_joint_rot_mat, local_jpos) # T' X 22 X 3 
            human_jnts = human_jnts + torch.from_numpy(aligned_root_trans[0][:, None, :]).float().to(human_jnts.device)# T' X 22 X 3 

            # Move the trajectory based on global head position. Make the head joint to x = 0, y = 0. 
            global_head_jpos = human_jnts[:, head_idx, :] # T' X 3 
            move_to_zero_trans = global_head_jpos[0:1].clone() # 1 X 3
            move_to_zero_trans[:, 2] = 0  
        
            global_jpos = human_jnts - move_to_zero_trans[None] # T' X 22 X 3  

            global_jvel = global_jpos[1:] - global_jpos[:-1] # (T'-1) X 22 X 3 

            query = {}

            query['local_rot_mat'] = cano_local_joint_rot_mat # T' X 22 X 3 X 3 
            query['local_rot_6d'] = cano_local_rot_6d # T' X 22 X 6

            query['global_jpos'] = global_jpos # T' X 22 X 3 
            query['global_jvel'] = torch.cat((global_jvel, \
                torch.zeros(1, 22, 3).to(global_jvel.device)), dim=0) # T' X 22 X 3 
            
            query['global_rot_mat'] = cano_global_joint_rot_mat # T' X 22 X 3 X 3 
            query['global_rot_6d'] = cano_global_rot_6d # T' X 22 X 6
        else:
            curr_seq_pose_aa = torch.cat((window_root_orient[:, None, :], window_pose_body), dim=1) # T' X 22 X 3 
            curr_seq_local_jpos = self.rest_human_offsets.repeat(curr_seq_pose_aa.shape[0], 1, 1) # T' X 22 X 3 

            curr_seq_pose_rot_mat = transforms.axis_angle_to_matrix(curr_seq_pose_aa)
            _, human_jnts = quat_fk_torch(curr_seq_pose_rot_mat, curr_seq_local_jpos)
            human_jnts = human_jnts + window_root_trans[:, None, :] # T' X 22 X 3  

            head_idx = 15 
            # Move the trajectory based on global head position. Make the head joint to x = 0, y = 0. 
            global_head_jpos = human_jnts[:, head_idx, :] # T' X 3 
            move_to_zero_trans = global_head_jpos[0:1].clone() # 1 X 3
            move_to_zero_trans[:, 2] = 0  
        
            global_jpos = human_jnts - move_to_zero_trans[None] # T' X 22 X 3  

            global_jvel = global_jpos[1:] - global_jpos[:-1] # (T'-1) X 22 X 3 

            local_joint_rot_mat = transforms.axis_angle_to_matrix(curr_seq_pose_aa) # T' X 22 X 3 X 3 
            global_joint_rot_mat = local2global_pose(local_joint_rot_mat) # T' X 22 X 3 X 3 

            local_rot_6d = transforms.matrix_to_rotation_6d(local_joint_rot_mat)
            global_rot_6d = transforms.matrix_to_rotation_6d(global_joint_rot_mat)

            query = {}

            query['local_rot_mat'] = local_joint_rot_mat # T' X 22 X 3 X 3 
            query['local_rot_6d'] = local_rot_6d # T' X 22 X 6

            query['global_jpos'] = global_jpos # T' X 22 X 3 
            query['global_jvel'] = torch.cat((global_jvel, \
                torch.zeros(1, 22, 3).to(global_jvel.device)), dim=0) # T' X 22 X 3 
            
            query['global_rot_mat'] = global_joint_rot_mat # T' X 22 X 3 X 3 
            query['global_rot_6d'] = global_rot_6d # T' X 22 X 6

        return query 

    def __getitem__(self, index):
        # index = 0
        data_input = self.window_data_dict[index]
        data_input = torch.from_numpy(data_input).float()

        num_joints = 22
        normalized_jpos = self.normalize_jpos(data_input[:, :num_joints*3].reshape(-1, num_joints, 3)) # T X 22 X 3 
        normalized_jvel = self.normalize_jvel(data_input[:, num_joints*3:num_joints*3*2].reshape(-1, num_joints, 3))
        global_joint_rot = data_input[:, 2*num_joints*3:] # T X (22*6)

        if self.opt.add_velocity_rep:
            new_data_input = torch.cat((normalized_jpos.reshape(-1, 66), normalized_jvel.reshape(-1, 66), \
                        global_joint_rot), dim=1)
        else:
            new_data_input = torch.cat((normalized_jpos.reshape(-1, 66), global_joint_rot), dim=1)

        return new_data_input 
        # T X (22*3+22*3+22*6) range [-1, 1]
