import os 
import math 
from tqdm.auto import tqdm

import torch
from torch import nn, Tensor
import torch.nn.functional as F
import pytorch3d.transforms as transforms 

import einops
from einops import rearrange, reduce
from einops.layers.torch import Rearrange

from inspect import isfunction

from body_model.body_model import BodyModel

from egoego.vis.mesh_motion import get_mesh_verts_faces_for_human_only 
from egoego.data.diffusion_amass_dataset_v2 import AMASSDataset, quat_ik_torch, quat_fk_torch

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype = torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

# sinusoidal positional embeds

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class LearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with learned sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered
        
class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)

class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)

class Conv1dBlock(nn.Module):
    '''
        Conv1d --> GroupNorm --> Mish
    '''

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            Rearrange('batch channels horizon -> batch channels 1 horizon'),
            nn.GroupNorm(n_groups, out_channels),
            Rearrange('batch channels 1 horizon -> batch channels horizon'),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)

class ResidualTemporalBlock(nn.Module):

    def __init__(self, inp_channels, out_channels, embed_dim, horizon, kernel_size=5):
        super().__init__()

        self.blocks = nn.ModuleList([
            Conv1dBlock(inp_channels, out_channels, kernel_size),
            Conv1dBlock(out_channels, out_channels, kernel_size),
        ])

        self.time_mlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(embed_dim, out_channels),
            Rearrange('batch t -> batch t 1'),
        )

        self.residual_conv = nn.Conv1d(inp_channels, out_channels, 1) \
            if inp_channels != out_channels else nn.Identity()

    def forward(self, x, t):
        '''
            x : [ batch_size x inp_channels x horizon ]
            t : [ batch_size x embed_dim ]
            returns:
            out : [ batch_size x out_channels x horizon ]
        '''
        out = self.blocks[0](x) + self.time_mlp(t)
        out = self.blocks[1](out)
        return out + self.residual_conv(x)

class TemporalUnet(nn.Module):

    def __init__(
        self,
        horizon,
        transition_dim,
        dim=32,
        dim_mults=(1, 2, 4, 8),
    ):
        super().__init__()

        dims = [transition_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        print(f'[ models/temporal ] Channel dimensions: {in_out}')

        time_dim = dim
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim * 4),
            nn.Mish(),
            nn.Linear(dim * 4, dim),
        )

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        print(in_out)
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                ResidualTemporalBlock(dim_in, dim_out, embed_dim=time_dim, horizon=horizon),
                ResidualTemporalBlock(dim_out, dim_out, embed_dim=time_dim, horizon=horizon),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))

            if not is_last:
                horizon = horizon // 2

        mid_dim = dims[-1]
        self.mid_block1 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim=time_dim, horizon=horizon)
        self.mid_block2 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim=time_dim, horizon=horizon)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(nn.ModuleList([
                ResidualTemporalBlock(dim_out * 2, dim_in, embed_dim=time_dim, horizon=horizon),
                ResidualTemporalBlock(dim_in, dim_in, embed_dim=time_dim, horizon=horizon),
                Upsample1d(dim_in) if not is_last else nn.Identity()
            ]))

            if not is_last:
                horizon = horizon * 2

        self.final_conv = nn.Sequential(
            Conv1dBlock(dim, dim, kernel_size=5),
            nn.Conv1d(dim, transition_dim//2, 1),
        )

    def forward(self, x, time):
        '''
            x : [ batch x horizon x transition ]
        '''

        x = einops.rearrange(x, 'b h t -> b t h')

        t = self.time_mlp(time)
        h = []

        for resnet, resnet2, downsample in self.downs:
            x = resnet(x, t)
            x = resnet2(x, t)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_block2(x, t)

        for resnet, resnet2, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, t)
            x = resnet2(x, t)
            x = upsample(x)

        x = self.final_conv(x)

        x = einops.rearrange(x, 'b t h -> b h t')
        return x
     
class TCNCondGaussianDiffusion(nn.Module):
    def __init__(
        self,
        d_feats,
        max_timesteps,
        out_dim,
        timesteps = 1000,
        loss_type = 'l1',
        objective = 'pred_noise',
        beta_schedule = 'cosine',
        p2_loss_weight_gamma = 0., # p2 loss weight, from https://arxiv.org/abs/2204.00227 - 0 is equivalent to weight of 1 across time - 1. is recommended
        p2_loss_weight_k = 1,
        add_noisy_head_condition=False,
        add_motion_loss=False,
        only_keep_root_trans=False,
        batch_size=None,
    ):
        super().__init__()

        self.add_noisy_head_condition = add_noisy_head_condition
        self.add_motion_loss = add_motion_loss 
        self.only_keep_root_trans = only_keep_root_trans 

        self.denoise_fn = TemporalUnet(horizon=max_timesteps, transition_dim=d_feats*2, \
        dim=128, dim_mults=(1, 2, 4, 8)) 
        # Input condition and noisy motion, noise level t, predict gt motion
        
        self.objective = objective

        self.seq_len = max_timesteps - 1 
        self.out_dim = out_dim 

        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # calculate p2 reweighting

        register_buffer('p2_loss_weight', (p2_loss_weight_k + alphas_cumprod / (1 - alphas_cumprod)) ** -p2_loss_weight_gamma)

        NUM_BETAS = 16
        self.smpl_batch_size = batch_size * (max_timesteps-1)  
        smplh_path = "/viscam/u/jiamanli/github/hm_interaction/smpl_all_models/smplh_amass"  
        male_bm_path = os.path.join(smplh_path, 'male/model.npz')
        self.male_bm = BodyModel(bm_path=male_bm_path, num_betas=NUM_BETAS, batch_size=self.smpl_batch_size)
        female_bm_path = os.path.join(smplh_path,  'female/model.npz')
        self.female_bm = BodyModel(bm_path=female_bm_path, num_betas=NUM_BETAS, batch_size=self.smpl_batch_size)
        self.bm_dict = {'male' : self.male_bm, 'female' : self.female_bm}
        for p in self.male_bm.parameters():
            p.requires_grad = False
        for p in self.female_bm.parameters():
            p.requires_grad = False 

        self.male_bm = self.male_bm.cuda()
        self.female_bm = self.female_bm.cuda()

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, x_cond, clip_denoised: bool):
        x_all = torch.cat((x, x_cond), dim=-1)
        model_output = self.denoise_fn(x_all, t)

        if self.objective == 'pred_noise':
            x_start = self.predict_start_from_noise(x, t = t, noise = model_output)
        elif self.objective == 'pred_x0':
            x_start = model_output
        else:
            raise ValueError(f'unknown objective {self.objective}')

        if clip_denoised:
            x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_start, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, x_cond, clip_denoised=True):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, x_cond=x_cond, clip_denoised=clip_denoised)
        noise = torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, x_start, cond_mask):
        device = self.betas.device

        b = shape[0]
        x = torch.randn(shape, device=device)
        x_cond = x_start * (1. - cond_mask) + \
            cond_mask * torch.randn_like(x_start).to(x_start.device)

        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            x = self.p_sample(x, torch.full((b,), i, device=device, dtype=torch.long), x_cond)    
            if not self.add_noisy_head_condition: # noiseless condition case 
                x = x_start * (1. - cond_mask) + cond_mask * x  

        return x # BS X T X D

    @torch.no_grad()
    def sample(self, x_start, cond_mask):
        # naive conditional sampling by replacing the noisy prediction with input target data. 
        self.denoise_fn.eval() 
        sample_res = self.p_sample_loop(x_start.shape, \
                x_start, cond_mask)
        # BS X T X D
        self.denoise_fn.train()
        return sample_res  

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    @property
    def loss_fn(self):
        if self.loss_type == 'l1':
            return F.l1_loss
        elif self.loss_type == 'l2':
            return F.mse_loss
        else:
            raise ValueError(f'invalid loss type {self.loss_type}')

    def add_noise_to_data_input(self, x_start):
        # x_start: BS X T X D(22*3+22*6)
        bs, num_steps, _ = x_start.shape

        x_start_jpos = x_start[:, :, :22*3].reshape(bs, num_steps, -1, 3)
        x_start_rot6d = x_start[:, :, 22*3:].reshape(bs, num_steps, -1, 6)
        x_start_rot_mat = transforms.rotation_6d_to_matrix(x_start_rot6d)
        x_start_quat = transforms.matrix_to_quaternion(x_start_rot_mat)

        # Maybe we don't need to add noise... 
        # head_idx = 15 
        # x_start_jpos[:, :, head_idx, :] = x_start_jpos[:, :, head_idx, :] + \
        #     torch.empty(x_start_jpos[:, :, head_idx, :].shape).normal_(mean=0.0, std=0.01).to(x_start_jpos.device)
        # x_start_quat[:, :, head_idx, :] = x_start_quat[:, :, head_idx, :] + \
        #     torch.empty(x_start_quat[:, :, head_idx, :].shape).normal_(mean=0.0, std=0.01).to(x_start_quat.device)
        
        noisy_x_start_rot_mat = transforms.quaternion_to_matrix(x_start_quat)
        noisy_x_start_rot_6d = transforms.matrix_to_rotation_6d(noisy_x_start_rot_mat)

        noisy_x_start = torch.cat((x_start_jpos.reshape(bs, num_steps, -1), \
                    noisy_x_start_rot_6d.reshape(bs, num_steps, -1)), dim=-1)

        return noisy_x_start # BS X T X D(22*3+22*6)

    def p_losses(self, x_start, cond_mask, t, noise=None):
        # x_start: BS X T X D
        # cond_mask: BS X T X D, missing regions are 1, head pose conditioned regions are 0.  
        b, timesteps, d_input = x_start.shape # BS X T X D(3+n_joints*4)
        noise = default(noise, lambda: torch.randn_like(x_start))

        x = self.q_sample(x_start=x_start, t=t, noise=noise) # noisy motion in noise level t. 
 
        if self.add_noisy_head_condition:
            noisy_x_start = self.add_noise_to_data_input(x_start.clone())
            masked_x_input = x 
            x_cond = noisy_x_start * (1. - cond_mask) + cond_mask * torch.randn_like(noisy_x_start).to(noisy_x_start.device)
        else:
            masked_x_input = x_start * (1. - cond_mask) + cond_mask * x # masked noisy motion 
            x_cond = x_start * (1. - cond_mask) + cond_mask * torch.randn_like(x_start).to(x_start.device)

        x_all = torch.cat((masked_x_input, x_cond), dim=-1)
        model_out = self.denoise_fn(x_all, t) 

        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        else:
            raise ValueError(f'unknown objective {self.objective}')

        # print("model_out nan: {0}".format(model_out.isnan().sum()))
        # print("target nan: {0}".format(target.isnan().sum()))
        # print("model_out:{0}".format(model_out))

        # import pdb 
        # pdb.set_trace() 

        if self.add_noisy_head_condition:
            loss = self.loss_fn(model_out, target, reduction = 'none')
            if self.add_motion_loss:
                root_v_loss, feet_contact_loss = self.get_motion_related_loss(model_out, target)
            else:
                root_v_loss = torch.zeros(1).to(target.device)
                feet_contact_loss = torch.zeros(1).to(target.device)
        else:
            loss = self.loss_fn(model_out * cond_mask, target * cond_mask, reduction = 'none')
            root_v_loss = torch.zeros(1).to(target.device)
            feet_contact_loss = torch.zeros(1).to(target.device)

        loss = reduce(loss, 'b ... -> b (...)', 'mean')

        loss = loss * extract(self.p2_loss_weight, t, loss.shape)
        
        return loss.mean(), root_v_loss, feet_contact_loss 

    def fk_human(self, data_input):
        # data_input: BS X T X D (3+22*6)
        bs, num_steps, _ = data_input.shape

        normalized_root_jpos = data_input[:, :, :3] # BS X T X 3 
        global_root_jpos = self.de_normalize_root_jpos(normalized_root_jpos) # BS X T X 3 
        
        global_rot_6d = data_input[:, :, 22*3:].reshape(bs, num_steps, 22, 6)
        global_rot_mat = transforms.rotation_6d_to_matrix(global_rot_6d) # BS X T X 22 X 3 X 3 

        local_rot_mat = quat_ik_torch(global_rot_mat.reshape(-1, 22, 3, 3)) # (BS*T) X 22 X 3 X 3 
        local_rot_mat = local_rot_mat.reshape(bs, num_steps, 22, 3, 3) # BS X T X 22 X 3 X 3 
        local_rot_aa_rep = transforms.matrix_to_axis_angle(local_rot_mat) # BS X T X 22 X 3 
        
        betas = torch.zeros(bs, 10).to(global_root_jpos.device)
        gender = ["male"] * bs 

        mesh_jnts = quat_fk_torch(global_root_jpos, \
        local_rot_aa_rep)
        # BS X T X 22 X 3, BS X T X Nv X 3

        return mesh_jnts

    def assign_min_max_for_normalize(self, global_jpos_min, global_jpos_max):
        self.global_jpos_min = global_jpos_min 
        self.global_jpos_max = global_jpos_max 

    def de_normalize_root_jpos(self, root_jpos):
        # root_jpos: BS X T X 3 
        normalized_jpos = (root_jpos + 1) * 0.5 # [0, 1] range
        root_jpos_min = self.global_jpos_min[:, 0:1, :] # 1 X 1 X 3 
        root_jpos_max = self.global_jpos_max[:, 0:1, :] # 1 X 1 X 3 
        de_jpos = normalized_jpos * (root_jpos_max.to(normalized_jpos.device)-\
        root_jpos_min.to(normalized_jpos.device)) + root_jpos_min.to(normalized_jpos.device)

        return de_jpos # BS X T X 3 

    def get_motion_related_loss(self, pred_out, gt_data):
        # pred_out: BS X T X D (22*3 + 22*6)
        # gt_data: BS X T X D (22*3 + 22*6)
        pred_root_v = pred_out[:, 1:, :3] - pred_out[:, :-1, :3] # BS X (T-1) X 3  
        gt_root_v = gt_data[:, 1:, :3] - gt_data[:, :-1, :3] # BS X (T-1) X 3 
        loss_root_v = F.l1_loss(pred_root_v, gt_root_v).mean()

        pred_jnts, pred_verts = self.fk_human(pred_out) # BS X T X 22 X 3 
        gt_jnts, gt_verts = self.fk_human(gt_data)

        lfoot_idx = 10
        rfoot_idx = 11 
        pred_lfoot_v = pred_out[:, 1:, lfoot_idx] - pred_out[:, :-1, lfoot_idx] # BS X T X 3 
        gt_lfoot_v = gt_data[:, 1:, lfoot_idx] - gt_data[:, :-1, lfoot_idx]
        pred_rfoot_v = pred_out[:, 1:, rfoot_idx] - pred_out[:, :-1, rfoot_idx]
        gt_rfoot_v = gt_data[:, 1:, rfoot_idx] - gt_data[:, :-1, rfoot_idx]
        threshold = 0.002 # meters 
        contact_lfoot_mask = gt_lfoot_v < threshold
        contact_rfoot_mask = gt_rfoot_v < threshold 
        loss_lfoot_contact = F.l1_loss(contact_lfoot_mask*pred_lfoot_v, contact_lfoot_mask*gt_lfoot_v).mean()
        loss_rfoot_contact = F.l1_loss(contact_rfoot_mask*pred_rfoot_v, contact_rfoot_mask*gt_rfoot_v).mean() 
        loss_contact_feet = loss_lfoot_contact + loss_rfoot_contact

        return loss_root_v, loss_contact_feet 

    def forward(self, x_start, cond_mask):
        # x_start: BS X T X D 
        # cond_mask: BS X T X D 
        bs = x_start.shape[0] 
        t = torch.randint(0, self.num_timesteps, (bs,), device=x_start.device).long()
        # print("t:{0}".format(t))
        curr_loss = self.p_losses(x_start, cond_mask, t)

        return curr_loss
        