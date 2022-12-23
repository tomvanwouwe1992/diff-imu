# This code is based on https://github.com/openai/guided-diffusion
"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
from utils.fixseed import fixseed
import os
import numpy as np
import torch
from utils.parser_util import edit_args
from utils.model_util import create_model_and_diffusion, load_model_wo_clip
from utils import dist_util
from model.cfg_sampler import ClassifierFreeSampleModel
from data_loaders.get_data import get_dataset_loader
import utils.rotation_conversions as geometry
from data_loaders.humanml.scripts.motion_process import recover_from_ric
from data_loaders import humanml_utils
import data_loaders.humanml.utils.paramUtil as paramUtil
from data_loaders.humanml.utils.plot_script import plot_3d_motion
import shutil


def main():
    model_name = 'model000600040.pt'
    current_working_directory = os.getcwd()
    current_working_directory = os.path.dirname(os.path.dirname(current_working_directory))
    output_directory = os.path.join(current_working_directory, 'diff-imu-output')
    os.chdir(output_directory)
    model_path = os.path.join(output_directory, 'save', 'trying_stuff', model_name)
    args = edit_args(model_path)
    args.model_path = model_path
    args.edit_mode = 'upper_body'
    args.num_samples = 10
    fixseed(args.seed)
    out_path = args.output_dir
    name = os.path.basename(os.path.dirname(args.model_path))
    niter = os.path.basename(args.model_path).replace('model', '').replace('.pt', '')
    max_frames = 196 if args.dataset in ['kit', 'humanml'] else 60
    fps = 12.5 if args.dataset == 'kit' else 20
    dist_util.setup_dist(args.device)
    if out_path == '':
        out_path = os.path.join(os.path.dirname(args.model_path),
                                'edit_{}_{}_{}_seed{}'.format(name, niter, args.edit_mode, args.seed))
        if args.text_condition != '':
            out_path += '_' + args.text_condition.replace(' ', '_').replace('.', '')

    print('Loading dataset...')
    assert args.num_samples <= args.batch_size, \
        f'Please either increase batch_size({args.batch_size}) or reduce num_samples({args.num_samples})'
    # So why do we need this check? In order to protect GPU from a memory overload in the following line.
    # If your GPU can handle batch size larger then default, you can specify it through --batch_size flag.
    # If it doesn't, and you still want to sample more prompts, run this script with different seeds
    # (specify through the --seed flag)
    args.batch_size = args.num_samples  # Sampling a single batch from the testset, with exactly args.num_samples
    data = get_dataset_loader(name=args.dataset, batch_size=args.batch_size, num_frames=max_frames,
                              data_folder=output_directory,
                              split='test',
                              hml_mode='train')
    # in train mode, you get both text and motion.
    # data.fixed_length = n_frames
    total_num_samples = args.num_samples * args.num_repetitions

    print("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args, data)

    print(f"Loading checkpoints from [{args.model_path}]...")
    state_dict = torch.load(args.model_path, map_location='cpu')
    load_model_wo_clip(model, state_dict)

    model = ClassifierFreeSampleModel(model)   # wrapping model with the classifier-free sampler
    model.to(dist_util.dev())
    model.eval()  # disable random masking

    iterator = iter(data)
    input_motions, model_kwargs = next(iterator)
    input_motions = input_motions.to(dist_util.dev())
    texts = [args.text_condition] * args.num_samples
    model_kwargs['y']['text'] = texts
    if args.text_condition == '':
        args.guidance_param = 0.  # Force unconditioned generation

    # add inpainting mask according to args
    assert max_frames == input_motions.shape[-1]
    gt_frames_per_sample = {}
    model_kwargs['y']['inpainted_motion'] = input_motions



    mask = np.full((112,), False)
    # edit based on 6 imu's
    mask[0:6] = np.full((6,), True)
    mask[2*6:3*6] = np.full((6,), True)
    mask[5*6:6*6] = np.full((6,), True)
    mask[7*6:8*6] = np.full((6,), True)
    mask[10 * 6:11 * 6] = np.full((6,), True)
    mask[13 * 6:14 * 6] = np.full((6,), True)




    if args.edit_mode == 'in_between':
        model_kwargs['y']['inpainting_mask'] = torch.ones_like(input_motions, dtype=torch.bool,
                                                               device=input_motions.device)  # True means use gt motion
        for i, length in enumerate(model_kwargs['y']['lengths'].cpu().numpy()):
            start_idx, end_idx = int(args.prefix_end * length), int(args.suffix_start * length)
            gt_frames_per_sample[i] = list(range(0, start_idx)) + list(range(end_idx, max_frames))
            model_kwargs['y']['inpainting_mask'][i, :, :,
            start_idx: end_idx] = False  # do inpainting in those frames
    elif args.edit_mode == 'upper_body':
        model_kwargs['y']['inpainting_mask'] = torch.tensor(mask, dtype=torch.bool,
                                                            device=input_motions.device)  # True is lower body data
        model_kwargs['y']['inpainting_mask'] = model_kwargs['y']['inpainting_mask'].unsqueeze(0).unsqueeze(
            -1).unsqueeze(-1).repeat(input_motions.shape[0], 1, input_motions.shape[2], input_motions.shape[3])

    all_motions = []
    all_lengths = []
    all_text = []
    labels = ['time', 'pelvis_tx', 'pelvis_ty', 'pelvis_tz', 'pelvis_tilt', 'pelvis_list', 'pelvis_rotation',
              'hip_flexion_l', 'hip_flexion_r',
              'hip_adduction_l', 'hip_adduction_r',
              'hip_rotation_l', 'hip_rotation_r',
              'knee_angle_l', 'knee_angle_r',
              'ankle_angle_l', 'ankle_angle_r',
              'subtalar_angle_l', 'subtalar_angle_r',
              'lumbar_extension', 'lumbar_bending', 'lumbar_rotation',
              'arm_flex_l', 'arm_flex_r',
              'arm_add_l', 'arm_add_r',
              'arm_rot_l', 'arm_rot_r',
              'elbow_flex_l', 'elbow_flex_r',
              'pro_sup_l', 'pro_sup_r']
    for rep_i in range(args.num_repetitions):
        print(f'### Start sampling [repetitions #{rep_i}]')

        # add CFG scale to batch
        model_kwargs['y']['scale'] = torch.ones(args.batch_size, device=dist_util.dev()) * args.guidance_param

        sample_fn = diffusion.p_sample_loop

        sample = sample_fn(
            model,
            (args.batch_size, model.njoints, model.nfeats, max_frames),
            clip_denoised=False,
            model_kwargs=model_kwargs,
            skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
            init_image=None,
            progress=True,
            dump_steps=None,
            noise=None,
            const_noise=False,
        )

        save_path = os.path.join(output_directory, 'save', 'trying_stuff', 'sample_' + str(rep_i))
        torch.save(sample, save_path)

        sample = torch.squeeze(sample).cpu()
        print(sample.size())
        sample = sample.permute((0, 2, 1))
        inpainted_motion = model_kwargs['y']['inpainted_motion']
        inpainted_motion_repetition = torch.squeeze(inpainted_motion).cpu()
        inpainted_motion_repetition = inpainted_motion_repetition.permute((0, 2, 1))
        for i in range(sample.size()[0]):
            time = np.reshape(np.linspace(0, 2.95, 60), (60, 1))
            sample_imu = sample[i, :, :14 * 6].numpy()
            sample_pelvis_rotation = 0 * sample[i, :, :3].numpy()  # still wrong
            sample_pelvis_translation = sample[i, :, 14 * 6:14 * 6 + 3].numpy()
            sample_pose = sample[i, :, 14 * 6 + 3:].numpy()

            inpainted_motion_imu = inpainted_motion_repetition[i, :, :14 * 6].numpy()
            inpainted_pelvis_rotation_6D = inpainted_motion_repetition[i, :, :6]
            inpainted_pelvis_rotation_matrix = geometry.rotation_6d_to_matrix(inpainted_pelvis_rotation_6D)
            inpainted_pelvis_rotation_euler = geometry.matrix_to_euler_angles(inpainted_pelvis_rotation_matrix,"YXZ")

            inpainted_motion_pelvis_rotation = inpainted_pelvis_rotation_euler.numpy()  # still wrong
            inpainted_motion_pelvis_translation = inpainted_motion_repetition[i, :, 14 * 6:14 * 6 + 3].numpy()
            inpainted_motion_pose = inpainted_motion_repetition[i, :, 14 * 6 + 3:].numpy()
            print('size of inpainted motion rotation', inpainted_motion_pelvis_rotation.shape)
            print('size of inpainted motion translation', inpainted_motion_pelvis_translation.shape)
            print('size of inpainted motion pose', inpainted_motion_pose.shape)
            data_inpainted_motion = np.concatenate((time, inpainted_motion_pelvis_translation, inpainted_motion_pelvis_rotation, inpainted_motion_pose), 1)
            data = np.concatenate((time, sample_pelvis_translation, sample_pelvis_rotation, sample_pose), 1)
            diff = data-data_inpainted_motion
            numpy2storage(labels, data,
                          os.path.join(output_directory, 'save', 'trying_stuff',
                                       'sample_' + str(rep_i) + '_' + str(i) + '.mot'))
            numpy2storage(labels, data_inpainted_motion,
                          os.path.join(output_directory, 'save', 'trying_stuff',
                                       'sample_inpainted_' + str(rep_i) + '_' + str(i) + '.mot'))

def numpy2storage(labels, data, storage_file):
    assert data.shape[1] == len(labels), "# labels doesn't match columns"
    assert labels[0] == "time"

    f = open(storage_file, 'w')
    f.write('name %s\n' % storage_file)
    f.write('datacolumns %d\n' % data.shape[1])
    f.write('datarows %d\n' % data.shape[0])
    f.write('range %f %f\n' % (np.min(data[:, 0]), np.max(data[:, 0])))
    f.write('endheader \n')

    for i in range(len(labels)):
        f.write('%s\t' % labels[i])
    f.write('\n')

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            f.write('%20.8f\t' % data[i, j])
        f.write('\n')

    f.close()

        # # Recover XYZ *positions* from HumanML3D vector representation
        # if model.data_rep == 'hml_vec':
        #     n_joints = 22 if sample.shape[1] == 263 else 21
        #     sample = data.dataset.t2m_dataset.inv_transform(sample.cpu().permute(0, 2, 3, 1)).float()
        #     sample = recover_from_ric(sample, n_joints)
        #     sample = sample.view(-1, *sample.shape[2:]).permute(0, 2, 3, 1)
        #
        # all_text += model_kwargs['y']['text']
        # all_motions.append(sample.cpu().numpy())
        # all_lengths.append(model_kwargs['y']['lengths'].cpu().numpy())
        #
        # print(f"created {len(all_motions) * args.batch_size} samples")


    # all_motions = np.concatenate(all_motions, axis=0)
    # all_motions = all_motions[:total_num_samples]  # [bs, njoints, 6, seqlen]
    # all_text = all_text[:total_num_samples]
    # all_lengths = np.concatenate(all_lengths, axis=0)[:total_num_samples]
    #
    # if os.path.exists(out_path):
    #     shutil.rmtree(out_path)
    # os.makedirs(out_path)
    #
    # npy_path = os.path.join(out_path, 'results.npy')
    # print(f"saving results file to [{npy_path}]")
    # np.save(npy_path,
    #         {'motion': all_motions, 'text': all_text, 'lengths': all_lengths,
    #          'num_samples': args.num_samples, 'num_repetitions': args.num_repetitions})
    # with open(npy_path.replace('.npy', '.txt'), 'w') as fw:
    #     fw.write('\n'.join(all_text))
    # with open(npy_path.replace('.npy', '_len.txt'), 'w') as fw:
    #     fw.write('\n'.join([str(l) for l in all_lengths]))
    #
    # print(f"saving visualizations to [{out_path}]...")
    # skeleton = paramUtil.kit_kinematic_chain if args.dataset == 'kit' else paramUtil.t2m_kinematic_chain
    #
    # # Recover XYZ *positions* from HumanML3D vector representation
    # if model.data_rep == 'hml_vec':
    #     input_motions = data.dataset.t2m_dataset.inv_transform(input_motions.cpu().permute(0, 2, 3, 1)).float()
    #     input_motions = recover_from_ric(input_motions, n_joints)
    #     input_motions = input_motions.view(-1, *input_motions.shape[2:]).permute(0, 2, 3, 1).cpu().numpy()
    #
    #
    # for sample_i in range(args.num_samples):
    #     caption = 'Input Motion'
    #     length = model_kwargs['y']['lengths'][sample_i]
    #     motion = input_motions[sample_i].transpose(2, 0, 1)[:length]
    #     save_file = 'input_motion{:02d}.mp4'.format(sample_i)
    #     animation_save_path = os.path.join(out_path, save_file)
    #     rep_files = [animation_save_path]
    #     print(f'[({sample_i}) "{caption}" | -> {save_file}]')
    #     plot_3d_motion(animation_save_path, skeleton, motion, title=caption,
    #                    dataset=args.dataset, fps=fps, vis_mode='gt',
    #                    gt_frames=gt_frames_per_sample.get(sample_i, []))
    #     for rep_i in range(args.num_repetitions):
    #         caption = all_text[rep_i*args.batch_size + sample_i]
    #         if caption == '':
    #             caption = 'Edit [{}] unconditioned'.format(args.edit_mode)
    #         else:
    #             caption = 'Edit [{}]: {}'.format(args.edit_mode, caption)
    #         length = all_lengths[rep_i*args.batch_size + sample_i]
    #         motion = all_motions[rep_i*args.batch_size + sample_i].transpose(2, 0, 1)[:length]
    #         save_file = 'sample{:02d}_rep{:02d}.mp4'.format(sample_i, rep_i)
    #         animation_save_path = os.path.join(out_path, save_file)
    #         rep_files.append(animation_save_path)
    #         print(f'[({sample_i}) "{caption}" | Rep #{rep_i} | -> {save_file}]')
    #         plot_3d_motion(animation_save_path, skeleton, motion, title=caption,
    #                        dataset=args.dataset, fps=fps, vis_mode=args.edit_mode,
    #                        gt_frames=gt_frames_per_sample.get(sample_i, []))
    #         # Credit for visualization: https://github.com/EricGuo5513/text-to-motion
    #
    #     all_rep_save_file = os.path.join(out_path, 'sample{:02d}.mp4'.format(sample_i))
    #     ffmpeg_rep_files = [f' -i {f} ' for f in rep_files]
    #     hstack_args = f' -filter_complex hstack=inputs={args.num_repetitions+1}'
    #     ffmpeg_rep_cmd = f'ffmpeg -y -loglevel warning ' + ''.join(ffmpeg_rep_files) + f'{hstack_args} {all_rep_save_file}'
    #     os.system(ffmpeg_rep_cmd)
    #     print(f'[({sample_i}) "{caption}" | all repetitions | -> {all_rep_save_file}]')
    #
    # abs_path = os.path.abspath(out_path)
    # print(f'[Done] Results are at [{abs_path}]')


if __name__ == "__main__":
    main()
