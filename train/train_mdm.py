# This code is based on https://github.com/openai/guided-diffusion
"""
Train a diffusion model on images.
"""

import os
import json
import shutil
from utils.fixseed import fixseed
from utils.parser_util import train_args
from utils import dist_util
from train.training_loop import TrainLoop
from data_loaders.get_data import get_dataset_loader
from utils.model_util import create_model_and_diffusion
from train.train_platforms import ClearmlPlatform, TensorboardPlatform, NoPlatform  # required for the eval operation

def main():
    args = train_args()
    args.seed = 10
    args.dataset = 'CMU'
    args.batch_size = 256
    args.train_platform_type = 'TensorboardPlatform'
    args.overwrite = True
    args.save_interval = 10000
    # args.resume_checkpoint = os.path.join(os.getcwd(),'save','trying_stuff_2','model000015000.pt')
    current_working_directory = os.getcwd()
    current_working_directory = os.path.dirname(os.path.dirname(current_working_directory))
    output_directory = os.path.join(current_working_directory,'diff-imu-output')
    os.chdir(output_directory)
    if os.path.exists(os.path.join(os.getcwd(),'save','trying_stuff')):
        shutil.rmtree(os.path.join(os.getcwd(),'save','trying_stuff'))
    args.save_dir = 'save/trying_stuff'

    fixseed(args.seed)
    train_platform_type = eval(args.train_platform_type)
    train_platform = train_platform_type(args.save_dir)
    train_platform.report_args(args, name='Args')

    if args.save_dir is None:
        raise FileNotFoundError('save_dir was not specified.')
    elif os.path.exists(args.save_dir) and not args.overwrite:
        raise FileExistsError('save_dir [{}] already exists.'.format(args.save_dir))
    elif not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    args_path = os.path.join(args.save_dir, 'args.json')
    with open(args_path, 'w') as fw:
        json.dump(vars(args), fw, indent=4, sort_keys=True)

    dist_util.setup_dist(args.device)

    print("creating data loader...")
    data = get_dataset_loader(name=args.dataset, batch_size=args.batch_size, num_frames=args.num_frames, data_folder = output_directory)

    print("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args, data)
    model.to(dist_util.dev())
    model.rot2xyz.smpl_model.eval()

    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters_wo_clip()) / 1000000.0))
    print("Training...")
    TrainLoop(args, train_platform, model, diffusion, data).run_loop()
    train_platform.close()

if __name__ == "__main__":
    main()
