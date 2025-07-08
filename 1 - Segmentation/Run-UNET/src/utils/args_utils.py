import argparse
import yaml
import sys
import os
sys.path.append(os.path.dirname(os.getcwd()))
sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.pardir)
    )
)
TRAINING_LOO = r'/home/miguel/GI/0 - Data Exploration & Analysis/UW-Madison/stomach_data_and_masks_preparation/train_loo'
VALIDATION_LOO = r'/home/miguel/GI/0 - Data Exploration & Analysis/UW-Madison/stomach_data_and_masks_preparation/val_loo'


# specifying the training params
def train_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--exp_id',type=str, default='exp/0')
    parser.add_argument('--wandb', action='store_true', help='If true run wandb logger')
    parser.add_argument('--seed',type=int, default=42)
    parser.add_argument('--lr', type=float, default='2e-3') #1e-4 is best
    parser.add_argument('--bs', type=int, default=32) # change if there's memory issues
    parser.add_argument('--epoch', type=int, default=50) # default 30
    parser.add_argument('--train_dir', type=str, default=TRAINING_LOO)
    parser.add_argument('--val_dir', type=str)
    parser.add_argument('--gen_model', type=str, default='') # path to the generator model
    parser.add_argument('--synthetic_real_ratio', type=float, default=0, help='Ratio of synthetic to real data in training')
    parser.add_argument('--unet_architecture', type=str, default='') # architecture to use
    args = parser.parse_args()
    return args

def test_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--save_dir',type=str, default='/home/syurtseven/gsoc/scripts/results')
    parser.add_argument('--seed',type=int, default=31)
    parser.add_argument('--sample_size', type=int, default=30)
    parser.add_argument('--model_path', type=str, default='')
    args = parser.parse_args()
    return args