from torch.utils.data import DataLoader

from sstgan.data.trajectories import TrajectoryDataset, seq_collate
from sstgan.utils import relative_to_abs, get_dset_path
import os
import argparse
import random
parser = argparse.ArgumentParser()
# Dataset options
parser.add_argument('--dataset_name', default='eth', type=str)
parser.add_argument('--delim', default='\t')
parser.add_argument('--loader_num_workers', default=4, type=int)
parser.add_argument('--obs_len', default=8, type=int)
parser.add_argument('--pred_len', default=12, type=int)
parser.add_argument('--skip', default=1, type=int)
# Optimization
parser.add_argument('--batch_size', default=64, type=int)  # 64

def data_loader(args, path):
    dset = TrajectoryDataset(
        path,
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        skip=args.skip,
        delim=args.delim)
    train_select_dset = select_data(list(dset))

    loader = DataLoader(
        train_select_dset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.loader_num_workers,
        collate_fn=seq_collate)
    return dset, loader


def select_data(data):
    num_samples = int(len(data) * 0.2)
    selected_data = random.sample(data, num_samples)
    return selected_data


if __name__ == '__main__':
    args = parser.parse_args()
    train_path = get_dset_path(args.dataset_name, "train")
    val_path = get_dset_path(args.dataset_name, "val")
    train_dset, train_loader = data_loader(args, train_path)
