import argparse
import time

import torch
from datetime import datetime
import numpy as np

parser = argparse.ArgumentParser()


# parser.add_argument('--checkpoint')
# parser.add_argument('--checkpoint', default='./sstgan_p_eth_0.6_with_model.pt', type=str)


def main(args):
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    for k, v in checkpoint['metrics_train'].items():
        print(k)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
