import argparse
import numpy as np
import torch
from solver import Solver

def main(args):
    seed = 1
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    solver = Solver(args)

    solver.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='toy-AE')

    #parser.add_argument('--config_file', default='./cifar_configs/config.yaml', type=str, help='Path to config file')
    parser.add_argument('--config_file', default='./OPG_configs/config.yaml', type=str, help='Path to config file')

    args = parser.parse_args()

    main(args)



