import argparse
import numpy as np
import torch
from utils import str2bool
from solver import SimCLR_Solver

def main(args):
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    solver = SimCLR_Solver(args)

    solver.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='toy-AE')

    parser.add_argument('--model', default='SimCLR', type=str, help='Model to evaluate, from list []')
    parser.add_argument('--dset_dir', default='../', type=str, help='Directory of the data')
    parser.add_argument('--config_file', default='./SimCLR_configs/cifar_config.yaml', type=str, help='Path of the config file')
    parser.add_argument('--train_mode', default=True, type=bool, help='Perform training if True') # if set to False, samle images from test set of cifar
    #parser.add_argument('--batch_size', default=64, type=int, help='Number of images to apply grad descent on each iter')
    #parser.add_argument('--test_portion', default=0.2, type=float, help='test set size')
    #parser.add_argument('--num_workers', default=1, type=int, help='dataloader num_workers')
    #parser.add_argument('--train_mode', default=True, type=bool, help='If True, use model for training')
    parser.add_argument('--dataset', default='cifar10', type=str, help='Name of the dataset [cifar10, OPG]')
    parser.add_argument('--cuda', default=True, type=str2bool, help='enable cuda')
    parser.add_argument('--seed', default=1, type=int, help='random seed')
    parser.add_argument('--ckpt_dir', default='../checkpoints', type=str, help='checkpoint directory')
    parser.add_argument('--ckpt_name', default='best.pth', type=str,
                        help='load previous checkpoint. insert checkpoint filename')
    parser.add_argument('--output_dir', default='../outputs', type=str, help='output directory')
    parser.add_argument('--exp_name', default='exp0', type=str, help='name of the experiment')

    ## specials for OPG dataset
    parser.add_argument('--csv_file', default='annotations.csv', type=str, help='File_name/ molar_yn/ class_1/ class_2/ class_3')
    parser.add_argument('--labeled', default=False, type=bool, help='If labels of the data exist on the csv file')
    parser.add_argument('--window_size', default=256, type=int, help='Window size for Interpolation')


    args = parser.parse_args()

    main(args)