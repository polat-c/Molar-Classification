import argparse
import numpy as np
import torch
from utils import str2bool
from solver import classification_Solver

def main(args):
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    solver = classification_Solver(args)

    solver.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='toy-AE')

    parser.add_argument('--model', default='ResNet', type=str, help='Model to evaluate, from list []')
    parser.add_argument('--dset_dir', default='../', type=str, help='Directory of the data')
    #parser.add_argument('--config_file', default='cifar_config.yaml', type=str, help='Path of the config file')
    #parser.add_argument('--train_mode', default=True, type=bool, help='Perform training if True') # if set to False, samle images from test set of cifar
    parser.add_argument('--batch_size', default=64, type=int, help='Number of images to apply grad descent on each iter')
    parser.add_argument('--test_portion', default=0.2, type=float, help='test set size')
    parser.add_argument('--num_workers', default=1, type=int, help='dataloader num_workers')
    parser.add_argument('--train_mode', default=True, type=bool, help='If True, use model for training')
    parser.add_argument('--epochs', default=300, type=int, help='Number of epochs for training')
    parser.add_argument('--learning_rate', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.8, help='Learning rate decay.')

    parser.add_argument('--in_channels', default=3, type=int, help='Number of input image channels')
    parser.add_argument('--n_classes', default=10, type=int, help='Amount of classes')
    parser.add_argument('--pretrained', default=False, type=bool, help='Pretrained weights on ImageNet will be used if set to True')

    parser.add_argument('--dataset', default='cifar10', type=str, help='Name of the dataset [cifar10, OPG]')
    parser.add_argument('--cuda', default=True, type=str2bool, help='enable cuda')
    parser.add_argument('--seed', default=1, type=int, help='random seed')
    parser.add_argument('--ckpt_dir', default='../classification_checkpoints', type=str, help='checkpoint directory')
    parser.add_argument('--ckpt_name', default='best.pth', type=str,
                        help='load previous checkpoint. insert checkpoint filename')
    parser.add_argument('--output_dir', default='../classification_outputs', type=str, help='output directory')
    parser.add_argument('--exp_name', default='exp0', type=str, help='name of the experiment')

    parser.add_argument('--eval_step', default=5, type=int,
                        help='number of iterations after which full model evaluation occurs')
    parser.add_argument('--gather_step', default=5, type=int,
                        help='number of iterations after which data is gathered for tensorboard')

    ## specials for Cifar dataset
    parser.add_argument('--input_shape', default=(32, 32, 3), type=tuple, help='shape of the input images')

    ## specials for OPG dataset
    parser.add_argument('--csv_file', default='annotations.csv', type=str, help='File_name/ molar_yn/ class_1/ class_2/ class_3')
    parser.add_argument('--labeled', default=True, type=bool, help='If labels of the data exist on the csv file') # This has to be True constantly
    parser.add_argument('--window_size', default=256, type=int, help='Window size for Interpolation')
    parser.add_argument('--considered_class', default=1, type=int, help='Class to consider for OPG dataset, see annotations.csv')

    ## special for classification
    parser.add_argument('--weights', default=None, type=str, help='Path to the weights')
    parser.add_argument('--test_portion_clf', default=2500, type=int, help='Amount of test images to use to train the classifier') # default for cifar dataset
    parser.add_argument('--use_test_to_train', default=True, type=bool, help='True If dataset=cifar and use test images to train the network')


    args = parser.parse_args()

    main(args)