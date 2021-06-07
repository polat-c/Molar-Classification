from __future__ import print_function, division
import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import torchvision.transforms.functional as F
from torch.nn.functional import interpolate
from torchvision.datasets import CIFAR10
from sklearn.preprocessing import OneHotEncoder

import SimpleITK as sitk
import utils



def returnCIFAR10data(args): ## train using 2000 images
    dset_dir = args.dset_dir  ## directory that contains 'cifar-10-batches-py'
    batch_size = args.batch_size
    test_portion = args.test_portion
    num_workers = args.num_workers
    train = args.train

    def split_sets(dset, test_portion):
        # Creating data indices for training and validation splits:
        dataset_size = len(dset)
        test_size = int(dataset_size * test_portion)
        train_size = int(dataset_size - test_size)

        train_set, test_set = random_split(dset, [train_size, test_size])

        return train_set, test_set

    #dset = CIFAR10(root=dset_dir, train=train, transform=transforms.ToTensor())

    if not train: # Train the classifier with the fraction of cifar10 test data
        dset = CIFAR10(root=dset_dir, train=train, transform=transforms.ToTensor())
        dset, _ = random_split(dset, [2500, len(dset)-2500])

        train_set, test_set = split_sets(dset, test_portion)

    ####################################################
    ####################################################
    ## this part is only to check if classifier works as intended ##
    else:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        train_set = CIFAR10(root=dset_dir, train=True, transform=transform_train)
        test_set = CIFAR10(root=dset_dir, train=False, transform=transform_test)
    ####################################################
    ####################################################

    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers,
                              pin_memory=True,
                              drop_last=False)

    test_loader = DataLoader(test_set,
                             batch_size=100,
                             shuffle=True,
                             num_workers=num_workers,
                             pin_memory=True,
                             drop_last=False)

    return train_loader, test_loader


############################################################
############################################################
############################################################


class DentalImagingDataset(Dataset):

    def __init__(self, root_dir,
                  csv_file, transform=None, considered_class = 1, ROI_size = 256): # considered_class 1,2 or 3 (see annotations.csv)

        self.annotations = pd.read_csv(csv_file)
        # TODO: handle problematic images better
        self.annotations = self.annotations[(self.annotations.File_name != '20_02.dcm-71_l') & (self.annotations.File_name != '19_11.dcm-20_l')]
        self.annotations.dropna(inplace=True)
        self.annotations.reset_index(drop=True, inplace=True)
        ########################################
        self.annotations = self.annotations[self.annotations['molar_yn'] == 1].iloc[:, [0, 2, 3, 4]]

        self.rootdir = root_dir
        self.transform = transform
        self.considered_class = considered_class
        self.ROI_size = ROI_size

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, item):

        # Image
        file_name = self.annotations.iloc[item, 0] + '.dcm'
        mask_name = file_name[:-4] + '.gipl'
        file_path = os.path.join(self.rootdir, file_name)
        mask_path = os.path.join(self.rootdir, mask_name)
        image = sitk.GetArrayFromImage(sitk.ReadImage(file_path))
        image = image.astype(float).reshape(image.shape[1], image.shape[2])
        mask = sitk.GetArrayFromImage(sitk.ReadImage(mask_path))
        mask = mask.astype(float).reshape(mask.shape[1], mask.shape[2])

        #if self.ROI_size is not None:
        ############### TRAIN WITH ROIS ##################
        # Basic transforms: ROI extraction, to tensor, normalize
        image = utils.extract_ROI(image, mask, self.ROI_size).astype(np.int64)
        ROI_width = 2 * self.ROI_size
        image = np.pad(image, ((0, ROI_width - image.shape[0]), (ROI_width - image.shape[1], 0)), 'constant', constant_values=image.min())
        image = torch.Tensor(image)
        image = image.unsqueeze(0)
        normalize = utils.Normalize()
        image = normalize(image)
        ##################################################

        # #else:
        # ################### TRAIN WITH PATCHES #################
        # # Basic transforms: ROI extraction, to tensor, normalize
        # image = torch.Tensor(image)
        # image = image.unsqueeze(0)
        # image = Normalize()(image)
        # image = Interpolate(window_size=256)(image)
        # ##########################################################

        if self.transform is not None:
            image = self.transform(image)

        # Labels
        if self.considered_class == 1:
            pad = 1
        else:
            pad = 4
        labels = int(self.annotations.iloc[item, self.considered_class]-pad)

        sample = [image, labels]

        return sample

def returnOPGdata(args):

    ## remember to add RandomResizedCrop to the server (solve the problem by running)

    dset_dir = args.dset_dir  ## directory that contains images
    csv_file = args.csv_file  ## csv file with annotations: File_name/ molar_yn/ class_1/ class_2/ class_3
    considered_class = args.considered_class
    ROI_size = args.ROI_size

    batch_size = args.batch_size
    test_portion = args.test_portion
    num_workers = args.num_workers

    def split_sets(dset, test_portion):
        # Creating data indices for training and validation splits:
        dataset_size = len(dset)
        test_size = int(dataset_size * test_portion)
        train_size = int(dataset_size - test_size)

        valid_set, test_set = random_split(dset, [train_size, test_size])

        return valid_set, test_set

    ####################################################
    ####################################################
    dset = DentalImagingDataset(root_dir=dset_dir, csv_file=csv_file,
                                transform=None, considered_class=considered_class, ROI_size=ROI_size)

    train_set, test_set = split_sets(dset, test_portion)

    ####################################################
    ####################################################
    #train_set = DentalImagingDataset(root_dir=dset_dir, csv_file=csv_file,
    #                            transform=None, considered_class=considered_class, ROI_size=ROI_size)
    #test_csv = csv_file[:-21]+'test_annotations.csv'
    #test_set = DentalImagingDataset(root_dir=dset_dir, csv_file=test_csv,
    #                            transform=None, considered_class=considered_class, ROI_size=ROI_size)
    ####################################################
    ####################################################

    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=False,
                              num_workers=num_workers,
                              pin_memory=True,
                              drop_last=False)

    test_loader = DataLoader(test_set,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=num_workers,
                             pin_memory=True,
                             drop_last=False)

    return train_loader, test_loader

class Normalize(object):
    """Normalizes the image tensor"""

    def __call__(self, image):
        mean = 12526.53
        std = 30368.829025877254
        image = F.normalize(image, [mean], [std])

        return image

class Interpolate(object):
    """Reduces size of image"""

    def __init__(self, window_size):
        self.window_size = window_size

    def __call__(self, image):
        image = interpolate(image.unsqueeze(0), size=(self.window_size, self.window_size))
        image = image.squeeze(0)

        return image

