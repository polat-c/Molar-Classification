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

from utils import GaussianBlur, Interpolate, Normalize
import SimpleITK as sitk
#from sklearn.preprocessing import OneHotEncoder
#from scipy.ndimage.measurements import center_of_mass


class SimCLRDataTransform(object): # needed in order to get both transformed and not-transformed data
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, sample):
        xi = transforms.ToTensor()(sample) # this can be changed by again applying a random transformation
        xj = self.transform(sample)
        return xi, xj


def returnCIFAR10data(args):
    dset_dir = args.dset_dir  ## directory that contains 'cifar-10-batches-py'
    batch_size = args.batch_size
    test_portion = args.test_portion
    num_workers = args.num_workers
    train = args.train_mode
    input_shape = args.input_shape
    test_portion_clf = args.test_portion_clf
    use_test_to_train = args.use_test_to_train

    if not train:
        use_test_to_train = False

    def split_sets(dset, test_portion):
        # Creating data indices for training and validation splits:
        dataset_size = len(dset)
        test_size = int(dataset_size * test_portion)
        train_size = int(dataset_size - test_size)

        valid_set, test_set = random_split(dset, [train_size, test_size])

        return valid_set, test_set

    ## datatransforms as described in the SimCLR paper for CIFAR dataset
    color_jitter = transforms.ColorJitter(0.8, 0.8, 0.8 , 0.2)
    data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=input_shape[0]),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.RandomApply([color_jitter], p=0.5),
                                          transforms.RandomGrayscale(p=0.2),
                                          #GaussianBlur(kernel_size=int(0.1 * input_shape[0])),
                                          transforms.ToTensor()])

    if not use_test_to_train:
        dset = CIFAR10(root=dset_dir, train=train, transform=SimCLRDataTransform(data_transforms))
    else:
        dset = CIFAR10(root=dset_dir, train=False, transform=SimCLRDataTransform(data_transforms))
        dset, _ = random_split(dset, [test_portion_clf, len(dset)-test_portion_clf])

    train_set, test_set = split_sets(dset, test_portion)

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

#################################################################################
#################################################################################
#################################################################################

class DentalImagingDataset(Dataset):

    def __init__(self, root_dir,
                  csv_file, transform=None, labeled=False, considered_class = 1): # considered_class 1,2 or 3 (see annotations.csv)

        if csv_file is not None and labeled:
            self.annotations = pd.read_csv(csv_file)
            self.annotations = self.annotations[self.annotations['molar_yn'] == 1].iloc[:, [0, 2, 3, 4]]
        elif csv_file is not None:
            self.annotations = pd.read_csv(csv_file)

        self.rootdir = root_dir
        self.transform = transform
        self.labeled = labeled
        self.considered_class = considered_class

        #if labeled:
        #    self.one_hot_enc = OneHotEncoder()
        #    self.one_hot_enc.fit(self.annotations.iloc[:, 1:].values)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, item):

        # Image
        file_name = self.annotations.iloc[item, 0] + '.dcm'
        file_path = os.path.join(self.rootdir, file_name)
        image = sitk.GetArrayFromImage(sitk.ReadImage(file_path))
        image = image.astype(float).reshape(image.shape[1], image.shape[2])

        if self.transform is not None:
            image = self.transform(image)

        # Labels
        if self.labeled:
            labels = torch.Tensor(self.annotations.iloc[item, self.considered_class+1], dtype=int)
            #labels = self.one_hot_enc.transform(labels).toarray()
            sample = [image, labels]
        else:
            sample = [image, 0]

        return sample

def returnOPGdata(args):

    ## remember to add RandomResizedCrop to the server (solve the problem by running)

    dset_dir = args.dset_dir  ## directory that contains images
    csv_file = args.csv_file  ## csv file with annotations: File_name/ molar_yn/ class_1/ class_2/ class_3
    labeled = args.labeled  ## if labeled == false : the output of dataloaders are [images, 0]
    batch_size = args.batch_size
    test_portion = args.test_portion
    num_workers = args.num_workers
    window_size = args.window_size
    if labeled:
        considered_class = args.considered_class
    else:
        considered_class = 1

    def split_sets(dset, test_portion):
        # Creating data indices for training and validation splits:
        dataset_size = len(dset)
        test_size = int(dataset_size * test_portion)
        train_size = int(dataset_size - test_size)

        valid_set, test_set = random_split(dset, [train_size, test_size])

        return valid_set, test_set

    ## datatransforms for OPG dataset
    data_transforms = transforms.Compose([transforms.ToTensor(),
                                          Normalize(),
                                          Interpolate(window_size=window_size),
                                          transforms.RandomResizedCrop(size=window_size, scale=(0.6,1.0)),
                                          transforms.RandomHorizontalFlip(),
                                          GaussianBlur(input_channels=1)])

    dset = DentalImagingDataset(root_dir=dset_dir, csv_file=csv_file,
                                transform=SimCLRDataTransform(data_transforms),
                                labeled=labeled, considered_class=considered_class)

    train_set, test_set = split_sets(dset, test_portion)

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


