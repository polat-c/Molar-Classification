import cv2
import numpy as np
import torch
import argparse
from scipy.ndimage.filters import gaussian_filter
from torch.nn.functional import interpolate

np.random.seed(0)

class Normalize(object):

    def __call__(self, image):
        mean, std = image.mean(), image.std()
        image = (image - mean) / std
        return image

class Interpolate(object):
    """Reduces size of image"""

    def __init__(self, window_size):
        self.window_size = window_size

    def __call__(self, image):
        image = interpolate(image.unsqueeze(0), size=(self.window_size, self.window_size))
        image = image.squeeze(0)

        return image

class GaussianBlur(object):
    # Implements Gaussian blur as described in the SimCLR paper
    def __init__(self, kernel_size=5, min=0.1, max=2.0, input_channels=3):
        self.min = min
        self.max = max
        # kernel size is set to be 10% of the image height/width
        self.kernel_size = kernel_size
        self.input_channels = input_channels

    def __call__(self, sample):
        sample = np.array(sample)

        # blur the image with a 50% chance
        prob = np.random.random_sample()

        if prob < 0.5:
            sigma = (self.max - self.min) * np.random.random_sample() + self.min
            if self.input_channels == 3:
                sample = cv2.GaussianBlur(sample, (self.kernel_size, self.kernel_size), sigma)
            elif self.input_channels == 1: # in case of dental surgery dataset
                sample = gaussian_filter(sample, sigma=sigma)


        return sample


def cuda(tensor, uses_cuda):
    return tensor.cuda() if uses_cuda else tensor

def str2bool(v):
    # codes from : https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse

    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

