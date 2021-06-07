import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from loss import NTXentLoss


class SimCLR(nn.Module):

    def __init__(self, in_channels=3, cifar=True, size=50, out_dim=256, backbone='resnet', pretrained=False): # add other possible backbones like darknet
        super(SimCLR, self).__init__()

        # ResNet backbone
        if backbone == 'resnet':
            if size == 18:
                self.model = models.resnet18(pretrained=False)
            elif size == 50:
                self.model = models.resnet50(pretrained=False)

        num_ftrs = self.model.fc.in_features

        # changing first layers, since CIFAR images are small compared to ones in other datasets
        if cifar:
            self.model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=3, bias=False)
            del self.model.maxpool
        else:
            self.model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # to compute h(x)
        self.features = nn.Sequential(*list(self.model.children())[:-1])  # all backbone without fc layer

        # projection MLP, to compute z(h)
        self.l1 = nn.Linear(num_ftrs, num_ftrs)
        self.l2 = nn.Linear(num_ftrs, out_dim)

    def forward(self, x):
        h = self.features(x)
        h = h.squeeze()

        x = self.l1(h)
        x = F.relu(x)
        x = self.l2(x)
        return h, x

    def set_writer(self, writer):
        """ Set writer, used for bookkeeping."""
        self.writer = writer

def SimCLR_criterion(zis, zjs, use_cuda=True, batch_size=64, temperature=0.5):

    criterion = NTXentLoss(use_cuda=use_cuda, batch_size=batch_size, temperature=temperature)
    loss = criterion(zis, zjs)

    return loss

#################################################################################
#################################################################################
#################################################################################

class ResNet(nn.Module):

    def __init__(self, in_channels=3, cifar=True, size=50, n_classes=2, pretrained=False, vanilla=True):
        super(ResNet, self).__init__()

        self.vanilla = vanilla

        if size == 18:
            self.model = models.resnet18(pretrained=pretrained)
        elif size == 50:
            self.model = models.resnet50(pretrained=pretrained)

        if cifar:
            self.model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=3, bias=False)
            del self.model.maxpool
        else:
            self.model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        num_ftrs = self.model.fc.in_features
        self.features = nn.Sequential(*list(self.model.children())[:-1])  # all backbone without fc layer
        self.l1 = nn.Linear(num_ftrs, num_ftrs)
        self.l2 = nn.Linear(num_ftrs, n_classes)

    def forward(self, x):

        if self.vanilla:
            x = self.features(x)
            x = x.squeeze()
            x = self.l2(x)
        else:
            x = self.features(x)
            x = x.squeeze()
            x = self.l1(x)
            x = F.relu(x)
            x = self.l2(x)

        return x

    def set_writer(self, writer):
        """ Set writer, used for bookkeeping."""
        self.writer = writer

def classification_criterion(x, labels):
    c_entropy = F.cross_entropy(x, labels)

    return c_entropy






