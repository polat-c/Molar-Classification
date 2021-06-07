import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models



class ResNet(nn.Module):

    def __init__(self, in_channels=3, cifar=True, size=50, n_classes=10, pretrained=False):
        super(ResNet, self).__init__()

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
        #self.l1 = nn.Linear(num_ftrs, n_classes)

        # 1024-unit MLP
        self.l1 = nn.Linear(num_ftrs, 1024)
        self.l2 = nn.Linear(1024, n_classes)

    def forward(self, x):

        x = self.features(x)
        x = x.squeeze()
        x = self.l1(x)

        x = F.relu(x)
        x = self.l2(x)

        return x

    def set_writer(self, writer):
        """ Set writer, used for bookkeeping."""
        self.writer = writer

def criterion(x, labels):
    c_entropy = F.cross_entropy(x, labels)

    return c_entropy

######################################################
######################################################
######################################################


class ResNet_custom(nn.Module):

    def __init__(self, in_channels=3, cifar=True, size=50, n_classes=10, pretrained=False):
        super(ResNet_custom, self).__init__()

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
        self.l1 = nn.Linear(num_ftrs, n_classes)

    def forward(self, x):

        x = self.features(x)
        x = x.squeeze()
        x = self.l1(x)

        return x

    def set_writer(self, writer):
        """ Set writer, used for bookkeeping."""
        self.writer = writer


######################################################
######################################################
######################################################


class ResNetXIn(nn.Module):
    '''Resnet implementation accepting inputs of varied sizes'''

    def __init__(self, in_channels = 3, cifar = True, size = 50, n_classes = 10, pretrain = False):
    #def __init__(self, size, pretrain, in_channels=1):
        super(ResNetXIn, self).__init__()

        # bring resnet
        if size == 18:
            self.model = models.resnet18(pretrained=pretrain)
        if size == 34:
            self.model = models.resnet34(pretrained=pretrain)
        if size == 50:
            self.model = models.resnet50(pretrained=pretrain)
        if size == 101:
            self.model = models.resnet101(pretrained=pretrain)
        if size == 152:
            self.model = models.resnet152(pretrained=pretrain)

        # your case
        self.model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        #self.model.fc = nn.Linear(self.model.fc.in_features, n_classes)

    def forward(self, x):
        return self.model(x)

    def set_writer(self, writer):
        """ Set writer, used for bookkeeping."""
        self.writer = writer





