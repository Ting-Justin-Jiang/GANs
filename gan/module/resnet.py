import torch
import torch.nn as nn
import torch.nn.functional as F


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class ResnetBlock(nn.Module):
    """
    ResNet Block for use in Generator and Discriminator
    """
    def __init__(self, in_features, out_features, stride=1):
        super(ResnetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_features, out_features, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.InstanceNorm2d(out_features)
        self.conv2 = nn.Conv2d(out_features, out_features, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.InstanceNorm2d(out_features)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_features != out_features:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_features, out_features, kernel_size=1, stride=stride, bias=False),
                nn.InstanceNorm2d(out_features)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNetGenerator(nn.Module):
    """
    Implementation of a ResNet-based GAN Generator
    """
    def __init__(self, in_dim, feature_dim=64):
        super(ResNetGenerator, self).__init__()
        self.l1 = nn.Sequential(
            nn.Linear(in_dim, feature_dim * 8 * 4 * 4, bias=False),
            nn.BatchNorm1d(feature_dim * 8 * 4 * 4),
            nn.ReLU()
        )

        self.l2 = nn.Sequential(
            self.dconv_bn_relu(feature_dim * 8, feature_dim * 4),
            ResnetBlock(feature_dim * 4, feature_dim * 4),  # Add ResNet block
            self.dconv_bn_relu(feature_dim * 4, feature_dim * 2),
            ResnetBlock(feature_dim * 2, feature_dim * 2),
            self.dconv_bn_relu(feature_dim * 2, feature_dim),
            ResnetBlock(feature_dim, feature_dim)
        )

        self.l3 = nn.Sequential(
            nn.ConvTranspose2d(feature_dim, 3, kernel_size=5, stride=2, padding=2, output_padding=1, bias=False),
            nn.Tanh()
        )
        self.apply(weights_init)

    def dconv_bn_relu(self, in_dim, out_dim):
        return nn.Sequential(
            nn.ConvTranspose2d(in_dim, out_dim, kernel_size=5, stride=2, padding=2, output_padding=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.l1(x)
        x = x.view(-1, 8 * 64, 4, 4)
        x = self.l2(x)
        x = self.l3(x)
        return x


class ResNetDiscriminator(nn.Module):
    """
    Implementation of a ResNet-based GAN Discriminator
    """
    def __init__(self, in_dim, feature_dim=64, is_critic=False):
        super(ResNetDiscriminator, self).__init__()
        self.l1 = nn.Sequential(
            nn.Conv2d(in_dim, feature_dim, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            ResnetBlock(feature_dim, feature_dim * 2, stride=2),
            ResnetBlock(feature_dim * 2, feature_dim * 4, stride=2),
            ResnetBlock(feature_dim * 4, feature_dim * 8, stride=2),
            nn.Conv2d(feature_dim * 8, 1, kernel_size=4, stride=1, padding=0)  # No sigmoid here
        )

        if not is_critic:
            self.l1.add_module('sigmoid', nn.Sigmoid())  # Apply sigmoid conditionally

        self.apply(weights_init)

    def forward(self, x):
        x = self.l1(x)
        x = x.view(-1)
        return x

