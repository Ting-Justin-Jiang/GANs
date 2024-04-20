import torch.nn as nn


def weights_init(m):
    """
    Weight initialization
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class Generator(nn.Module):
    """
    Implementation of GAN Generator
    """
    def __init__(self, in_dim, feature_dim=64):
        super(Generator, self).__init__()

        # input: (batch, 100)
        self.l1 = nn.Sequential(
            nn.Linear(in_dim, feature_dim * 8 * 4 * 4, bias=False),
            nn.BatchNorm1d(feature_dim * 8 * 4 * 4),
            nn.ReLU()
        )
        self.l2 = nn.Sequential(
            self.dconv_bn_relu(feature_dim * 8, feature_dim * 4),  # (batch, feature_dim * 8, 8, 8)
            self.dconv_bn_relu(feature_dim * 4, feature_dim * 2),  # (batch, feature_dim * 4, 16, 16)
            self.dconv_bn_relu(feature_dim * 2, feature_dim),      # (batch, feature_dim * 2, 32, 32)
        )
        self.l3 = nn.Sequential(
            nn.ConvTranspose2d(feature_dim, 3, kernel_size=5, stride=2,
                               padding=2, output_padding=1, bias=False),
            nn.Tanh()
        )
        self.apply(weights_init)

    def dconv_bn_relu(self, in_dim, out_dim):
        return nn.Sequential(
            nn.ConvTranspose2d(in_dim, out_dim, kernel_size=5, stride=2,
                               padding=2, output_padding=1, bias=False),  # double height and width
            nn.BatchNorm2d(out_dim),
            nn.ReLU(True)
        )

    def forward(self, x):
        y = self.l1(x)
        y = y.view(y.size(0), -1, 4, 4)
        y = self.l2(y)
        y = self.l3(y)
        return y


class Discriminator(nn.Module):
    """
    Implementation of GAN Discriminator
    """
    def __init__(self, in_dim, feature_dim=64, is_critic=False):
        super(Discriminator, self).__init__()

        # input: (batch, 3, 64, 64)
        layers = [
            nn.Conv2d(in_dim, feature_dim, kernel_size=4, stride=2, padding=1),  # (batch, feature_dim, 32, 32)
            nn.LeakyReLU(0.2),
            self.conv_in_lrelu(feature_dim, feature_dim * 2),      # (batch, feature_dim * 2, 16, 16)
            self.conv_in_lrelu(feature_dim * 2, feature_dim * 4),  # (batch, feature_dim * 4, 8, 8)
            self.conv_in_lrelu(feature_dim * 4, feature_dim * 8),  # (batch, feature_dim * 8, 4, 4)
            nn.Conv2d(feature_dim * 8, 1, kernel_size=4, stride=1, padding=0)  # (batch, 1, 1, 1)
        ]

        if not is_critic:
            layers.append(nn.Sigmoid())

        self.l1 = nn.Sequential(*layers)
        self.apply(weights_init)

    def conv_in_lrelu(self, in_dim, out_dim):
        return nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 4, 2, 1),
            nn.InstanceNorm2d(out_dim),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        y = self.l1(x)
        y = y.view(-1)
        return y
