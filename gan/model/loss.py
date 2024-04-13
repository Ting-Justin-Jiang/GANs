import torch
import torch.nn as nn


class GANLossGenerator(nn.Module):
    def __init__(self):
        super(GANLossGenerator, self).__init__()
        self.loss = nn.BCELoss()

    def forward(self, f_logit, r_label):
        # The labels should be the same shape as f_logit, and generally expects 1s for true labels
        return self.loss(f_logit, r_label)


class GANLossDiscriminator(nn.Module):
    def __init__(self):
        super(GANLossDiscriminator, self).__init__()
        self.loss = nn.BCELoss()

    def forward(self, r_logit, f_logit, r_label, f_label):
        # Real loss
        r_loss = self.loss(r_logit, r_label)
        # Fake loss
        f_loss = self.loss(f_logit, f_label)
        # Average losses for the total discriminator loss
        loss_d = (r_loss + f_loss) / 2
        return loss_d


class ACGANLossGenerator(nn.Module):
    pass


class ACGANLossDiscriminator(nn.Module):
    pass


class WassersteinGANLossGenerator(nn.Module):
    pass


class WassersteinGANLossDiscriminator(nn.Module):
    pass


class WassersteinGANLossGPDiscriminator(nn.Module):
    pass


class DRAGANLossDiscriminator(nn.Module):
    pass


class R1(nn.Module):
    pass


class R2(nn.Module):
    pass


class RLC(nn.Module):
    pass
