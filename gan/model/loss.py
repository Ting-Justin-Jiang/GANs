import torch
import torch.nn as nn
import numpy as np


class GANLossGenerator(nn.Module):
    def __init__(self):
        super(GANLossGenerator, self).__init__()
        self.loss = nn.BCELoss()

    def forward(self, r_label, f_logit):
        # The labels should be the same shape as f_logit, and generally expects 1s for true labels
        return self.loss(f_logit, r_label)


class GANLossDiscriminator(nn.Module):
    def __init__(self):
        super(GANLossDiscriminator, self).__init__()
        self.loss = nn.BCELoss()

    def forward(self, r_imgs, f_imgs, r_label, f_label, r_logit, f_logit):
        # Real loss
        r_loss = self.loss(r_logit, r_label)
        # Fake loss
        f_loss = self.loss(f_logit, f_label)
        loss_d = (r_loss + f_loss) / 2
        return loss_d


class WassersteinGANLossGenerator(nn.Module):
    def __init__(self):
        super(WassersteinGANLossGenerator, self).__init__()

    def forward(self, f_imgs, discriminator):
        loss = -torch.mean(discriminator(f_imgs))
        return loss


class WassersteinGANLossDiscriminator(nn.Module):
    def __init__(self):
        super(WassersteinGANLossDiscriminator, self).__init__()

    def forward(self, r_imgs, f_imgs, r_label, f_label, r_logit, f_logit):
        loss = -torch.mean(r_logit) + torch.mean(f_logit)
        return loss


class ACGANLossGenerator(nn.Module):
    pass


class ACGANLossDiscriminator(nn.Module):
    pass


class WassersteinGANLossGPDiscriminator(nn.Module):
    def __init__(self):
        super(WassersteinGANLossGPDiscriminator, self).__init__()

    def gp(self, r_imgs, f_imgs, discriminator):
        # Assuming self.D is the discriminator network
        alpha = torch.rand((r_imgs.size(0), 1, 1, 1), dtype=torch.float32, device='cuda')
        interpolates = (alpha * r_imgs + (1 - alpha) * f_imgs).requires_grad_(True)
        d_interpolates = discriminator(interpolates)

        # Create gradient outputs tensor
        grad_outputs = torch.ones(d_interpolates.size(), device=r_imgs.device, requires_grad=False)

        # Calculate gradients
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]

        # Reshape gradients to calculate norm
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def forward(self, r_imgs, f_imgs, r_label, f_label, r_logit, f_logit, discriminator):
        gradient_penalty = self.gp(r_imgs, f_imgs, discriminator)
        loss = -torch.mean(r_logit) + torch.mean(f_logit) + gradient_penalty
        return loss


class DRAGANLossDiscriminator(nn.Module):
    pass


class R1(nn.Module):
    pass


class R2(nn.Module):
    pass


class RLC(nn.Module):
    pass
