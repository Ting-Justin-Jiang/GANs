import logging
import os
from datetime import datetime

import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from typing import Optional, Tuple
from tqdm import tqdm

from .loss import WassersteinGANLossGPDiscriminator, R1, DRAGANLossDiscriminator, RLC


class GANEngine(object):
    """
    This class implements a GAN wrapper
    """

    def __init__(self,
                 config: dict,
                 generator: nn.Module,
                 discriminator: nn.Module,
                 generator_optimizer: torch.optim.Optimizer,
                 discriminator_optimizer: torch.optim.Optimizer,
                 generator_loss_function: nn.Module,
                 discriminator_loss_function: nn.Module,
                 regularization_loss: Optional[nn.Module] = None) -> None:

        # Save parameters
        self.steps: int = 0
        self.config: dict = config
        self.dataloader: torch.utils = None
        self.generator: nn.Module = generator
        self.discriminator: nn.Module = discriminator
        self.generator_optimizer: torch.optim.Optimizer = generator_optimizer
        self.discriminator_optimizer: torch.optim.Optimizer = discriminator_optimizer
        self.generator_loss_function: nn.Module = generator_loss_function
        self.discriminator_loss_function: nn.Module = discriminator_loss_function
        self.regularization_loss: nn.Module = regularization_loss

        FORMAT = '%(asctime)s - %(levelname)s: %(message)s'
        logging.basicConfig(level=logging.INFO,
                            format=FORMAT,
                            datefmt='%Y-%m-%d %H:%M')

    def prepare_environment(self):
        """
        Use this funciton to prepare function
        """
        os.makedirs(self.config['log_dir'], exist_ok=True)
        os.makedirs(self.config['ckpt_dir'], exist_ok=True)

        # update dir by time
        time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self.config['log_dir'] = os.path.join(self.config['log_dir'], time + f'_{self.config["model_type"]}')
        self.config['ckpt_dir'] = os.path.join(self.config['ckpt_dir'], time + f'_{self.config["model_type"]}')
        os.makedirs(self.config['log_dir'])
        os.makedirs(self.config['ckpt_dir'])

        # create dataloader
        if self.config['dataset'] == "CIFAR10":
            transform = transforms.Compose([
                transforms.Resize(64),  # Resize images to 64x64 for easier processing
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
            os.makedirs('data', exist_ok=True)
            dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
            self.dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=2)
        else:
            # todo custom dataset
            ...

        # model preparation
        if torch.cuda.is_available():
            self.generator.cuda()
            self.discriminator.cuda()

        self.generator.train()
        self.discriminator.train()

    def train(self, instance_noise=True) -> None:
        for e, epoch in enumerate(range(self.config["n_epoch"])):
            progress_bar = tqdm(self.dataloader)
            progress_bar.set_description(f"Epoch {e + 1}")
            for i, data in enumerate(progress_bar):
                imgs = data[0].cuda()
                bs = imgs.size(0)

                # *    Train D        *
                # Reset gradients
                self.generator_optimizer.zero_grad()
                self.discriminator_optimizer.zero_grad()

                # Sample generation
                z = torch.randn(bs, self.config["z_dim"]).cuda()
                r_imgs = imgs.cuda()
                f_imgs = self.generator(z)
                r_label = torch.ones(bs).cuda()
                f_label = torch.zeros(bs).cuda()

                # Discriminator forwarding
                r_logit = self.discriminator(r_imgs)
                f_logit = self.discriminator(f_imgs)

                # Regularization term
                if self.regularization_loss is not None:
                    if isinstance(self.regularization_loss, R1):
                        discriminator_loss: torch.Tensor = self.regularization_loss(
                            r_logit, r_imgs)

                    elif isinstance(self.regularization_loss, RLC):
                        discriminator_loss: torch.Tensor = self.regularization_loss(
                            r_logit, f_logit
                        )
                    else:
                        discriminator_loss: torch.Tensor = self.regularization_loss(
                            f_logit, f_imgs
                        )
                else:
                    discriminator_loss = torch.zeros(1).cuda()

                # Compute loss
                if isinstance(self.discriminator_loss_function, WassersteinGANLossGPDiscriminator):
                    discriminator_loss: torch.Tensor = discriminator_loss + self.discriminator_loss_function(
                        r_logit, f_logit,
                        r_label, f_label,
                        self.discriminator,
                        r_imgs, f_imgs
                    )

                elif isinstance(self.discriminator_loss_function, DRAGANLossDiscriminator):
                    discriminator_loss: torch.Tensor = discriminator_loss + self.discriminator_loss_function(
                        r_logit, f_logit,
                        r_label, f_label,
                        r_imgs, self.discriminator
                    )
                else:
                    discriminator_loss: torch.Tensor = discriminator_loss + self.discriminator_loss_function(
                        r_logit, f_logit,
                        r_label, f_label
                    )

                # Compute gradients
                discriminator_loss.backward()
                # Perform optimization
                self.discriminator_optimizer.step()

                # *    Train G        *
                # Reset gradients
                if self.steps % self.config["n_critic"] == 0:
                    self.generator_optimizer.zero_grad()
                    self.discriminator_optimizer.zero_grad()

                    # Generate some fake images.
                    z = torch.randn(bs, self.config["z_dim"]).cuda()
                    f_imgs = self.generator(z)

                    # Generator forwarding
                    f_logit = self.discriminator(f_imgs)

                    # Compute generator loss
                    generator_loss: torch.Tensor = self.generator_loss_function(f_logit, r_label)

                    # Compute gradients
                    generator_loss.backward()
                    # Perform optimization
                    self.generator_optimizer.step()

                if self.steps % 10 == 0:
                    progress_bar.set_postfix(loss_G=generator_loss.item(), loss_D=discriminator_loss.item())

                self.steps += 1

            # Evaluate generator
            self.generator.eval()
            z_samples = torch.randn(100, self.config["z_dim"]).cuda()

            f_imgs_sample = (self.generator(z_samples).data + 1) / 2.0
            filename = os.path.join(self.config['log_dir'], f'Epoch_{epoch + 1:03d}.jpg')
            torchvision.utils.save_image(f_imgs_sample, filename, nrow=10)
            # logging.info(f'Save some samples to {filename}.')

            # Show some images during training.
            grid_img = torchvision.utils.make_grid(f_imgs_sample.cpu(), nrow=10)
            plt.figure(figsize=(10, 10))
            plt.imshow(grid_img.permute(1, 2, 0))
            plt.show()

            self.generator.train()

            if (e + 1) % 5 == 0 or e == 0:
                # Save the checkpoints.
                torch.save(self.generator.state_dict(), os.path.join(self.config['ckpt_dir'], f'G_{e}.pth'))
                torch.save(self.discriminator.state_dict(), os.path.join(self.config['ckpt_dir'], f'D_{e}.pth'))

        logging.info("Finish training")
