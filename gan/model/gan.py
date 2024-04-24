import logging
import os
from datetime import datetime

import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from typing import Optional, Tuple
from tqdm import tqdm

from .loss import *
from .data import *
from .utils import *


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
        self.config: dict = config
        self.steps: int = 0
        self.z_samples: torch.tensor = torch.randn(100, self.config["z_dim"]).cuda()
        self.dataloader: torch.utils = None

        self.generator: nn.Module = generator
        self.discriminator: nn.Module = discriminator
        self.generator_optimizer: torch.optim.Optimizer = generator_optimizer
        self.discriminator_optimizer: torch.optim.Optimizer = discriminator_optimizer
        self.generator_loss_function: nn.Module = generator_loss_function
        self.discriminator_loss_function: nn.Module = discriminator_loss_function
        self.regularization_loss: nn.Module = regularization_loss

        # Prepare TensorBoard
        time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        tensorboard_dir = os.path.join(self.config['run_dir'], time)
        os.makedirs(tensorboard_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=tensorboard_dir)

        # Prepare logging
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
            self.dataloader = DataLoader(dataset, batch_size=self.config['batch_size'], shuffle=True, num_workers=2)
        elif self.config['dataset'] == "Crypko":
            dataset = get_dataset(os.path.join(self.config["data_dir"], 'faces'))
            self.dataloader = DataLoader(dataset, batch_size=self.config["batch_size"], shuffle=True, num_workers=2)

        # model preparation
        if torch.cuda.is_available():
            self.generator.cuda()
            self.discriminator.cuda()

        self.generator.train()
        self.discriminator.train()

    def train(self) -> None:
        for e, epoch in enumerate(range(self.config["n_epoch"])):
            progress_bar = tqdm(self.dataloader)
            progress_bar.set_description(f"Epoch {e + 1}")
            for i, data in enumerate(progress_bar):
                if self.config["dataset"] == "CIFAR10":
                    imgs = data[0].cuda()
                else:
                    imgs = data.cuda()
                bs = imgs.size(0)

                # *    Train D        *
                # Sample generation
                z = torch.randn(bs, self.config["z_dim"]).cuda()
                r_imgs = imgs.cuda()
                f_imgs = self.generator(z)
                r_label = torch.ones(bs).cuda()
                f_label = torch.zeros(bs).cuda()

                # Discriminator forwarding
                r_logit = self.discriminator(r_imgs)
                f_logit = self.discriminator(f_imgs)

                # Compute loss
                if isinstance(self.discriminator_loss_function, WassersteinGANLossGPDiscriminator):
                    discriminator_loss: torch.Tensor = self.discriminator_loss_function(
                        r_imgs, f_imgs,
                        r_label, f_label,
                        r_logit, f_logit,
                        self.discriminator
                    )
                else:
                    discriminator_loss: torch.Tensor = self.discriminator_loss_function(
                        r_imgs, f_imgs,
                        r_label, f_label,
                        r_logit, f_logit
                    )

                self.discriminator_optimizer.zero_grad()
                discriminator_loss.backward()
                self.discriminator_optimizer.step()

                # Gradient Clipping
                if self.config["model_type"] == "WGAN":
                    for p in self.discriminator.parameters():
                        p.data.clamp_(-self.config["clip_value"], self.config["clip_value"])

                # *    Train G        *
                # Reset gradients
                if self.steps % self.config["n_critic"] == 0:
                    # Generate some fake images.
                    z = torch.randn(bs, self.config["z_dim"]).cuda()
                    f_imgs = self.generator(z)

                    # Generator forwarding
                    f_logit = self.discriminator(f_imgs)

                    # Compute generator loss
                    if isinstance(self.generator_loss_function, WassersteinGANLossGenerator):
                        generator_loss: torch.Tensor = self.generator_loss_function(f_imgs, self.discriminator)
                    else:
                        generator_loss: torch.Tensor = self.generator_loss_function(r_label, f_logit)

                    self.generator_optimizer.zero_grad()
                    generator_loss.backward()
                    self.generator_optimizer.step()

                # Logging
                if self.steps % 10 == 0:
                    if self.config["model_type"] == "WGAN" or "WGAN-GP":
                        # Calculate Wasserstein distance and log it
                        wasserstein_distance = r_logit.mean() - f_logit.mean()
                        self.writer.add_scalar('Wasserstein Distance', wasserstein_distance.item(), self.steps)
                    self.writer.add_scalar('Loss/Generator', abs(generator_loss.item()), self.steps)
                    self.writer.add_scalar('Loss/Discriminator', abs(discriminator_loss.item()), self.steps)

                    progress_bar.set_postfix(loss_G=generator_loss.item(), loss_D=discriminator_loss.item())

                self.steps += 1

            # Evaluate generator
            self.generator.eval()

            f_imgs_sample = (self.generator(self.z_samples).data + 1) / 2.0
            filename = os.path.join(self.config['log_dir'], f'Epoch_{epoch + 1:03d}.jpg')
            torchvision.utils.save_image(f_imgs_sample, filename, nrow=10)

            # Show some images during training
            grid_img = torchvision.utils.make_grid(f_imgs_sample.cpu(), nrow=10)
            plt.figure(figsize=(10, 10))
            plt.imshow(grid_img.permute(1, 2, 0))
            plt.show()

            self.writer.add_images('Generated Images', f_imgs_sample, self.steps, dataformats='NCHW')

            self.generator.train()

            if (e + 1) % 1 == 0 or e == 0:
                # Save the checkpoints
                torch.save(self.generator.state_dict(), os.path.join(self.config['ckpt_dir'], f'G_{e}.pth'))
                torch.save(self.discriminator.state_dict(), os.path.join(self.config['ckpt_dir'], f'D_{e}.pth'))

        logging.info("Finish training")

        # Validation on all the saved model
        generator_checkpoints = load_model_checkpoints(self.config['ckpt_dir'])
        for epoch, path in generator_checkpoints.items():
            logging.info(f'Calculating FID for Epoch: {epoch}, Path: {path}')
            self.generator.load_state_dict(torch.load(path))
            self.writer.add_scalar('FID score', calculate_fid(self.generator, self.dataloader, device='cuda'), epoch)

        logging.info("Finish evaluating")
        self.writer.close()

    def inference(self, generator_path, n_generate=1000, n_output=30, show=False):
        self.generator.load_state_dict(torch.load(generator_path))
        self.generator.cuda()
        self.generator.eval()
        z = torch.randn(n_generate, self.config["z_dim"]).cuda()
        imgs = (self.generator(z).data + 1) / 2.0

        os.makedirs('output', exist_ok=True)
        for i in range(n_generate):
            torchvision.utils.save_image(imgs[i], f'output/{i + 1}.jpg')

        if show:
            row, col = n_output // 10 + 1, 10
            grid_img = torchvision.utils.make_grid(imgs[:n_output].cpu(), nrow=row)
            plt.figure(figsize=(row, col))
            plt.imshow(grid_img.permute(1, 2, 0))
            plt.show()
