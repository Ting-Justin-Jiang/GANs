import torch
import torch.optim as optim
import torch.nn as nn

from gan.model import GANEngine
from gan.model.loss import *
from gan.module import Discriminator, Generator


def main(config):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize models
    generator = Generator(in_dim=config["z_dim"]).to(device)
    discriminator = Discriminator(in_dim=3).to(device)

    # Optimizers
    generator_optimizer = optim.Adam(generator.parameters(), lr=config["lr_generator"], betas=(config["beta1"], 0.999))
    discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=config["lr_discriminator"], betas=(config["beta1"], 0.999))

    # Loss functions
    generator_loss_function = GANLossGenerator()
    discriminator_loss_function = GANLossDiscriminator()

    # Initialize GANEngine
    gan_engine = GANEngine(config=config,
                           generator=generator,
                           discriminator=discriminator,
                           generator_optimizer=generator_optimizer,
                           discriminator_optimizer=discriminator_optimizer,
                           generator_loss_function=generator_loss_function,
                           discriminator_loss_function=discriminator_loss_function)

    # Prepare environment and train
    gan_engine.prepare_environment()
    gan_engine.train()


if __name__ == "__main__":
    config = {
        "n_epoch": 200,
        "z_dim": 100,
        "model_type": "DCGAN",
        "log_dir": "./logs",
        "ckpt_dir": "./checkpoints",
        "batch_size": 128,
        "dataset": "CIFAR10",
        "lr_generator": 0.0002,
        "lr_discriminator": 0.0002,
        "beta1": 0.5,
        "n_critic": 2,
    }

    main(config)
