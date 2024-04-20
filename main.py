import argparse

import torch.optim as optim

from gan.model import GANEngine
from gan.model.loss import *
from gan.module import Discriminator, Generator
from gan.model.utils import same_seeds

VERSION = {
    'GAN': {
        'base': "DC",
        'optimizer': "Adam",
        'generator_loss_function': GANLossGenerator(),
        'discriminator_loss_function': GANLossDiscriminator(),
        'is_critic': False
    },
    'WGAN': {
        'base': "DC",
        'optimizer': "RMSProp",
        'generator_loss_function': WassersteinGANLossGenerator(),
        'discriminator_loss_function': WassersteinGANLossDiscriminator(),
        'is_critic': True
    },
    'WGAN-GP': {
        'base': "DC",
        'optimizer': "Adam",
        'generator_loss_function': WassersteinGANLossGenerator(),
        'discriminator_loss_function': WassersteinGANLossGPDiscriminator(),
        'is_critic': True
    }
}


def main(config):
    # Set version
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    version_dict = VERSION[config["model_type"]]

    # Initialize models
    if version_dict['base'] == "DC":
        generator = Generator(in_dim=config["z_dim"], feature_dim=config["image_size"]).to(device)
        discriminator = Discriminator(in_dim=3, feature_dim=config["image_size"], is_critic=version_dict["is_critic"]).to(device)
        print(f"Generator class: {generator.__class__.__name__}")
        print(f"Discriminator class: {discriminator.__class__.__name__}")
    else:
        return

    # Optimizers
    if version_dict['optimizer'] == "Adam":
        generator_optimizer = optim.Adam(generator.parameters(), lr=config["lr_generator"], betas=(config["beta1"], 0.999))
        discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=config["lr_discriminator"], betas=(config["beta1"], 0.999))
    elif version_dict['optimizer'] == "RMSProp" and config['model_type'] == "WGAN":
        generator_optimizer = optim.RMSprop(generator.parameters(), lr=config["lr_generator"])
        discriminator_optimizer = optim.RMSprop(discriminator.parameters(), lr=config["lr_discriminator"])
    else:
        return
    print(f"Generator Optimizer class: {generator_optimizer.__class__.__name__}")
    print(f"Discriminator Optimizer class: {discriminator_optimizer.__class__.__name__}")

    # Loss functions
    generator_loss_function = version_dict['generator_loss_function']
    discriminator_loss_function = version_dict['discriminator_loss_function']
    print(f"Generator Loss Function class: {type(generator_loss_function).__name__}")
    print(f"Discriminator Loss Function class: {type(discriminator_loss_function).__name__}")

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", choices=VERSION.keys(), default="WGAN-GP")
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--z_dim", type=int, default=100)
    parser.add_argument("--n_epoch", type=int, default=50)
    parser.add_argument("--dataset", type=str, default="Crypko", choices=["CIFAR10", "Crypko"])
    parser.add_argument("--seed", type=int, default=2024)

    parser.add_argument("--lr_generator", type=float, default=1e-4)
    parser.add_argument("--lr_discriminator", type=float, default=1e-4)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--n_critic", type=int, default=2)
    parser.add_argument("--clip_value", type=int, default=1)  # only for WGAN, this will be checked

    parser.add_argument("--log_dir", type=str, default="./logs")
    parser.add_argument("--ckpt_dir", type=str, default="./checkpoints")
    parser.add_argument("--data_dir", type=str, default="./data")

    args = parser.parse_args()
    config = vars(args)

    same_seeds(config['seed'])
    main(config)
