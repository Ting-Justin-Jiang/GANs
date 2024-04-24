import os
import re

import random
import torch
import numpy as np
from torch.nn import functional as F

from pytorch_fid import fid_score
from torchvision.utils import save_image


def same_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def calculate_fid(generator, dataloader, device='cuda', num_images=2048, batch_size=64):
    generator.to(device)
    generator.eval()

    real_path = './real_images'
    fake_path = './fake_images'
    os.makedirs(real_path, exist_ok=True)
    os.makedirs(fake_path, exist_ok=True)

    # Generate and save real images
    real_count = 0
    for real_images in dataloader:
        for real_image in real_images:
            if real_count >= num_images:
                break
            real_image = (real_image + 1) / 2
            save_image(real_image, os.path.join(real_path, f'real_{real_count}.png'))
            real_count += 1
        if real_count >= num_images:
            break

    # Generate and save fake images
    fake_count = 0
    with torch.no_grad():
        while fake_count < num_images:
            z = torch.randn(batch_size, 100, device=device)
            fake_images = generator(z)
            for fake_image in fake_images:
                if fake_count >= num_images:
                    break
                fake_image = (fake_image + 1) / 2
                save_image(fake_image, os.path.join(fake_path, f'fake_{fake_count}.png'))
                fake_count += 1

    # Calculate FID score
    fid = fid_score.calculate_fid_given_paths([real_path, fake_path], batch_size, device, 2048)

    # Clean up directories
    for folder in [real_path, fake_path]:
        for filename in os.listdir(folder):
            os.remove(os.path.join(folder, filename))
        os.rmdir(folder)

    return fid


def calculate_wasserstein_distance(generator, discriminator, dataloader, device='cuda'):
    """
    Calculate the Wasserstein distance for a GAN model during inference.
    """
    generator.eval()
    discriminator.eval()

    with torch.no_grad():
        real_scores = []
        fake_scores = []
        for data in dataloader:
            real_imgs = data[0].to(device) if isinstance(data, list) else data.to(device)
            bs = real_imgs.size(0)

            # Generate fake images
            z = torch.randn(bs, generator.z_dim).to(device)
            fake_imgs = generator(z)

            # Evaluate discriminator outputs
            real_logit = discriminator(real_imgs)
            fake_logit = discriminator(fake_imgs)

            real_scores.append(real_logit.mean().item())
            fake_scores.append(fake_logit.mean().item())

        # Calculate Wasserstein distance
        wasserstein_distance = sum(real_scores) / len(real_scores) - sum(fake_scores) / len(fake_scores)

    return wasserstein_distance


def load_model_checkpoints(checkpoint_dir, load_discriminator=False):
    files = os.listdir(checkpoint_dir)
    generator_files = [f for f in files if f.startswith('G_') and f.endswith('.pth')]

    epochs = [int(re.search(r'G_(\d+).pth', f).group(1)) for f in generator_files]
    epochs.sort()

    generator_checkpoints = {epoch: os.path.join(checkpoint_dir, f'G_{epoch}.pth') for epoch in epochs}
    discriminator_checkpoints = {epoch: os.path.join(checkpoint_dir, f'D_{epoch}.pth') for epoch in epochs}

    if load_discriminator:
        return generator_checkpoints, discriminator_checkpoints
    else:
        return generator_checkpoints


def latent_interpolation(num_points, latent_dim, device='cpu'):
    """
    Generates interpolated latent vectors between two random points in the latent space.
    """
    z0 = torch.randn(1, latent_dim).to(device)
    z1 = torch.randn(1, latent_dim).to(device)

    # Generate interpolation coefficients
    alphas = torch.linspace(0, 1, steps=num_points).to(device).view(-1, 1)

    # Perform linear interpolation
    z = alphas * z0 + (1 - alphas) * z1
    z = z.view(num_points, latent_dim)
    return z


def compute_cosine_similarity(features1, features2):
    """
    Compute the cosine similarity between two sets of features.
    """
    # Normalize the feature vectors
    features1_norm = F.normalize(features1, p=2, dim=1)
    features2_norm = F.normalize(features2, p=2, dim=1)

    # Compute the cosine similarity
    similarity_matrix = torch.mm(features1_norm, features2_norm.t())
    return similarity_matrix


def flip_labels_for_generator(target_real):
    return 1 - target_real


def noisy_labels(labels, p_flip=0.05):
    """
    # Custom function for noisy labels
    :param labels:
    :param p_flip:
    :return:
    """
    # Flipping labels with probability p_flip
    flipped = labels.clone()
    select_to_flip = torch.rand(len(labels)) < p_flip
    flipped[select_to_flip] = 1 - flipped[select_to_flip]
    return flipped


def smooth_labels(labels, label_noise=0.1):
    """
    # Custom function for smooth labels
    :param labels:
    :param label_noise:
    :return:
    """
    # Adding some noise to the labels for smoothing
    noise = torch.rand(len(labels), device=labels.device) * label_noise
    smoothed = labels + noise
    smoothed = torch.clamp(smoothed, 0, 1)
    return smoothed


def extract_features(model, dataloader, device, num_samples=100):
    """
    Extracts features from the model up to num_samples images.
    """
    features = []
    with torch.no_grad():
        for i, data in enumerate(dataloader, 0):
            # Check if the dataloader returns labels, if not, just use the data
            if len(data) == 2:
                inputs, _ = data
            else:
                inputs = data[0]  # Only data, no labels

            inputs = inputs.to(device)
            output = model(inputs)
            output = output.view(output.size(0), -1)
            features.append(output)
            if len(features) * dataloader.batch_size >= num_samples:
                break
    features = torch.cat(features, 0)
    features = features[:num_samples]
    return features
