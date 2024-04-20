import random
import torch
import numpy as np
from torch.nn import functional as F


def same_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def latent_interpolation(num_points, latent_dim, device='cpu'):
    """
    Generates interpolated latent vectors between two random points in the latent space.
    """
    z0 = torch.randn(1, latent_dim).to(device)
    z1 = torch.randn(1, latent_dim).to(device)
    alphas = torch.linspace(0, 1, steps=num_points).to(device).view(-1, 1)

    # Linear interpolation
    z = alphas * z0 + (1 - alphas) * z1  # Broadcast alphas across the latent dimension

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
