import json
import torch
import numpy as np
import os
import matplotlib.pyplot as plt

def load_latent_map(file_path='latent_manipulation_map.json'):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} not found. Run the 01_data_analysis notebook.")
    with open(file_path, 'r') as f:
        return json.load(f)

def get_samples_by_indices(dataset, indices, device='cpu'):
    samples = []
    for idx in indices:
        img, _ = dataset[idx]
        samples.append(img)
    return torch.stack(samples).to(device)

def show_image_tensor(tensor, title=None):
    image = tensor.cpu().permute(1, 2, 0).numpy()
    plt.imshow(image)
    if title:
        plt.title(title)
    plt.axis('off')
    plt.show()

@torch.no_grad()
def compute_attribute_vector(encoder, dataset, pos_indices, neg_indices, device='cpu'):
    encoder.eval()

    def collect_mu(indices):
        mus = []
        for idx in indices:
            img, cond = dataset[idx]
            img  = img.unsqueeze(0).to(device)
            cond = cond.unsqueeze(0).to(device)
            mu, _ = encoder(img, cond)
            mus.append(mu)
        return torch.cat(mus, dim=0).mean(dim=0)

    mean_pos = collect_mu(pos_indices)
    mean_neg = collect_mu(neg_indices)
    return mean_pos - mean_neg