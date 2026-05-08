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
    
    pos_batch = get_samples_by_indices(dataset, pos_indices, device)
    mu_pos, _ = encoder(pos_batch)
    mean_pos = mu_pos.mean(dim=0)
    
    neg_batch = get_samples_by_indices(dataset, neg_indices, device)
    mu_neg, _ = encoder(neg_batch)
    mean_neg = mu_neg.mean(dim=0)
    
    return mean_pos - mean_neg