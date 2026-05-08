# Latent-Space-Explorer

Project for exploring latent space manipulations using a Variational Autoencoder (VAE) on the CelebA dataset.

## Requirements

- Python 3.8+
- PyTorch + torchvision
- kagglehub, pandas, matplotlib, pillow

Install dependencies:
```bash
pip install -r requirements.txt
````

## Project Structure

- **preprocessing.py** – CelebA dataset class and DataLoader creation (train/val/test). Images are center-cropped and resized to 64x64.
- **main.py** – Simple script to test data loading.
- **01_data_analysis.ipynb** – Exploratory Data Analysis notebook (attribute distributions, correlations, latent manipulation map generation).
- **utils.py** – Utility functions for loading the manipulation map, retrieving images by index, and computing attribute vectors in latent space.
- **latent_manipulation_map.json** – Precomputed positive/negative sample indices for 10 facial attributes (Smiling, Male, Eyeglasses, etc.).
- **requirements.txt** – List of required packages.

## How to Run

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
    ````
2. **Test data pipeline**
   ```bash
   python main.py
   ````
   The first run will automatically download the CelebA dataset (~1.3 GB) via kagglehub.
   
