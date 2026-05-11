# Latent-Space-Explorer

Project for exploring latent space manipulations using a **Conditional Variational Autoencoder (CVAE)** trained on the **CelebA** dataset.

By utilizing a conditional vector composed of **40 binary facial attributes**, the model enables targeted semantic modifications of generated faces, such as:

- adding glasses,
- changing hairstyle,
- forcing a smile,
- modifying makeup intensity,
- generating combinations of facial traits,

while preserving the core identity and structure of the generated image.

---

## Requirements

- Python 3.8+
- PyTorch + torchvision
- kagglehub, pandas, matplotlib, pillow
- tensorboard

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
- **model.py** – Contains the Conditional Variational Autoencoder (CVAE) architecture. Both the Encoder and Decoder are conditioned on the 40-dimensional attribute vector.
- **train.py** – The main training loop. Features custom VAE loss (Reconstruction + KL Divergence), KL weight annealing to prevent latent collapse, dynamic Learning Rate scheduling (ReduceLROnPlateau), and real-time TensorBoard logging.
- **requirements.txt** – List of required packages.
- **checkpoints/** *(git-ignored)* – Directory where the best model weights (*best_model.pt*) are saved during training.

- **runs/** *(git-ignored)* – Directory containing TensorBoard event logs for monitoring training metrics.

---

## How to Run

## 1. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 2. Test data pipeline

```bash
python main.py
```

> **Note:** The first run will automatically download the CelebA dataset (~1.3 GB) via `kagglehub`.

---

## 3. Train the Model

To start training the CVAE model from scratch, run:

```bash
python train.py
```

You can monitor the training progress, loss curves (Reconstruction vs KL), and real-time image generation by opening a second terminal and running:

```bash
tensorboard --logdir runs
```

Access the dashboard at:

```text
http://localhost:6006
```

---

## 4. Latent Space Manipulation (Inference)

Once the model is trained and `best_model.pt` is saved in the `checkpoints/` directory, open the Jupyter Notebook to explore the generated results and perform latent space manipulations:

```bash
jupyter notebook 01_data_analysis.ipynb
```