import os
import torch
import pandas as pd
import kagglehub
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class CelebAKaggleDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row['image_id']
        
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        
        labels = torch.tensor(row.drop(['image_id', 'partition']).values.astype(float)).float()
        labels = torch.where(labels < 0, 0, labels) 

        if self.transform:
            image = self.transform(image)

        return image, labels

def get_transforms(img_size=64):
    return transforms.Compose([
        transforms.CenterCrop(178),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

def get_celeba_dataloaders(batch_size=64, img_size=64):
    path = kagglehub.dataset_download("jessicali9530/celeba-dataset")
    
    img_dir = os.path.join(path, "img_align_celeba", "img_align_celeba")
    attr_path = os.path.join(path, "list_attr_celeba.csv")
    partition_path = os.path.join(path, "list_eval_partition.csv")

    attr_df = pd.read_csv(attr_path)
    partition_df = pd.read_csv(partition_path)
    
    full_df = pd.merge(attr_df, partition_df, on="image_id")

    transform = get_transforms(img_size)

    train_df = full_df[full_df['partition'] == 0]
    valid_df = full_df[full_df['partition'] == 1]
    test_df  = full_df[full_df['partition'] == 2]

    train_ds = CelebAKaggleDataset(train_df, img_dir, transform)
    valid_ds = CelebAKaggleDataset(valid_df, img_dir, transform)
    test_ds  = CelebAKaggleDataset(test_df,  img_dir, transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader, test_loader