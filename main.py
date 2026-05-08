from preprocessing import get_celeba_dataloaders

if __name__ == "__main__":
    train_l, valid_l, test_l = get_celeba_dataloaders(batch_size=32)
    print("dataset downloaded successfully")
    
    img, label = next(iter(train_l))
    print(f"Batch shape: {img.shape}")