import yaml
import torch
import random
import numpy as np
from torch.utils.data import DataLoader
from torch import nn
from tqdm import tqdm

from dataset import OffroadDataset
from model import get_model
from metrics import mean_iou

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main():
    with open("configs/baseline.yaml") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = OffroadDataset("data", "train", cfg["image_size"])
    val_ds = OffroadDataset("data", "val", cfg["image_size"])

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False)

    model = get_model(cfg["num_classes"]).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"])

    best_iou = 0.0

    for epoch in range(cfg["epochs"]):
        model.train()
        train_loss = 0.0

        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        model.eval()
        val_iou = 0.0

        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                val_iou += mean_iou(outputs, masks, cfg["num_classes"])

        val_iou /= len(val_loader)

        print(f"Epoch {epoch+1} | Loss: {train_loss:.4f} | Val mIoU: {val_iou:.4f}")

        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), "models/deeplabv3plus_resnet50.pth")

if __name__ == "__main__":
    main()
