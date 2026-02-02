import os
import cv2
import torch
import numpy as np
from dataset import OffroadDataset
from model import get_model

# Color palette for 6 classes (arbitrary but clear)
COLOR_MAP = {
    0: (0, 0, 0),        # class 0
    1: (255, 0, 0),      # class 1
    2: (0, 255, 0),      # class 2
    3: (0, 0, 255),      # class 3
    4: (255, 255, 0),    # class 4
    5: (255, 0, 255),    # class 5
}

def colorize_mask(mask):
    h, w = mask.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    for cls, color in COLOR_MAP.items():
        colored[mask == cls] = color
    return colored

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_model(num_classes=6)
    model.load_state_dict(torch.load("models/deeplabv3plus_resnet50.pth", map_location=device))
    model.to(device)
    model.eval()

    test_ds = OffroadDataset("data", split="test")

    os.makedirs("results/masks", exist_ok=True)
    os.makedirs("results/visuals", exist_ok=True)

    with torch.no_grad():
        for image, name in test_ds:
            image_tensor = image.unsqueeze(0).to(device)
            output = model(image_tensor)

            pred = torch.argmax(output, dim=1).squeeze().cpu().numpy()

            # ---- Save colored mask ----
            colored_mask = colorize_mask(pred)
            cv2.imwrite(f"results/masks/{name}", colored_mask)

            # ---- Create side-by-side visualization ----
            original = (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            original = cv2.cvtColor(original, cv2.COLOR_RGB2BGR)

            vis = np.hstack([original, colored_mask])
            cv2.imwrite(f"results/visuals/{name}", vis)

    print("Inference complete. Results saved to:")
    print(" - results/masks/")
    print(" - results/visuals/")

if __name__ == "__main__":
    main()
