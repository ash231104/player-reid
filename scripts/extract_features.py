from torchvision import models, transforms
from PIL import Image
import torch
import os
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load ResNet50 and remove classification head
resnet = models.resnet50(pretrained=True)
resnet.fc = torch.nn.Identity()
resnet = resnet.to(device).eval()

# Define preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def extract_features(image_folder, output_file):
    features = []
    names = []

    for img_name in sorted(os.listdir(image_folder)):
        img_path = os.path.join(image_folder, img_name)
        img = Image.open(img_path).convert('RGB')
        input_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            embedding = resnet(input_tensor).cpu().numpy().flatten()

        features.append(embedding)
        names.append(img_name)

    os.makedirs("features", exist_ok=True)
    np.savez(output_file, names=names, embeddings=features)

if __name__ == "__main__":
    extract_features("crops/broadcast", "features/broadcast.npz")
    extract_features("crops/tacticam", "features/tacticam.npz")
