import os
import torch
from data import get_datasets, get_dataloader
from torchvision import models
import torch.nn as nn
import torch.optim as optim
import tqdm as tqdm
import numpy as np
import matplotlib.pyplot as plt

###################
# CONFIGURATION
###################

BATCH_SIZE = 128
DATA_PATH = "/Users/Monod/Downloads/temporary/data"
OUTPUT_PATH = "/Users/Monod/Downloads/temporary/models"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_CLASSES = 3

##############
# DATA LOADING
##############

val_dataset = get_datasets(path=os.path.join(DATA_PATH, "val"))
val_dataloader = get_dataloader(
    dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False
)

########
# Model
########

model = models.resnet18(pretrained=True)

# Replace final layer
model.fc = nn.Linear(model.fc.in_features, N_CLASSES)

# load the trained model weights
model.load_state_dict(torch.load(os.path.join(OUTPUT_PATH, "model_weights.pth")))

##############
# EVALUATION
##############

model.eval()

correct = 0
total = 0
missclassified_images = []
missclassified_labels = []
missclassified_preds = []

with torch.no_grad():
    for images, labels in val_dataloader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        predicted = torch.argmax(probs, dim=1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        for img, label, pred in zip(images, labels, predicted):
            if label != pred:
                missclassified_images.append(img.cpu())
                missclassified_labels.append(label.cpu())
                missclassified_preds.append(pred.cpu())

print(f"Accuracy : {100 * correct / total:.2f}%")


# Function to show images
def imshow(img, title):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = img * std + mean
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(title)
    plt.axis("off")


# show miscassified images
for i in range(len(missclassified_images)):
    imshow(
        missclassified_images[i],
        title=f"True: {missclassified_labels[i]}, Pred: {missclassified_preds[i]}",
    )
    plt.show()
