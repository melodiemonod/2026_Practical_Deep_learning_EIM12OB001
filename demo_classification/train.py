import os
import torch
from data import get_datasets, get_dataloader
from torchvision import models
import torch.nn as nn
import torch.optim as optim
import tqdm as tqdm

###################
# CONFIGURATION
###################

BATCH_SIZE = 128
EPOCHS = 3
LR = 1e-3
DATA_PATH = "/Users/Monod/Downloads/temporary/data"
OUTPUT_PATH = "/Users/Monod/Downloads/temporary/models"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_CLASSES = 3

##############
# DATA LOADING
##############

train_dataset = get_datasets(path=os.path.join(DATA_PATH, "train"))

train_dataloader = get_dataloader(
    dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True
)

########
# Model
########

model = models.resnet18(pretrained=True)

# Freeze earlu layers (transfer learning)
for param in model.parameters():
    param.requires_grad = False

# Replace final layer
model.fc = nn.Linear(model.fc.in_features, N_CLASSES)

####################
# LOSS AND OPTIMIZER
####################

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=LR)

################
# TRAINING LOOP
################

model.train()
for epoch in range(EPOCHS):
    total_loss = 0
    loop = tqdm.tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    for images, labels in loop:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    epoch_loss = total_loss / len(train_dataloader)
    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {epoch_loss:.4f}")


#############
# SAVE WEIGHTS
#############

torch.save(model.state_dict(), os.path.join(OUTPUT_PATH, "model_weights.pth"))
print("Training complete")
