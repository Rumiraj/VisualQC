import os
import random
import copy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms
from sklearn.model_selection import train_test_split

# ---------------------------------
# SETTINGS
# ---------------------------------

DATA_FOLDER = "data_cropped"
MODEL_FOLDER = "models"

BATCH_SIZE = 16
EPOCHS = 30
LEARNING_RATE = 0.0001
RANDOM_SEED = 42

os.makedirs(MODEL_FOLDER, exist_ok=True)

random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# ---------------------------------
# GPU
# ---------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device: {device}")

if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ---------------------------------
# IMAGE TRANSFORMATIONS
# ---------------------------------

# Training transformations
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),

    transforms.RandomRotation(15),

    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# Validation and testing transformations
eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# ---------------------------------
# CUSTOM DATASET CLASS
# ---------------------------------

class VisualQCDataset(datasets.ImageFolder):

    def find_classes(self, directory):

        classes = ["defective", "good"]

        class_to_idx = {
            "defective": 0,
            "good": 1
        }

        return classes, class_to_idx


# ---------------------------------
# LOAD DATASETS
# ---------------------------------

# Training dataset with augmentation
train_data = VisualQCDataset(
    root=DATA_FOLDER,
    transform=train_transform
)

# Validation and testing dataset
eval_data = VisualQCDataset(
    root=DATA_FOLDER,
    transform=eval_transform
)

# Dataset used for splitting
dataset = train_data

class_names = dataset.classes

print("Classes:", class_names)
print("Total images:", len(dataset))

# ---------------------------------
# DATASET SPLITTING
# ---------------------------------

indices = list(range(len(dataset)))
labels = [dataset.targets[i] for i in indices]

train_indices, remaining_indices = train_test_split(
    indices,
    test_size=0.30,
    stratify=labels,
    random_state=RANDOM_SEED
)

remaining_labels = [
    dataset.targets[i] for i in remaining_indices
]

val_indices, test_indices = train_test_split(
    remaining_indices,
    test_size=0.50,
    stratify=remaining_labels,
    random_state=RANDOM_SEED
)

train_dataset = Subset(
    train_data,
    train_indices
)

val_dataset = Subset(
    eval_data,
    val_indices
)

test_dataset = Subset(
    eval_data,
    test_indices
)

print(f"Training images: {len(train_dataset)}")
print(f"Validation images: {len(val_dataset)}")
print(f"Testing images: {len(test_dataset)}")

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

# ---------------------------------
# LOAD PRETRAINED RESNET18
# ---------------------------------

weights = models.ResNet18_Weights.DEFAULT

model = models.resnet18(weights=weights)

# Freeze all layers
for parameter in model.parameters():
    parameter.requires_grad = False

# Unfreeze the final convolutional block
for parameter in model.layer4.parameters():
    parameter.requires_grad = True

# Replace classification layer
model.fc = nn.Linear(
    model.fc.in_features,
    len(class_names)
)

model = model.to(device)

# ---------------------------------
# TRAINING CONFIGURATION
# ---------------------------------

# ---------------------------------
# CLASS WEIGHTS
# ---------------------------------

from collections import Counter

# Count labels in the training split only
train_labels = [
    dataset.targets[i]
    for i in train_indices
]

class_counts = Counter(train_labels)

num_classes = len(class_names)
num_training_images = len(train_labels)

class_weights = torch.tensor(
    [
        num_training_images /
        (num_classes * class_counts[i])
        for i in range(num_classes)
    ],
    dtype=torch.float32,
    device=device
)

print("Class weights:", class_weights.tolist())

# Weighted loss function
criterion = nn.CrossEntropyLoss(
    weight=class_weights
)

optimizer = torch.optim.Adam([
    {
        "params": model.layer4.parameters(),
        "lr": 0.00001
    },
    {
        "params": model.fc.parameters(),
        "lr": 0.0001
    }
])

best_accuracy = -1.0
best_model_weights = copy.deepcopy(model.state_dict())

# ---------------------------------
# TRAINING LOOP
# ---------------------------------

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item() * images.size(0)

        predictions = torch.argmax(outputs, dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    train_loss = running_loss / total
    train_accuracy = correct / total

    # ---------------------------------
    # VALIDATION
    # ---------------------------------

    model.eval()

    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            predictions = torch.argmax(outputs, dim=1)

            val_correct += (
                predictions == labels
            ).sum().item()

            val_total += labels.size(0)

    val_accuracy = val_correct / val_total

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Loss: {train_loss:.4f} | "
        f"Train Acc: {train_accuracy:.2%} | "
        f"Val Acc: {val_accuracy:.2%}"
    )

    if val_accuracy > best_accuracy:

        best_accuracy = val_accuracy

        best_model_weights = copy.deepcopy(
            model.state_dict()
        )

# ---------------------------------
# RESTORE BEST MODEL
# ---------------------------------

model.load_state_dict(best_model_weights)

# ---------------------------------
# FINAL TEST
# ---------------------------------

model.eval()

test_correct = 0
test_total = 0

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        predictions = torch.argmax(outputs, dim=1)

        test_correct += (
            predictions == labels
        ).sum().item()

        test_total += labels.size(0)

test_accuracy = test_correct / test_total

# ---------------------------------
# SAVE MODEL
# ---------------------------------

model_path = os.path.join(
    MODEL_FOLDER,
    "visualqc_resnet18_cropped.pth"
)

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "class_names": class_names,
        "test_accuracy": test_accuracy,
        "validation_accuracy": best_accuracy,
        "random_seed": RANDOM_SEED
    },
    model_path
)

print()
print("================================")
print("       TRAINING COMPLETE        ")
print("================================")

print(f"Best validation accuracy: {best_accuracy:.2%}")
print(f"Test accuracy: {test_accuracy:.2%}")
print(f"Model saved: {model_path}")

print("================================")