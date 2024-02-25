from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

from PixMix import PixMix

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LR = 0.001
IMGSZ = (32,32)
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
BATCHSZ = 32
EPOCHS = 10


if __name__ == "__main__":
    model = models.resnet18(pretrained=False).to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=LR)

    train_transform = transforms.Compose([
        PixMix(),   # this is customizable
        transforms.Resize(IMGSZ),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])

    no_transform = transforms.Compose([
        transforms.Resize(IMGSZ),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])

    train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCHSZ, shuffle=True, num_workers=4)

    val_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=no_transform)
    val_loader = DataLoader(val_dataset, batch_size=BATCHSZ, shuffle=False, num_workers=4)

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for images, labels in tqdm(train_loader, desc=f"Train ({epoch+1}/{EPOCHS})"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_acc = 100 * correct / total

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Val ({epoch+1}/{EPOCHS})"):
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = 100 * correct / total

        print(f"Epoch {epoch+1}, Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.2f}%")

