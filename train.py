import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

from config import NUM_CLASSES, FEATURES_PATH, LABELS_PATH, MODEL_WEIGHTS_PATH
from model import LipRead3D


class LipReadingDataset(Dataset):
    def __init__(self, data_path, labels_path):
        self.X = np.load(data_path)
        self.y = np.load(labels_path)

    def __len__(self): return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.long)


def train_pipeline(epochs=30, batch_size=16, learning_rate=0.001):
    print("\n🚀 --- Training Pixel-Aware 3D Model ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = LipReadingDataset(FEATURES_PATH, LABELS_PATH)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = LipRead3D(num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3)

    best_acc = 0.0
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / train_size

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = correct / total if total > 0 else 0.0
        scheduler.step(val_acc)
        print(
            f"Epoch [{epoch+1:02d}/{epochs}] | Train Loss: {epoch_loss:.4f} | Val Accuracy: {val_acc*100:.2f}%")

        if val_acc >= best_acc and val_acc > 0:
            best_acc = val_acc
            torch.save(model.state_dict(), MODEL_WEIGHTS_PATH)

    print(f"\n✅ Training Complete! Best Accuracy: {best_acc*100:.2f}%")


if __name__ == "__main__":
    train_pipeline(epochs=25)
