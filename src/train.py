import torch
import torch.optim as optim
import torch.nn as nn
from src.data_loader import get_dataloaders
from src.model import build_model

def train_model(data_dir, epochs=10, lr=0.001, batch_size=32, device="cuda"):
    train_loader, val_loader, _, classes = get_dataloaders(data_dir, batch_size)
    model = build_model(len(classes)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        running_loss, correct = 0, 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()

        acc = correct / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss:.3f}, Acc: {acc:.3f}")

    torch.save(model.state_dict(), "potato_model.pth")
    print("Model saved as potato_model.pth")

if __name__ == "__main__":
    train_model("data")
