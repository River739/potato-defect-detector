import torch
from sklearn.metrics import classification_report, confusion_matrix
from src.data_loader import get_dataloaders
from src.model import build_model

def evaluate_model(data_dir, model_path="potato_model.pth", device="cuda"):
    _, _, test_loader, classes = get_dataloaders(data_dir)
    model = build_model(len(classes))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            outputs = model(imgs)
            preds = outputs.argmax(1)
            y_true.extend(labels.numpy())
            y_pred.extend(preds.numpy())

    print(classification_report(y_true, y_pred, target_names=classes))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
