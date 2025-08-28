import torch
from PIL import Image
from torchvision import transforms
from src.model import build_model

def predict_image(img_path, model_path="potato_model.pth", device="cpu"):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    image = Image.open(img_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    model = build_model(num_classes=2)  # Defect / No Defect
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    with torch.no_grad():
        output = model(image)
        prediction = output.argmax(1).item()

    return "Defect" if prediction == 1 else "No Defect"
