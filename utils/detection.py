import torch
from torchvision import transforms
from PIL import Image

model_path = "models/latest.pt"
model = torch.jit.load(model_path, map_location="cpu")
model.eval()

transform = transforms.Compose([transforms.ToTensor()])

label_map = {
    1: "beef",
    2: "bell_pepper",
    3: "broccoli",
    4: "cabbage",
    5: "calamansi",
    6: "carrot",
    7: "chicken",
    8: "eggplant",
    9: "garlic",
    10: "ginger",
    11: "onions",
    13: "pork",
    14: "potato",
    15: "tofu",
    16: "tomato",
    17: "turmeric_powder",
}


def detect_ingredients(image):
    if image.mode != "RGB":
        image = image.convert("RGB")

    img_tensor = transform(image)
    with torch.no_grad():
        output = model([img_tensor])

    if isinstance(output, list) and len(output) > 0:
        predictions = output[0]

        if (
            isinstance(predictions, dict)
            and "labels" in predictions
            and "scores" in predictions
        ):
            labels = predictions["labels"]
            scores = predictions["scores"]
        else:
            return []
    else:
        return []

    detected = []
    for label, score in zip(labels, scores):
        if score > 0.01:
            label_name = label_map.get(label.item(), f"Class {label.item()}")
            confidence = score.item() * 100
            print(f"Detected: {label_name} with confidence {confidence:.1f}%")

            detected.append((label_name, confidence))

    return detected
