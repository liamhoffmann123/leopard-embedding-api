from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import torch
import numpy as np
from PIL import Image
from io import BytesIO
import torchvision.transforms as transforms
from transformers import AutoModel
from ultralytics import YOLO

MODEL_TAG = "conservationxlabs/miewid-msv3"

app = FastAPI(title="Leopard Embedding API")

model = AutoModel.from_pretrained(MODEL_TAG, trust_remote_code=True)
model.eval()

# MegaDetector v5 - downloads automatically
detector = YOLO("yolov8n.pt")
transform = transforms.Compose([
    transforms.Resize((440, 440)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

class EmbedRequest(BaseModel):
    imageUrl: str

class EmbedResponse(BaseModel):
    modelName: str
    modelVersion: str
    embeddingDimension: int
    embedding: list[float]
    cropped: bool

def crop_animal(image: Image.Image) -> tuple[Image.Image, bool]:
    img_array = np.array(image)
    results = detector(img_array, conf=0.3, verbose=False)

    if results and len(results[0].boxes) > 0:
        boxes = results[0].boxes
        # COCO animal class IDs: cat=15, dog=16, horse=17, sheep=18, cow=19,
        # elephant=20, bear=21, zebra=22, giraffe=23, bird=14
        animal_classes = [14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
        
        # Filter for animals only
        animal_mask = torch.tensor([int(c) in animal_classes for c in boxes.cls])
        
        if animal_mask.any():
            animal_boxes = boxes[animal_mask]
            best_idx = animal_boxes.conf.argmax()
            box = animal_boxes.xyxy[best_idx].cpu().numpy()
        else:
            # No animal found, use highest confidence detection anyway
            best_idx = boxes.conf.argmax()
            box = boxes.xyxy[best_idx].cpu().numpy()

        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])

        pad = 10
        w, h = image.size
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad)
        y2 = min(h, y2 + pad)

        return image.crop((x1, y1, x2, y2)), True

    return image, False

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/embed", response_model=EmbedResponse)
def embed(req: EmbedRequest):
    image_url = req.imageUrl.strip()

    if image_url.lower().endswith(".mp4"):
        raise HTTPException(status_code=400, detail="Videos are not supported")

    try:
        response = requests.get(image_url, timeout=30)
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail=f"Could not fetch image. Status {response.status_code}")

        image = Image.open(BytesIO(response.content)).convert("RGB")
        image, was_cropped = crop_animal(image)
        tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            output = model(tensor)

        if not isinstance(output, torch.Tensor):
            raise HTTPException(status_code=500, detail="Model output was not a tensor")

        vector = output.cpu().numpy().flatten().tolist()

        return EmbedResponse(
            modelName="miewid-msv3",
            modelVersion="1.0",
            embeddingDimension=len(vector),
            embedding=vector,
            cropped=was_cropped
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))