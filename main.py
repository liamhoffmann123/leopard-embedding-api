from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import torch
from PIL import Image
from io import BytesIO
import torchvision.transforms as transforms
from transformers import AutoModel

MODEL_TAG = "conservationxlabs/miewid-msv2"

app = FastAPI(title="Leopard Embedding API")

model = AutoModel.from_pretrained(MODEL_TAG, trust_remote_code=True)
model.eval()

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
        tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            output = model(tensor)

        if not isinstance(output, torch.Tensor):
            raise HTTPException(status_code=500, detail="Model output was not a tensor")

        vector = output.cpu().numpy().flatten().tolist()

        return EmbedResponse(
            modelName="miewid-msv2",
            modelVersion="1.0",
            embeddingDimension=len(vector),
            embedding=vector
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))