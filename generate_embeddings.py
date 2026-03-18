import pyodbc
import requests
import json
import torch
import numpy as np
from PIL import Image
from io import BytesIO
import torchvision.transforms as transforms
from transformers import AutoModel
from ultralytics import YOLO

# ------------------------
# CONFIG
# ------------------------

SQL_CONNECTION = "DRIVER={ODBC Driver 17 for SQL Server};SERVER=tcp:ingweresearch.database.windows.net,1433;DATABASE=IngweResearchProject;UID=IngweResearch2025;PWD=Ingwe@!2025;Encrypt=yes;TrustServerCertificate=no;Connection Timeout=30;"
BLOB_BASE_URL = "https://ingweidimages.blob.core.windows.net/sightings/"
MODEL_TAG = "conservationxlabs/miewid-msv3"

# ------------------------
# Load AI models
# ------------------------

model = AutoModel.from_pretrained(MODEL_TAG, trust_remote_code=True)
model.eval()

detector = YOLO("yolov8n.pt")

transform = transforms.Compose([
    transforms.Resize((440, 440)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ------------------------
# Crop animal from image
# ------------------------

def crop_animal(image):
    img_array = np.array(image)
    results = detector(img_array, conf=0.3, verbose=False)

    if results and len(results[0].boxes) > 0:
        boxes = results[0].boxes
        animal_classes = [14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
        animal_mask = torch.tensor([int(c) in animal_classes for c in boxes.cls])

        if animal_mask.any():
            animal_boxes = boxes[animal_mask]
            best_idx = animal_boxes.conf.argmax()
            box = animal_boxes.xyxy[best_idx].cpu().numpy()
        else:
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

# ------------------------
# DB connection
# ------------------------

conn = pyodbc.connect(SQL_CONNECTION)
cursor = conn.cursor()

# First delete all old v2 embeddings
print("Deleting old v2 embeddings...")
cursor.execute("DELETE FROM LeopardMediaEmbedding WHERE ModelName = 'miewid-msv2'")
conn.commit()
print("Old embeddings deleted.")

# Now get all images to re-process
query = """
SELECT 
    lm.LeopardMediaId,
    lm.LeopardId,
    lm.BlobName
FROM LeopardMedia lm
LEFT JOIN LeopardMediaEmbedding e
    ON e.LeopardMediaId = lm.LeopardMediaId
WHERE e.LeopardMediaId IS NULL
"""

rows = cursor.execute(query).fetchall()
print("Images to process:", len(rows))

# ------------------------
# Process each image
# ------------------------

for i, row in enumerate(rows):
    leopardMediaId = str(row.LeopardMediaId)
    leopardId = str(row.LeopardId)
    blobName = row.BlobName

    image_url = BLOB_BASE_URL + blobName
    try:
        print(f"[{i+1}/{len(rows)}] Processing: {blobName}")

        response = requests.get(image_url, timeout=30)

        content_type = response.headers.get("Content-Type", "")
        if response.status_code != 200:
            print(f"  SKIP: status {response.status_code}")
            continue

        if not content_type.lower().startswith("image/"):
            print(f"  SKIP: content-type {content_type}")
            continue

        image = Image.open(BytesIO(response.content)).convert("RGB")
        image, cropped = crop_animal(image)

        tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            embedding = model(tensor)

        vector = embedding.numpy().flatten().tolist()
        embedding_json = json.dumps(vector)

        cursor.execute("""
        INSERT INTO LeopardMediaEmbedding
        (
            LeopardMediaId,
            LeopardId,
            ModelName,
            ModelVersion,
            EmbeddingJson,
            EmbeddingDimension,
            IsActive,
            CreatedUtc
        )
        VALUES
        (?, ?, 'miewid-msv3', '1.0', ?, ?, 1, GETUTCDATE())
        """,
        leopardMediaId,
        leopardId,
        embedding_json,
        len(vector)
        )

        conn.commit()
        print(f"  OK (cropped={cropped})")

    except Exception as e:
        print(f"  FAIL: {str(e)}")

print("Done!")