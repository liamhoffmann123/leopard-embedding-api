import pyodbc
import requests
import json
import torch
import numpy as np
from PIL import Image
from io import BytesIO
import torchvision.transforms as transforms
from transformers import AutoModel

# ------------------------
# CONFIG
# ------------------------

SQL_CONNECTION = "DRIVER={ODBC Driver 17 for SQL Server};SERVER=tcp:ingweresearch.database.windows.net,1433;DATABASE=IngweResearchProject;UID=IngweResearch2025;PWD=Ingwe@!2025;Encrypt=yes;TrustServerCertificate=no;Connection Timeout=30;"
BLOB_BASE_URL = "https://ingweidimages.blob.core.windows.net/sightings/"
MODEL_TAG = "conservationxlabs/miewid-msv2"

# ------------------------
# Load AI model
# ------------------------

model = AutoModel.from_pretrained(MODEL_TAG, trust_remote_code=True)
model.eval()

transform = transforms.Compose([
    transforms.Resize((440, 440)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

# ------------------------
# DB connection
# ------------------------

conn = pyodbc.connect(SQL_CONNECTION)
cursor = conn.cursor()

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

for row in rows:
    leopardMediaId = str(row.LeopardMediaId)
    leopardId = str(row.LeopardId)
    blobName = row.BlobName

    image_url = BLOB_BASE_URL + blobName
    try:
        print("Trying:", image_url)

        response = requests.get(image_url, timeout=30)

        content_type = response.headers.get("Content-Type", "")
        status_code = response.status_code

        if status_code != 200:
            print("Failed:", blobName, "status", status_code)
            continue

        if not content_type.lower().startswith("image/"):
            try:
                preview = response.text[:200].replace("\n", " ")
            except:
                preview = "non-text response"
            print("Failed:", blobName, "content-type", content_type, "preview", preview)
            continue

        image = Image.open(BytesIO(response.content)).convert("RGB")

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
            CreatedUtc
        )
        VALUES
        (?, ?, 'miewid-msv2', '1.0', ?, ?, GETUTCDATE())
        """,
        leopardMediaId,
        leopardId,
        embedding_json,
        len(vector)
        )

        conn.commit()
        print("Processed:", blobName)

    except Exception as e:
        print("Failed:", blobName, str(e))