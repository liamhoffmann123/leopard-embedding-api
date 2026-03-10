import pyodbc
import requests
import json
import torch
from PIL import Image
from io import BytesIO
import torchvision.transforms as transforms
from transformers import AutoModel

SQL_CONNECTION = "DRIVER={ODBC Driver 17 for SQL Server};SERVER=tcp:ingweresearch.database.windows.net,1433;DATABASE=IngweResearchProject;UID=IngweResearch2025;PWD=Ingwe@!2025;Encrypt=yes;TrustServerCertificate=no;Connection Timeout=30;"
MODEL_TAG = "conservationxlabs/miewid-msv2"

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

conn = pyodbc.connect(SQL_CONNECTION)
cursor = conn.cursor()

query = """
SELECT
    sm.SightingMediaId,
    sm.SightingId,
    sm.StorageUrl
FROM dbo.SightingMedia sm
LEFT JOIN dbo.SightingMediaEmbedding e
    ON e.SightingMediaId = sm.SightingMediaId
WHERE e.SightingMediaId IS NULL
  AND sm.StorageUrl IS NOT NULL
  AND sm.Status = 'Active'
"""

rows = cursor.execute(query).fetchall()

print("Sighting images to process:", len(rows))

for row in rows:
    sightingMediaId = str(row.SightingMediaId)
    sightingId = str(row.SightingId)
    image_url = row.StorageUrl

    try:
        print("Trying:", image_url)

        if image_url.lower().endswith(".mp4"):
            print("Skipping video:", sightingMediaId)
            continue

        response = requests.get(image_url, timeout=30)

        if response.status_code != 200:
            print("Failed:", sightingMediaId, "status", response.status_code)
            continue

        image = Image.open(BytesIO(response.content)).convert("RGB")
        tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            embedding = model(tensor)

        vector = embedding.numpy().flatten().tolist()
        embedding_json = json.dumps(vector)

        cursor.execute("""
        INSERT INTO dbo.SightingMediaEmbedding
        (
            SightingMediaId,
            SightingId,
            ModelName,
            ModelVersion,
            EmbeddingJson,
            EmbeddingDimension,
            CreatedUtc
        )
        VALUES
        (?, ?, 'miewid-msv2', '1.0', ?, ?, GETUTCDATE())
        """,
        sightingMediaId,
        sightingId,
        embedding_json,
        len(vector)
        )

        conn.commit()
        print("Processed:", sightingMediaId)

    except Exception as e:
        print("Failed:", sightingMediaId, str(e))

cursor.close()
conn.close()
print("Done")