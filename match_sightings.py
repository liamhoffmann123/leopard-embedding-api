import pyodbc
import json
import numpy as np
from collections import defaultdict

SQL_CONNECTION = "DRIVER={ODBC Driver 17 for SQL Server};SERVER=tcp:ingweresearch.database.windows.net,1433;DATABASE=IngweResearchProject;UID=IngweResearch2025;PWD=Ingwe@!2025;Encrypt=yes;TrustServerCertificate=no;Connection Timeout=30;"
TOP_K = 5

def cosine_similarity(a, b):
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm == 0 or b_norm == 0:
        return 0.0
    return float(np.dot(a, b) / (a_norm * b_norm))

conn = pyodbc.connect(SQL_CONNECTION)
cursor = conn.cursor()

sighting_rows = cursor.execute("""
SELECT
    sme.SightingMediaEmbeddingId,
    sme.SightingMediaId,
    sme.SightingId,
    sme.EmbeddingJson
FROM dbo.SightingMediaEmbedding sme
WHERE NOT EXISTS
(
    SELECT 1
    FROM dbo.SightingMatchCandidate smc
    WHERE smc.SightingId = sme.SightingId
)
""").fetchall()

leopard_rows = cursor.execute("""
SELECT
    lme.LeopardMediaEmbeddingId,
    lme.LeopardMediaId,
    lme.LeopardId,
    lme.EmbeddingJson
FROM dbo.LeopardMediaEmbedding lme
WHERE lme.IsActive = 1
""").fetchall()

print("Sightings to match:", len(sighting_rows))
print("Leopard reference embeddings:", len(leopard_rows))

leopard_vectors = []
for row in leopard_rows:
    try:
        vector = np.array(json.loads(row.EmbeddingJson), dtype=np.float32)
        leopard_vectors.append({
            "LeopardMediaId": str(row.LeopardMediaId),
            "LeopardId": str(row.LeopardId),
            "Vector": vector
        })
    except Exception:
        pass

for row in sighting_rows:
    sighting_media_embedding_id = str(row.SightingMediaEmbeddingId)
    sighting_media_id = str(row.SightingMediaId)
    sighting_id = str(row.SightingId)

    try:
        sighting_vector = np.array(json.loads(row.EmbeddingJson), dtype=np.float32)

        leopard_scores = defaultdict(list)

        for leopard in leopard_vectors:
            score = cosine_similarity(sighting_vector, leopard["Vector"])
            leopard_scores[leopard["LeopardId"]].append({
                "Score": score,
                "LeopardMediaId": leopard["LeopardMediaId"]
            })

        ranked = []
        for leopard_id, matches in leopard_scores.items():
            best_match = max(matches, key=lambda x: x["Score"])
            ranked.append({
                "LeopardId": leopard_id,
                "Score": best_match["Score"],
                "LeopardMediaId": best_match["LeopardMediaId"]
            })

        ranked.sort(key=lambda x: x["Score"], reverse=True)
        top_matches = ranked[:TOP_K]

        existing = cursor.execute("""
        SELECT COUNT(*)
        FROM dbo.SightingMatchCandidate
        WHERE SightingId = ?
        """, sighting_id).fetchone()[0]

        if existing > 0:
            print("Skipping already matched sighting:", sighting_id)
            continue

        rank = 1
        for match in top_matches:
            cursor.execute("""
            INSERT INTO dbo.SightingMatchCandidate
            (
                SightingId,
                LeopardId,
                LeopardMediaId,
                Score,
                Rank,
                ModelName,
                ModelVersion,
                CreatedUtc
            )
            VALUES
            (?, ?, ?, ?, ?, 'miewid-msv2', '1.0', GETUTCDATE())
            """,
            sighting_id,
            match["LeopardId"],
            match["LeopardMediaId"],
            round(match["Score"], 6),
            rank
            )
            rank += 1

        conn.commit()
        print("Matched sighting:", sighting_id)

    except Exception as e:
        print("Failed sighting:", sighting_id, str(e))

cursor.close()
conn.close()
print("Done")