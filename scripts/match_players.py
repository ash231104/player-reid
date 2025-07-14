import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
import json

def match_players(broadcast_file, tacticam_file, output_file="results/matches.json"):
    data_b = np.load(broadcast_file)
    data_t = np.load(tacticam_file)

    embeddings_b = data_b["embeddings"]
    embeddings_t = data_t["embeddings"]
    names_b = data_b["names"]
    names_t = data_t["names"]

    matches = {}

    for idx_t, emb_t in enumerate(embeddings_t):
        sims = cosine_similarity([emb_t], embeddings_b)[0]
        best_idx = np.argmax(sims)
        matches[names_t[idx_t]] = names_b[best_idx]

    os.makedirs("results", exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(matches, f, indent=2)

if __name__ == "__main__":
    match_players("features/broadcast.npz", "features/tacticam.npz")
