import os
import json
import cv2
import numpy as np

def visualize_matches(broadcast_dir, tacticam_dir, matches_file, output_dir="results/match_visuals"):
    os.makedirs(output_dir, exist_ok=True)

    with open(matches_file, "r") as f:
        matches = json.load(f)

    for tacticam_name, broadcast_name in matches.items():
        tacticam_path = os.path.join(tacticam_dir, tacticam_name)
        broadcast_path = os.path.join(broadcast_dir, broadcast_name)

        if not os.path.exists(tacticam_path) or not os.path.exists(broadcast_path):
            continue

        img_tac = cv2.imread(tacticam_path)
        img_brd = cv2.imread(broadcast_path)

        # Resize to same height
        h = min(img_tac.shape[0], img_brd.shape[0])
        img_tac = cv2.resize(img_tac, (int(img_tac.shape[1] * h / img_tac.shape[0]), h))
        img_brd = cv2.resize(img_brd, (int(img_brd.shape[1] * h / img_brd.shape[0]), h))

        concat = np.concatenate((img_tac, img_brd), axis=1)
        save_path = os.path.join(output_dir, f"{tacticam_name.replace('.jpg', '')}_match.jpg")
        cv2.imwrite(save_path, concat)

if __name__ == "__main__":
    visualize_matches(
        broadcast_dir="crops/broadcast",
        tacticam_dir="crops/tacticam",
        matches_file="results/matches.json"
    )
