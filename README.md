# ⚽ Player Re-Identification – Liat.ai Assignment

## 🎯 Objective
Given two video clips (`broadcast.mp4` and `tacticam.mp4`) of the same soccer match from different angles, the goal is to identify and match players across both views by assigning consistent player IDs. This project demonstrates how visual appearance-based re-identification can be achieved using a combination of detection and feature extraction techniques.

---

## 🧠 Approach Overview

1. **Player Detection**  
   Using a fine-tuned YOLOv11 model (`best.pt`) to detect players in both videos frame by frame.

2. **Player Cropping**  
   Save detected player crops from each frame into separate folders for each video.

3. **Feature Extraction**  
   Use a pretrained ResNet50 model to extract 2048-dimensional appearance embeddings from each player crop.

4. **Player Matching**  
   Compute cosine similarity between embeddings from both views to find the best match.

5. **Visualization**  
   Display matched player images side-by-side to verify identity consistency across cameras.

---

## 🛠️ How to Run

### 1. Player Detection
Detect players in each video and save cropped images:

''' bash
python scripts/detect_players.py '''

### 2. Feature Extraction
Extract appearance features from each crop using ResNet50:

''' bash
python scripts/extract_features.py '''

### 3. Match Players Across Views
Match tacticam players to broadcast players using cosine similarity:

''' bash
python scripts/match_players.py '''
### 4. Visualize Matched Players
Save side-by-side image comparisons for visual inspection:

''' bash
python scripts/visualize_matches.py '''


## 🧪 Output
📄 results/matches.json:
A dictionary mapping each Tacticam crop to the best matching Broadcast crop.

🖼️ results/match_visuals/:
Side-by-side images showing the matched players from both views.

📓 view_matches.ipynb:
A Jupyter Notebook to preview matches inline.

## Testing 
''' bash
1. mkdir crops\input720p

2. detect_players("15sec_input_720p.mp4", "crops/input720p", model_path="best.pt", video_label="input720p")

3. from detect_players import detect_players

if __name__ == "__main__":
    detect_players("15sec_input_720p.mp4", "crops/input720p", model_path="best.pt", video_label="input720p")

4. python scripts/test_input720p.py
5. extract_features("crops/input720p", "features/input720p.npz")
6. match_players("features/broadcast.npz", "features/input720p.npz", output_file="results/matches_input720p.json")
7. visualize_matches(
    broadcast_dir="crops/broadcast",
    tacticam_dir="crops/input720p",
    matches_file="results/matches_input720p.json",
    output_dir="results/match_visuals_input720p"
)'''




🧰 Requirements
Install the required Python packages:

''' bash:
pip install ultralytics torch torchvision opencv-python scikit-learn matplotlib '''
Also ensure you have Python 3.8+ and a working virtual environment if possible.

## 📁 Project Structure
''' bash:
player-reid/
├── best.pt
├── broadcast.mp4
├── tacticam.mp4
├── 15sec_input_720p.mp4          # (Optional test clip)
├── crops/
│   ├── broadcast/                # YOLO cropped images
│   └── tacticam/
├── features/
│   ├── broadcast.npz             # ResNet50 embeddings
│   └── tacticam.npz
├── results/
│   ├── matches.json              # Matched IDs
│   └── match_visuals/           # Side-by-side comparisons
├── scripts/
│   ├── detect_players.py
│   ├── extract_features.py
│   ├── match_players.py
│   └── visualize_matches.py
├── view_matches.ipynb           # Optional Jupyter preview
└── README.md
'''
## 👤 Author 
Ashlesha Verma


## 📬 Notes
The model best.pt is a fine-tuned YOLOv11 model trained on soccer player & ball detection.

ResNet50 was used without its classification head to obtain deep feature embeddings.

Matching relies purely on visual similarity (no tracking/temporal modeling used).