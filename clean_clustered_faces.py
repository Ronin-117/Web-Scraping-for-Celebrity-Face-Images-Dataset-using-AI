import os
import shutil
from pathlib import Path
import numpy as np
from sklearn.cluster import DBSCAN
import face_recognition
from concurrent.futures import ProcessPoolExecutor, as_completed

# Terminal color codes for pretty logging
class LogColor:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    BOLD = '\033[1m'
    ENDC = '\033[0m'

# SETTINGS
SOURCE_ROOT = "cropped_pics"      # Root directory containing celebrity folders
DEST_ROOT = "cleaned_dataset"      # Where to save cleaned folders
IMAGE_EXTENSIONS = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp']
DBSCAN_EPS = 0.6                   # Clustering threshold - adjust as needed
DBSCAN_MIN_SAMPLES = 3
MAX_WORKERS = os.cpu_count()       # Number of processes for parallelism

def log(msg, color=LogColor.ENDC):
    print(f"{color}{msg}{LogColor.ENDC}")

def get_image_paths(folder):
    return [p for ext in IMAGE_EXTENSIONS for p in Path(folder).glob(ext)]

def get_embedding(image_path):
    try:
        img = face_recognition.load_image_file(str(image_path))
        encodings = face_recognition.face_encodings(img)
        return (str(image_path), encodings[0]) if encodings else (str(image_path), None)
    except Exception as e:
        return (str(image_path), None)

def clean_folder(src_folder, dest_folder):
    log(f"\n{LogColor.BOLD}Processing folder: {src_folder}{LogColor.ENDC}", LogColor.OKBLUE)
    image_paths = get_image_paths(src_folder)
    log(f"  Found {len(image_paths)} images...", LogColor.OKBLUE)

    embeddings = []
    valid_paths = []

    # Use multiprocessing for embedding extraction
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_path = {executor.submit(get_embedding, img_path): img_path for img_path in image_paths}
        for idx, future in enumerate(as_completed(future_to_path)):
            img_path, emb = future.result()
            if emb is not None:
                embeddings.append(emb)
                valid_paths.append(Path(img_path))
            else:
                log(f"    Skipping (no face): {Path(img_path).name}", LogColor.WARNING)
            if (idx + 1) % 10 == 0 or idx == len(image_paths) - 1:
                log(f"    Processed {idx + 1}/{len(image_paths)} images", LogColor.OKCYAN)

    if len(embeddings) < DBSCAN_MIN_SAMPLES:
        log(f"  [!] Skipping {src_folder}: not enough faces found ({len(embeddings)})", LogColor.FAIL)
        return

    embeddings = np.array(embeddings)
    clustering = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES).fit(embeddings)
    labels = clustering.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    log(f"  DBSCAN found {n_clusters} clusters (noise: {(labels == -1).sum()})", LogColor.OKCYAN)

    if n_clusters == 0:
        log(f"  [!] Skipping {src_folder}: no clusters found.", LogColor.FAIL)
        return

    # Find largest cluster
    label_counts = np.bincount(labels[labels != -1])
    largest_cluster = np.argmax(label_counts)
    cluster_indices = np.where(labels == largest_cluster)[0]
    cluster_paths = [valid_paths[i] for i in cluster_indices]
    log(f"  Largest cluster size: {len(cluster_paths)}", LogColor.OKGREEN)

    # Mirror the folder structure in destination
    dest_folder.mkdir(parents=True, exist_ok=True)
    for idx, img_path in enumerate(cluster_paths):
        dest_path = dest_folder / img_path.name
        shutil.copy2(img_path, dest_path)
        if (idx + 1) % 10 == 0 or idx == len(cluster_paths) - 1:
            log(f"    Copied {idx + 1}/{len(cluster_paths)} images...", LogColor.OKCYAN)
    log(f"  {LogColor.BOLD}[DONE]{LogColor.ENDC} Processed {src_folder}, cleaned images saved to {dest_folder}", LogColor.OKGREEN)

def main():
    src_root = Path(SOURCE_ROOT)
    dest_root = Path(DEST_ROOT)
    for subfolder in src_root.iterdir():
        if subfolder.is_dir():
            rel_path = subfolder.relative_to(src_root)
            new_dest_folder = dest_root / rel_path
            clean_folder(subfolder, new_dest_folder)
    log("\nAll folders processed.", LogColor.HEADER)

if __name__ == "__main__":
    main()