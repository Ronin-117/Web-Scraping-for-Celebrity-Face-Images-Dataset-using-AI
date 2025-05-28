import os
import shutil
from pathlib import Path
import numpy as np
import face_recognition
from concurrent.futures import ProcessPoolExecutor, as_completed
import hdbscan
from sklearn.metrics.pairwise import euclidean_distances

class LogColor:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    BOLD = '\033[1m'
    ENDC = '\033[0m'

SOURCE_ROOT = "cropped_pics"
DEST_ROOT = "cleaned_dataset_hdb"
IMAGE_EXTENSIONS = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp']
HDBSCAN_MIN_CLUSTER_SIZE = 3
MAX_WORKERS = os.cpu_count()
MERGE_THRESHOLD = 3  # distance between cluster centroids to merge (tune as needed)

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

def merge_clusters(embeddings, labels, threshold=MERGE_THRESHOLD):
    unique_labels = [l for l in set(labels) if l != -1]
    if len(unique_labels) <= 1:
        return [np.where(labels == unique_labels[0])[0]] if unique_labels else []

    # Compute centroids for each cluster
    centroids = []
    cluster_indices = []
    for lbl in unique_labels:
        idxs = np.where(labels == lbl)[0]
        cluster_indices.append(idxs)
        centroids.append(np.mean(embeddings[idxs], axis=0))
    centroids = np.array(centroids)

    # Compute pairwise distances between centroids
    centroid_dists = euclidean_distances(centroids)
    n = len(centroids)
    # Build groups of clusters to be merged, using union-find
    parent = list(range(n))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[py] = px

    for i in range(n):
        for j in range(i+1, n):
            if centroid_dists[i, j] < threshold:
                union(i, j)

    # Group cluster indices by merged parent
    merged_groups = {}
    for i in range(n):
        p = find(i)
        if p not in merged_groups:
            merged_groups[p] = []
        merged_groups[p].extend(cluster_indices[i])
    return list(merged_groups.values())

def clean_folder(src_folder, dest_folder):
    log(f"\n{LogColor.BOLD}Processing folder: {src_folder}{LogColor.ENDC}", LogColor.OKBLUE)
    image_paths = get_image_paths(src_folder)
    log(f"  Found {len(image_paths)} images...", LogColor.OKBLUE)

    embeddings = []
    valid_paths = []

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

    if len(embeddings) < HDBSCAN_MIN_CLUSTER_SIZE:
        log(f"  [!] Skipping {src_folder}: not enough faces found ({len(embeddings)})", LogColor.FAIL)
        return

    embeddings = np.array(embeddings)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE)
    labels = clusterer.fit_predict(embeddings)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    log(f"  HDBSCAN found {n_clusters} clusters (noise: {(labels == -1).sum()})", LogColor.OKCYAN)

    if n_clusters == 0:
        log(f"  [!] Skipping {src_folder}: no clusters found.", LogColor.FAIL)
        return

    # Merge similar clusters
    merged_clusters = merge_clusters(embeddings, labels, threshold=MERGE_THRESHOLD)
    # Find the merged group with the most images
    largest_merged = max(merged_clusters, key=len)
    log(f"  Merged clusters: {len(merged_clusters)} groups. Largest merged group size: {len(largest_merged)}", LogColor.OKGREEN)

    # Copy all images from the largest merged cluster group
    dest_folder.mkdir(parents=True, exist_ok=True)
    for idx, i in enumerate(largest_merged):
        img_path = valid_paths[i]
        dest_path = dest_folder / img_path.name
        shutil.copy2(img_path, dest_path)
        if (idx + 1) % 10 == 0 or idx == len(largest_merged) - 1:
            log(f"    Copied {idx + 1}/{len(largest_merged)} images...", LogColor.OKCYAN)
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