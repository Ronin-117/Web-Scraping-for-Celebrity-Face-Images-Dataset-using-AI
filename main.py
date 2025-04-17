from data_scraper2 import download_images
from pydantic import BaseModel
from config import GEMINI_API_KEY, MODEL_ID
from google import genai
from google.genai import types
import json
import os

import shutil # Import shutil for moving files
from PIL import Image # Import Pillow
import numpy as np # Import numpy
from mtcnn.mtcnn import MTCNN # Import MTCNN
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

# --- Configuration ---
IMAGES_FOLDER = "images"
FACE_DATASET_FOLDER = "face_dataset"
GROUP_FACE_FOLDER = "group_face"
MAX_WORKERS_FACE_PROCESSING = 10 # Adjust based on your CPU cores

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


client = genai.Client(api_key=GEMINI_API_KEY)


def get_new_celebrity_names():
    image_folder = "images"  # Specify the folder to check
    celebrity_folders = []

    if os.path.exists(image_folder):
        for item in os.listdir(image_folder):
            item_path = os.path.join(image_folder, item)
            if os.path.isdir(item_path):
                celebrity_folders.append(item)
    else:
        print(f"Folder '{image_folder}' not found.")

    
    # Generate content using the Gemini API
    response = client.models.generate_content(
            model=MODEL_ID,
            contents=f"Generate a list of celebrity names .like this: 'celebrity : their name'. make sure the list names that are different to {celebrity_folders} ",
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
            ),
        )

    try:
        parsed_data = response.candidates[0].content.parts[0].text
        #print(parsed_data)
        data = json.loads(parsed_data)  # Parse the JSON string into a Python list of dictionaries
        celebrities = [item["celebrity"] for item in data]  # Extract the celebrity names using the correct key
        print("Celebrities:", celebrities)
    except json.JSONDecodeError as e:  # Catch JSON parsing errors specifically
        print(f"Error parsing JSON: {e}")
        celebrities = []
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        celebrities = []
    
    return celebrities

def start_scraping(celebrity_count, images_per_folder):
    while len(os.listdir("images")) < celebrity_count:

        celebrities = get_new_celebrity_names()

        for celebrity in celebrities:
            download_images(
                search_terms=[
                    f"{celebrity} images",
                    f"{celebrity} awards",
                    f"{celebrity} actor",
                    f"{celebrity} potraits",
                    f"{celebrity} hd",
                    f"{celebrity} movie scenes",
                    f"{celebrity} photoshoot",
                ],
                save_name=f"{celebrity}",
                num_images=images_per_folder
            )
            if len(os.listdir("images")) >= celebrity_count:
                break
                    

# --- Face Processing Functions ---

def process_single_image(image_path, celebrity_name, detector):
    """
    Processes a single image: detects faces, crops single faces, moves group photos.
    Returns status: 'cropped', 'group', 'no_face', 'error', 'skipped'.
    """
    try:
        # Check if file exists and is an image file (basic check)
        if not os.path.isfile(image_path) or not image_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
             logging.warning(f"Skipping non-image or non-existent file: {image_path}")
             return 'skipped', image_path, None

        img = Image.open(image_path).convert('RGB') # Load and convert to RGB
        pixels = np.asarray(img)

        # Detect faces
        results = detector.detect_faces(pixels)

        base_filename = os.path.basename(image_path)
        celebrity_folder_name = celebrity_name.replace(" ", "_")

        # --- Case 1: Exactly one face detected ---
        if len(results) == 1:
            x1, y1, width, height = results[0]['box']
            # Ensure coordinates are positive and within bounds
            x1, y1 = abs(x1), abs(y1)
            x2, y2 = x1 + width, y1 + height

            # Add some padding (optional, adjust as needed)
            padding_w = int(width * 0.1)
            padding_h = int(height * 0.1)
            x1 = max(0, x1 - padding_w)
            y1 = max(0, y1 - padding_h)
            x2 = min(pixels.shape[1], x2 + padding_w)
            y2 = min(pixels.shape[0], y2 + padding_h)

            face = img.crop((x1, y1, x2, y2))

            # Create destination folder if it doesn't exist
            face_dest_folder = os.path.join(FACE_DATASET_FOLDER, celebrity_folder_name)
            os.makedirs(face_dest_folder, exist_ok=True)

            # Save cropped face
            face_filename = os.path.splitext(base_filename)[0] + "_face.jpg" # Save as jpg
            face_save_path = os.path.join(face_dest_folder, face_filename)
            face.save(face_save_path, "JPEG")
            # logging.info(f"Cropped face saved: {face_save_path}")
            return 'cropped', image_path, face_save_path

        # --- Case 2: More than one face detected ---
        elif len(results) > 1:
            # Create destination folder if it doesn't exist
            group_dest_folder = os.path.join(GROUP_FACE_FOLDER, celebrity_folder_name)
            os.makedirs(group_dest_folder, exist_ok=True)

            # Move original image
            group_save_path = os.path.join(group_dest_folder, base_filename)
            try:
                shutil.move(image_path, group_save_path)
                # logging.info(f"Moved group photo: {group_save_path}")
                return 'group', image_path, group_save_path
            except Exception as move_err:
                 logging.error(f"Error moving file {image_path} to {group_save_path}: {move_err}")
                 return 'error', image_path, None # Indicate error during move

        # --- Case 3: No faces detected ---
        else:
            # logging.debug(f"No face detected in: {image_path}")
            return 'no_face', image_path, None

    except FileNotFoundError:
        logging.error(f"File not found during processing: {image_path}")
        return 'error', image_path, None
    except UnidentifiedImageError: # Catch Pillow error for corrupted/unsupported images
        logging.error(f"Cannot identify image file (possibly corrupt or unsupported format): {image_path}")
        # Optionally move corrupted files elsewhere
        # corrupt_folder = os.path.join("corrupt_images", celebrity_folder_name)
        # os.makedirs(corrupt_folder, exist_ok=True)
        # try:
        #     shutil.move(image_path, os.path.join(corrupt_folder, base_filename))
        # except Exception: pass # Ignore move error if it also fails
        return 'error', image_path, None
    except Exception as e:
        logging.error(f"Error processing image {image_path}: {e}")
        # Log traceback for debugging complex errors if needed
        # import traceback
        # logging.error(traceback.format_exc())
        return 'error', image_path, None


def process_images_for_faces():
    """Iterates through downloaded images, detects faces, and organizes them."""
    logging.info("--- Starting Face Processing Phase ---")
    if not os.path.exists(IMAGES_FOLDER):
        logging.error(f"Source images folder '{IMAGES_FOLDER}' not found. Cannot process faces.")
        return

    os.makedirs(FACE_DATASET_FOLDER, exist_ok=True)
    os.makedirs(GROUP_FACE_FOLDER, exist_ok=True)

    detector = MTCNN() # Initialize MTCNN detector once
    logging.info("MTCNN detector initialized.")

    celebrity_folders = [d for d in os.listdir(IMAGES_FOLDER) if os.path.isdir(os.path.join(IMAGES_FOLDER, d))]
    total_images_to_process = 0
    image_tasks = []

    # Collect all image paths first
    for celebrity_folder_name in celebrity_folders:
        celebrity_name = celebrity_folder_name.replace("_", " ") # Get original name back
        source_celebrity_folder = os.path.join(IMAGES_FOLDER, celebrity_folder_name)
        try:
            for filename in os.listdir(source_celebrity_folder):
                image_path = os.path.join(source_celebrity_folder, filename)
                if os.path.isfile(image_path):
                     image_tasks.append((image_path, celebrity_name))
                     total_images_to_process += 1
        except OSError as e:
            logging.error(f"Error reading directory {source_celebrity_folder}: {e}")

    logging.info(f"Found {total_images_to_process} images across {len(celebrity_folders)} folders to process.")

    processed_count = 0
    cropped_count = 0
    group_count = 0
    no_face_count = 0
    error_count = 0
    skipped_count = 0

    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=MAX_WORKERS_FACE_PROCESSING) as executor:
        # Submit tasks: process_single_image(image_path, celebrity_name, detector)
        future_to_task = {executor.submit(process_single_image, task[0], task[1], detector): task for task in image_tasks}

        for future in as_completed(future_to_task):
            original_path, _ = future_to_task[future] # Get original path for logging
            try:
                status, _, result_path = future.result() # Get the status and result path
                processed_count += 1

                if status == 'cropped':
                    cropped_count += 1
                elif status == 'group':
                    group_count += 1
                elif status == 'no_face':
                    no_face_count += 1
                elif status == 'error':
                    error_count += 1
                elif status == 'skipped':
                    skipped_count += 1

                if processed_count % 100 == 0: # Log progress every 100 images
                    logging.info(f"Processed: {processed_count}/{total_images_to_process} | "
                                 f"Cropped: {cropped_count} | Group: {group_count} | "
                                 f"NoFace: {no_face_count} | Errors: {error_count} | Skipped: {skipped_count}")

            except Exception as exc:
                error_count += 1
                processed_count += 1 # Count it as processed even if it failed
                logging.error(f'Task for {original_path} generated an exception: {exc}')


    logging.info("--- Face Processing Complete ---")
    logging.info(f"Summary: Total Processed: {processed_count}/{total_images_to_process}")
    logging.info(f"  - Single Faces Cropped: {cropped_count}")
    logging.info(f"  - Group Photos Moved: {group_count}")
    logging.info(f"  - No Faces Detected: {no_face_count}")
    logging.info(f"  - Skipped Files: {skipped_count}")
    logging.info(f"  - Errors Encountered: {error_count}")
    logging.info(f"Cropped faces saved to: '{FACE_DATASET_FOLDER}'")
    logging.info(f"Group photos moved to: '{GROUP_FACE_FOLDER}'")
    logging.info(f"Original images with no faces or errors remain in: '{IMAGES_FOLDER}' (unless moved due to error)")


if __name__ == "__main__":
    TARGET_CELEBRITY_COUNT = 100
    IMAGES_PER_CELEBRITY = 1300 

    try:
        # Step 1: Scrape images
        start_scraping(celebrity_count=TARGET_CELEBRITY_COUNT, images_per_folder=IMAGES_PER_CELEBRITY)
    except Exception as e :
        print("scarping stopped midway!!")

    # Step 2: Process downloaded images for faces
    process_images_for_faces()

    logging.info("--- All Tasks Finished ---")

