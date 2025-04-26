from data_scraper2 import download_images
from pydantic import BaseModel
from config import GEMINI_API_KEY, MODEL_ID
from google import genai
from google.genai import types
import json
import os

import shutil # Import shutil for moving files
from PIL import Image, UnidentifiedImageError
import numpy as np # Import numpy
from mtcnn.mtcnn import MTCNN # Import MTCNN
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import time
import gc # Import garbage collector

# --- Configuration ---
IMAGES_FOLDER = "images"
FACE_DATASET_FOLDER = "face_dataset"
GROUP_FACE_FOLDER = "group_face"
MAX_WORKERS_FACE_PROCESSING = 6 # Adjust based on your CPU cores
PROCESSED_LOG_FILE = "processed_images.log" # <<< CHANGE 2: Log file for resumability

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
            contents=f"Generate a list of celebrity names .like this: 'celebrity : their name'. make sure the listed names are different to names in this list '{celebrity_folders} ' ",
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
                    f"{celebrity} photoshoot",
                    f"{celebrity} Solo images",
                    f"{celebrity} Face",
                    f"{celebrity} Smiling",
                ],
                save_name=f"{celebrity}",
                num_images=images_per_folder
            )
            if len(os.listdir("images")) >= celebrity_count:
                break
                    

# --- Face Processing Functions ---

def load_processed_log(log_file):
    """Loads the set of already processed image paths from the log file."""
    processed = set()
    if os.path.exists(log_file):
        try:
            with open(log_file, 'r') as f:
                for line in f:
                    processed.add(line.strip())
        except IOError as e:
            logging.error(f"Error reading processed log file {log_file}: {e}")
    logging.info(f"Loaded {len(processed)} paths from processed log file.")
    return processed

def log_processed_image(log_file, image_path):
    """Appends a successfully processed image path to the log file."""
    try:
        with open(log_file, 'a') as f:
            f.write(image_path + '\n')
    except IOError as e:
        logging.error(f"Error writing to processed log file {log_file} for path {image_path}: {e}")


def process_single_image(image_path, celebrity_name, detector, face_dest_base, group_dest_base):
    """
    Processes a single image: detects faces, crops single faces, moves group photos.
    Returns status: 'cropped', 'group', 'no_face', 'error', 'skipped', 'already_processed'.
    Also deletes the original image upon successful processing (crop, group move, no_face).
    """
    img = None
    pixels = None
    status = 'error' # Default status
    result_path = None

    try:
        # Basic check if it's a file (redundant if listdir was used, but safe)
        if not os.path.isfile(image_path):
             logging.warning(f"Skipping non-file path: {image_path}")
             return 'skipped', image_path, None

        # More robust image type check
        if not image_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
             logging.debug(f"Skipping non-image file extension: {image_path}")
             # <<< CHANGE 3: Option to delete non-image files immediately >>>
             # try:
             #     os.remove(image_path)
             #     logging.info(f"Deleted non-image file: {image_path}")
             # except OSError as e:
             #     logging.error(f"Error deleting non-image file {image_path}: {e}")
             return 'skipped', image_path, None

        # Load image
        img = Image.open(image_path)
        # Ensure it's RGB - MTCNN expects 3 channels
        if img.mode != 'RGB':
            img = img.convert('RGB')
        pixels = np.asarray(img)

        # --- Detect faces ---
        # This is the most resource-intensive part
        results = detector.detect_faces(pixels)

        base_filename = os.path.basename(image_path)
        celebrity_folder_name = celebrity_name.replace(" ", "_") # Should match folder name

        # --- Case 1: Exactly one face detected ---
        if len(results) == 1:
            x1, y1, width, height = results[0]['box']
            # Basic check for invalid box coords (negative width/height)
            if width <= 0 or height <= 0:
                 logging.warning(f"Invalid face box detected in {image_path}: w={width}, h={height}. Skipping face crop.")
                 status = 'no_face' # Treat as no face found
            else:
                x1, y1 = abs(x1), abs(y1) # Ensure start coords are non-negative
                x2, y2 = x1 + width, y1 + height

                # Add padding (optional, adjust as needed)
                padding_w = int(width * 0.15) # Slightly more padding can be good
                padding_h = int(height * 0.15)
                x1 = max(0, x1 - padding_w)
                y1 = max(0, y1 - padding_h)
                x2 = min(pixels.shape[1], x2 + padding_w) # Use numpy shape
                y2 = min(pixels.shape[0], y2 + padding_h)

                # Crop face
                face = img.crop((x1, y1, x2, y2))

                # Create destination folder
                face_dest_folder = os.path.join(face_dest_base, celebrity_folder_name)
                os.makedirs(face_dest_folder, exist_ok=True)

                # Save cropped face (use JPEG for consistency)
                face_filename = os.path.splitext(base_filename)[0] + "_face.jpg"
                face_save_path = os.path.join(face_dest_folder, face_filename)
                face.save(face_save_path, "JPEG", quality=90) # Specify quality
                status = 'cropped'
                result_path = face_save_path

        # --- Case 2: More than one face detected ---
        elif len(results) > 1:
            group_dest_folder = os.path.join(group_dest_base, celebrity_folder_name)
            os.makedirs(group_dest_folder, exist_ok=True)
            group_save_path = os.path.join(group_dest_folder, base_filename)

            # Attempt to move the file
            try:
                shutil.move(image_path, group_save_path)
                status = 'group'
                result_path = group_save_path
            except Exception as move_err:
                 logging.error(f"Error moving group file {image_path} to {group_save_path}: {move_err}")
                 status = 'error' # Keep status as error if move fails

        # --- Case 3: No faces detected ---
        else:
            status = 'no_face'
            result_path = None # No output file generated

        # <<< CHANGE 4: Delete original image if processed successfully (cropped, group, no_face) >>>
        # Only delete if status indicates success *and* it wasn't a move failure
        if status in ['cropped', 'no_face']:
            try:
                os.remove(image_path)
                # logging.debug(f"Deleted original image ({status}): {image_path}")
            except OSError as e:
                logging.error(f"Error deleting original image {image_path} after processing ({status}): {e}")
                # Don't change status back to error, log was already potentially written
        elif status == 'group':
             # logging.debug(f"Original image moved (group): {image_path} -> {result_path}")
             pass # Already moved, no need to delete

        return status, image_path, result_path

    except FileNotFoundError:
        logging.error(f"File not found during processing: {image_path}")
        return 'error', image_path, None
    except UnidentifiedImageError:
        logging.error(f"Cannot identify image file (corrupt/unsupported): {image_path}")
        # <<< CHANGE 5: Option to delete corrupt files >>>
        try:
            os.remove(image_path)
            logging.info(f"Deleted corrupt/unsupported image file: {image_path}")
        except OSError as e:
            logging.error(f"Error deleting corrupt file {image_path}: {e}")
        return 'error', image_path, None
    except MemoryError:
        logging.error(f"Memory Error processing image {image_path}. Image might be too large or system RAM exhausted.")
        # This image likely needs to be skipped or handled differently.
        # Consider moving it to an 'errors' folder instead of retrying.
        return 'error', image_path, None
    except Exception as e:
        logging.error(f"Generic error processing image {image_path}: {e}", exc_info=False) # Set exc_info=True for full traceback
        # Consider moving problematic files to an error directory
        return 'error', image_path, None
    finally:
        # <<< CHANGE 6: Explicitly delete large objects >>>
        del pixels
        if img:
            img.close() # Close the PIL image object
            del img
        # gc.collect() # Calling gc frequently might slow things down, do it periodically in the main loop instead


def process_images_for_faces():
    """Iterates through downloaded images, detects faces, organizes them, and supports resuming."""
    logging.info("--- Starting Face Processing Phase ---")
    if not os.path.exists(IMAGES_FOLDER):
        logging.error(f"Source images folder '{IMAGES_FOLDER}' not found. Cannot process faces.")
        return

    os.makedirs(FACE_DATASET_FOLDER, exist_ok=True)
    os.makedirs(GROUP_FACE_FOLDER, exist_ok=True)

    # <<< CHANGE 7: Load processed image log for resuming >>>
    processed_set = load_processed_log(PROCESSED_LOG_FILE)

    try:
        detector = MTCNN() # Initialize MTCNN detector once
        logging.info("MTCNN detector initialized.")
    except Exception as e:
        logging.error(f"Failed to initialize MTCNN detector: {e}. Face processing cannot continue.")
        return


    celebrity_folders = [d for d in os.listdir(IMAGES_FOLDER) if os.path.isdir(os.path.join(IMAGES_FOLDER, d))]
    tasks_to_submit = []

    # Collect image paths to process, skipping already processed ones
    logging.info("Scanning for images to process...")
    total_found = 0
    for celebrity_folder_name in celebrity_folders:
        celebrity_name = celebrity_folder_name.replace("_", " ") # Get original name back
        source_celebrity_folder = os.path.join(IMAGES_FOLDER, celebrity_folder_name)
        try:
            if not os.path.isdir(source_celebrity_folder): continue # Skip if not a directory

            for filename in os.listdir(source_celebrity_folder):
                image_path = os.path.join(source_celebrity_folder, filename)
                total_found += 1
                # <<< CHANGE 8: Check against processed log >>>
                if os.path.isfile(image_path) and image_path not in processed_set:
                     tasks_to_submit.append((image_path, celebrity_name))
                # Optional: Log skipped count
                # elif image_path in processed_set:
                #     logging.debug(f"Skipping already processed: {image_path}")

        except OSError as e:
            logging.error(f"Error reading directory {source_celebrity_folder}: {e}")

    total_images_to_process = len(tasks_to_submit)
    logging.info(f"Found {total_found} total files in {IMAGES_FOLDER}.")
    logging.info(f"Need to process {total_images_to_process} new images.")

    if total_images_to_process == 0:
        logging.info("No new images found to process.")
        return # Exit if nothing to do

    processed_count = 0
    cropped_count = 0
    group_count = 0
    no_face_count = 0
    error_count = 0
    skipped_count = 0 # Files skipped due to non-image format etc.

    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=MAX_WORKERS_FACE_PROCESSING) as executor:
        # Submit tasks
        future_to_task = {
            executor.submit(process_single_image, task[0], task[1], detector, FACE_DATASET_FOLDER, GROUP_FACE_FOLDER): task
            for task in tasks_to_submit
        }

        logging.info(f"Submitted {len(future_to_task)} tasks to {MAX_WORKERS_FACE_PROCESSING} workers.")

        for future in as_completed(future_to_task):
            original_path, celebrity_name = future_to_task[future]
            try:
                status, _, result_path = future.result() # Get the status and result path
                processed_count += 1

                # <<< CHANGE 9: Log successful processing for resumability >>>
                if status in ['cropped', 'group', 'no_face']:
                    log_processed_image(PROCESSED_LOG_FILE, original_path)
                    if status == 'cropped': cropped_count += 1
                    elif status == 'group': group_count += 1
                    elif status == 'no_face': no_face_count += 1
                elif status == 'error':
                    error_count += 1
                elif status == 'skipped':
                    skipped_count += 1
                    # Also log skipped files so they aren't retried endlessly
                    log_processed_image(PROCESSED_LOG_FILE, original_path)


                # Log progress periodically
                if processed_count % 100 == 0 or processed_count == total_images_to_process:
                    logging.info(f"Progress: {processed_count}/{total_images_to_process} | "
                                 f"Cropped: {cropped_count} | Group: {group_count} | "
                                 f"NoFace: {no_face_count} | Errors: {error_count} | Skipped: {skipped_count}")
                    # <<< CHANGE 10: Periodic garbage collection >>>
                    gc.collect()


            except Exception as exc:
                # Catch errors from future.result() itself (unexpected task failure)
                error_count += 1
                processed_count += 1 # Count it as processed (attempted)
                logging.error(f'Task for {original_path} generated an exception upon retrieval: {exc}')
                # Optionally log this path to the processed log as well to avoid retrying a fundamentally broken task
                # log_processed_image(PROCESSED_LOG_FILE, original_path)


    logging.info("--- Face Processing Complete ---")
    logging.info(f"Summary: Processed {processed_count}/{total_images_to_process} new images.")
    logging.info(f"  - Single Faces Cropped: {cropped_count}")
    logging.info(f"  - Group Photos Moved: {group_count}")
    logging.info(f"  - No Faces Detected (Originals Deleted): {no_face_count}")
    logging.info(f"  - Skipped Files (Non-image/etc., Logged): {skipped_count}")
    logging.info(f"  - Errors Encountered: {error_count}")
    logging.info(f"Cropped faces saved to: '{FACE_DATASET_FOLDER}'")
    logging.info(f"Group photos moved to: '{GROUP_FACE_FOLDER}'")
    logging.info(f"Successfully processed images (cropped, group, no_face, skipped) logged in: '{PROCESSED_LOG_FILE}'")
    logging.info(f"Original images folder '{IMAGES_FOLDER}' should now contain only error files or unprocessed items.")



if __name__ == "__main__":
    TARGET_CELEBRITY_COUNT = 100
    IMAGES_PER_CELEBRITY = 1300 

    try:
        # Step 1: Scrape images
        start_scraping(celebrity_count=TARGET_CELEBRITY_COUNT, images_per_folder=IMAGES_PER_CELEBRITY)
    except Exception as e :
        print("scarping stopped midway!!")

    # Step 2: Process downloaded images for faces
    try:
        process_images_for_faces()
        logging.info("--- Face Processing Phase Finished ---")
    except Exception as e:
        logging.error(f"Face processing stopped due to an error: {e}", exc_info=True)


    logging.info("--- All Tasks Finished ---")

