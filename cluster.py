import face_recognition
import os
import shutil
from PIL import Image, UnidentifiedImageError, ImageDraw
import random
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

# --- Configuration ---
PARENT_CELEBRITY_FOLDER = "cropped_pics"
FINAL_CLEANED_DATA_DIR = "cleaned_dataset"
NUM_SAMPLE_IMAGES_FOR_REFERENCE = 15
MIN_MATCHES_FOR_AN_ENCODING_TO_BE_CANDIDATE = 3
MAX_REFERENCE_ENCODINGS_TO_USE = 5
FACE_COMPARISON_TOLERANCE = 0.6
REFERENCE_GATHERING_TOLERANCE = 0.55
OPERATION_MODE = 'copy' # 'move' or 'copy'
FACE_DETECTION_MODEL = 'hog'
UPSAMPLE_FACE_LOCATION = 1
NUM_JITTERS_ENCODING = 1
DEBUG_SAVE_DOMINANT_FACE_IMAGE = True
MAX_WORKERS = os.cpu_count() or 4
# --- End Configuration ---

def _save_debug_dominant_face(image_path, face_locations, celebrity_name, debug_output_base_dir, suffix):
    if not DEBUG_SAVE_DOMINANT_FACE_IMAGE: return
    try:
        if not face_locations or not image_path: return
        if not os.path.exists(image_path):
            print(f"      DEBUG: Source image for debug save not found: {image_path}")
            return

        img = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        current_face_location = face_locations[0]
        (top, right, bottom, left) = current_face_location
        img_width, img_height = img.size
        if not (0 <= top < bottom <= img_height and 0 <= left < right <= img_width):
            print(f"      DEBUG: Invalid face location {current_face_location} for image {image_path}. Skipping save.")
            return

        draw.rectangle(((left, top), (right, bottom)), outline="lime", width=3)
        face_image = img.crop((left, top, right, bottom))

        debug_root = os.path.join(os.path.dirname(debug_output_base_dir), "_debug_info")
        os.makedirs(debug_root, exist_ok=True)
        debug_celebrity_dir = os.path.join(debug_root, "_dominant_faces_debug", celebrity_name)
        os.makedirs(debug_celebrity_dir, exist_ok=True)

        marked_image_filename = f"{celebrity_name}_{suffix}_marked.jpg"
        img.save(os.path.join(debug_celebrity_dir, marked_image_filename))
        cropped_face_filename = f"{celebrity_name}_{suffix}_cropped.jpg"
        face_image.save(os.path.join(debug_celebrity_dir, cropped_face_filename))
    except Exception as e:
        print(f"      DEBUG: Error saving dominant face image for {celebrity_name} from {image_path}: {e}")

def get_top_N_face_encodings(celebrity_folder_path, num_samples, min_matches_for_candidate,
                             max_references, celebrity_name, debug_output_base_dir,
                             gathering_tolerance, detection_model, upsample_loc, num_jitters_enc):
    print(f"  Attempting to find top reference faces for: {celebrity_name} (Detection: {detection_model})")
    all_image_files_in_folder = [f for f in os.listdir(celebrity_folder_path) if os.path.isfile(os.path.join(celebrity_folder_path, f))]
    image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp')
    image_files = [f for f in all_image_files_in_folder if f.lower().endswith(image_extensions)]

    if not image_files:
        print(f"    No image files found in: {celebrity_folder_path}")
        return []
    random.shuffle(image_files)
    sample_image_files = image_files[:num_samples]

    if len(sample_image_files) < min_matches_for_candidate and len(sample_image_files) > 0:
        print(f"    Warning for {celebrity_name}: Only {len(sample_image_files)} images available to sample, less than min_matches ({min_matches_for_candidate}). Proceeding.")
    elif not sample_image_files:
        print(f"    No images to sample for {celebrity_name}.")
        return []

    print(f"    Sampling {len(sample_image_files)} images for {celebrity_name}: {sample_image_files}")
    all_sample_encodings, all_sample_locations, all_sample_sources = [], [], []

    for filename in sample_image_files:
        filepath = os.path.join(celebrity_folder_path, filename)
        try:
            image_data = face_recognition.load_image_file(filepath)
            if detection_model == 'cnn':
                face_locations = face_recognition.face_locations(image_data, model=detection_model)
            else:
                face_locations = face_recognition.face_locations(image_data, number_of_times_to_upsample=upsample_loc, model=detection_model)
            encodings = face_recognition.face_encodings(image_data, face_locations, num_jitters=num_jitters_enc)
            if encodings:
                all_sample_encodings.extend(encodings)
                all_sample_locations.extend(face_locations)
                all_sample_sources.extend([filepath] * len(encodings))
                print(f"        Found {len(encodings)} face(s) in {filename}")
        except UnidentifiedImageError:
            print(f"      Warning: Skipping unidentified image file (sample): {filename}")
        except Exception as e:
            print(f"      Error processing sample image {filename}: {e}")

    if not all_sample_encodings:
        print(f"    No faces found in any sample images for {celebrity_name}.")
        return []
    if len(all_sample_encodings) < min_matches_for_candidate:
        print(f"    Found only {len(all_sample_encodings)} faces in samples for {celebrity_name} (< min_matches {min_matches_for_candidate}).")
        if DEBUG_SAVE_DOMINANT_FACE_IMAGE and all_sample_encodings:
            for i, enc in enumerate(all_sample_encodings):
                 _save_debug_dominant_face(all_sample_sources[i], [all_sample_locations[i]], celebrity_name, debug_output_base_dir, f"candidate_low_sample_{i}")
        print(f"    Using all {len(all_sample_encodings)} found faces as potential references due to low count.")
        return all_sample_encodings

    unique_reference_candidates, processed_indices = [], [False] * len(all_sample_encodings)
    for i in range(len(all_sample_encodings)):
        if processed_indices[i]: continue
        current_encoding, current_location, current_source = all_sample_encodings[i], all_sample_locations[i], all_sample_sources[i]
        similar_encodings_indices = [i]
        for j in range(i + 1, len(all_sample_encodings)):
            if not processed_indices[j]:
                if face_recognition.compare_faces([current_encoding], all_sample_encodings[j], tolerance=gathering_tolerance)[0]:
                    similar_encodings_indices.append(j)
        count = len(similar_encodings_indices)
        if count >= min_matches_for_candidate:
            unique_reference_candidates.append({"encoding": current_encoding, "count": count, "source_image": current_source, "location": current_location})
            for idx in similar_encodings_indices: processed_indices[idx] = True
        else: processed_indices[i] = True

    if not unique_reference_candidates:
        print(f"    No face group for {celebrity_name} met min_matches_for_candidate ({min_matches_for_candidate}).")
        return []
    unique_reference_candidates.sort(key=lambda x: x["count"], reverse=True)

    final_reference_encodings = []
    print(f"    Found {len(unique_reference_candidates)} unique candidate face groups for {celebrity_name}:")
    for i, candidate in enumerate(unique_reference_candidates):
        print(f"      Candidate {i+1}: Count={candidate['count']}, Source={os.path.basename(candidate['source_image'])}")
        if i < max_references:
            final_reference_encodings.append(candidate["encoding"])
            _save_debug_dominant_face(candidate["source_image"], [candidate["location"]], celebrity_name, debug_output_base_dir, f"top_ref_{i}_count_{candidate['count']}")
        else: break
    if final_reference_encodings: print(f"    Selected {len(final_reference_encodings)} top reference encodings for {celebrity_name}.")
    else: print(f"    Could not select any final reference encodings for {celebrity_name}.")
    return final_reference_encodings

def verify_and_prepare(args):
    filename, celebrity_folder_path, known_face_encodings, main_tolerance, detection_model, upsample_loc, num_jitters_enc = args
    original_filepath = os.path.join(celebrity_folder_path, filename)
    try:
        img_pil = Image.open(original_filepath); img_pil.verify(); img_pil.close()
    except (FileNotFoundError, UnidentifiedImageError, IOError):
        return (filename, False, "corrupted")
    try:
        unknown_image_data = face_recognition.load_image_file(original_filepath)
        if detection_model == 'cnn':
            unknown_face_locations = face_recognition.face_locations(unknown_image_data, model=detection_model)
        else:
            unknown_face_locations = face_recognition.face_locations(unknown_image_data, number_of_times_to_upsample=upsample_loc, model=detection_model)
        if not unknown_face_locations:
            return (filename, False, "no_face")
        unknown_face_encodings = face_recognition.face_encodings(unknown_image_data, unknown_face_locations, num_jitters=num_jitters_enc)
        if known_face_encodings:
            for unknown_encoding in unknown_face_encodings:
                matches = face_recognition.compare_faces(known_face_encodings, unknown_encoding, tolerance=main_tolerance)
                if True in matches:
                    return (filename, True, None)
        return (filename, False, "not_match")
    except Exception as e:
        return (filename, False, f"error:{e}")

def process_celebrity_folder(celebrity_folder_path, celebrity_name, final_cleaned_data_dir_base,
                             num_samples_ref, min_matches_candidate, max_refs,
                             main_tolerance, gathering_tol, operation_mode,
                             detection_model, upsample_loc, num_jitters_enc):
    print(f"\nProcessing celebrity: {celebrity_name} from folder: {celebrity_folder_path}")
    known_face_encodings = get_top_N_face_encodings(
        celebrity_folder_path, num_samples_ref, min_matches_candidate, max_refs,
        celebrity_name, final_cleaned_data_dir_base,
        gathering_tol, detection_model, upsample_loc, num_jitters_enc
    )

    if not known_face_encodings:
        print(f"  Could not determine reference faces for {celebrity_name}. Moving/Copying original folder for manual check.")
        manual_review_root = os.path.join(os.path.dirname(final_cleaned_data_dir_base), "_needs_manual_reference_check")
        os.makedirs(manual_review_root, exist_ok=True)
        dest_folder_path = os.path.join(manual_review_root, celebrity_name)
        try:
            if os.path.exists(celebrity_folder_path):
                if operation_mode == 'move':
                    if os.path.exists(dest_folder_path):
                        for item in os.listdir(celebrity_folder_path): shutil.move(os.path.join(celebrity_folder_path, item), os.path.join(dest_folder_path, item))
                        if not os.listdir(celebrity_folder_path): os.rmdir(celebrity_folder_path)
                    else: shutil.move(celebrity_folder_path, dest_folder_path)
                    print(f"  Moved folder {celebrity_name} to {dest_folder_path}.")
                else:
                    if os.path.exists(dest_folder_path): shutil.rmtree(dest_folder_path)
                    shutil.copytree(celebrity_folder_path, dest_folder_path)
                    print(f"  Copied folder {celebrity_name} to {dest_folder_path}.")
        except Exception as e: print(f"  Error moving/copying folder {celebrity_name} for manual check: {e}")
        return

    confirmed_celebrity_output_folder = os.path.join(final_cleaned_data_dir_base, celebrity_name)
    os.makedirs(confirmed_celebrity_output_folder, exist_ok=True)
    print(f"  Cleaning images for {celebrity_name}. Confirmed images to: {confirmed_celebrity_output_folder}")

    image_files_to_process = []
    if os.path.exists(celebrity_folder_path):
        image_files_to_process = [f for f in os.listdir(celebrity_folder_path) if os.path.isfile(os.path.join(celebrity_folder_path, f))]
    else:
        print(f"    Original celebrity folder not found for processing: {celebrity_folder_path}")
        return

    total_images = len(image_files_to_process)
    deleted_or_skipped_count = 0

    # Prepare arguments for parallel processing
    args_list = [
        (
            filename, celebrity_folder_path, known_face_encodings, main_tolerance,
            detection_model, upsample_loc, num_jitters_enc
        )
        for filename in image_files_to_process
    ]
    results = []
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_filename = {executor.submit(verify_and_prepare, args): args[0] for args in args_list}
        for idx, future in enumerate(as_completed(future_to_filename)):
            filename, is_match, reason = future.result()
            original_filepath = os.path.join(celebrity_folder_path, filename)
            if not os.path.exists(original_filepath): continue
            if is_match:
                base, ext = os.path.splitext(filename)
                temp_fname = filename
                counter = 1
                dest_path = os.path.join(confirmed_celebrity_output_folder, temp_fname)
                while os.path.exists(dest_path):
                    temp_fname = f"{base}_{counter}{ext}"
                    dest_path = os.path.join(confirmed_celebrity_output_folder, temp_fname)
                    counter += 1
                if operation_mode == 'move':
                    shutil.move(original_filepath, dest_path)
                elif operation_mode == 'copy':
                    shutil.copy2(original_filepath, dest_path)
                print(f"    [{idx+1}/{total_images}] {filename}: MATCH -> Saved to {os.path.basename(confirmed_celebrity_output_folder)}")
            else:
                deleted_or_skipped_count += 1
                if operation_mode == 'move' and os.path.exists(original_filepath):
                    try:
                        os.remove(original_filepath)
                        print(f"    [{idx+1}/{total_images}] {filename}: NO MATCH ({reason}) -> Deleted.")
                    except Exception as e_del:
                        print(f"    [{idx+1}/{total_images}] {filename}: Error deleting file: {e_del}")
                else:
                    print(f"    [{idx+1}/{total_images}] {filename}: NO MATCH ({reason}) -> Skipped.")

    print(f"  Finished cleaning for {celebrity_name}. {deleted_or_skipped_count} images were skipped (copy mode) or deleted (move mode).")
    if operation_mode == 'move':
        if os.path.exists(celebrity_folder_path) and not os.listdir(celebrity_folder_path):
            try:
                os.rmdir(celebrity_folder_path)
                print(f"  Removed empty original folder: {celebrity_folder_path}")
            except OSError as e:
                print(f"  Could not remove original folder {celebrity_folder_path}: {e}")
        elif os.path.exists(celebrity_folder_path):
             print(f"  Original folder {celebrity_folder_path} not removed (may still contain files or was moved for manual review).")

if __name__ == "__main__":
    if not os.path.isdir(PARENT_CELEBRITY_FOLDER):
        print(f"Error: The parent celebrity folder does not exist: {PARENT_CELEBRITY_FOLDER}")
        exit()

    os.makedirs(FINAL_CLEANED_DATA_DIR, exist_ok=True)
    if DEBUG_SAVE_DOMINANT_FACE_IMAGE:
         os.makedirs(os.path.join(os.path.dirname(FINAL_CLEANED_DATA_DIR), "_debug_info"), exist_ok=True)

    celebrity_subfolders = [d for d in os.listdir(PARENT_CELEBRITY_FOLDER) if os.path.isdir(os.path.join(PARENT_CELEBRITY_FOLDER, d))]
    if not celebrity_subfolders:
        print(f"No celebrity subfolders found in {PARENT_CELEBRITY_FOLDER}.")
        exit()

    print(f"Starting streamlined bulk cleaning for {len(celebrity_subfolders)} celebrity folders.")
    print(f"Outputting ONLY confirmed celebrity images to: {FINAL_CLEANED_DATA_DIR}")
    if OPERATION_MODE == 'move':
        print("WARNING: OPERATION_MODE is 'move'. Non-target images WILL BE DELETED from source folders.")
    print(f"Detection Model: {FACE_DETECTION_MODEL}{f', HOG Upsample: {UPSAMPLE_FACE_LOCATION}' if FACE_DETECTION_MODEL == 'hog' else ''}")
    print(f"Encoding Jitters: {NUM_JITTERS_ENCODING}")
    print(f"Sample images: {NUM_SAMPLE_IMAGES_FOR_REFERENCE}, Min matches for ref: {MIN_MATCHES_FOR_AN_ENCODING_TO_BE_CANDIDATE}, Max refs: {MAX_REFERENCE_ENCODINGS_TO_USE}")
    print(f"Main tolerance: {FACE_COMPARISON_TOLERANCE}, Ref gathering tolerance: {REFERENCE_GATHERING_TOLERANCE}")
    print("-" * 30)

    for celebrity_folder_name in celebrity_subfolders:
        if celebrity_folder_name.startswith("_") or celebrity_folder_name == os.path.basename(FINAL_CLEANED_DATA_DIR):
            print(f"Skipping special/output folder: {celebrity_folder_name}")
            continue
        cleaned_celebrity_output_path = os.path.join(FINAL_CLEANED_DATA_DIR, celebrity_folder_name)
        if os.path.isdir(cleaned_celebrity_output_path):
            print(f"Skipping '{celebrity_folder_name}': Cleaned folder already exists at '{cleaned_celebrity_output_path}'.")
            continue
        current_celebrity_folder_path = os.path.join(PARENT_CELEBRITY_FOLDER, celebrity_folder_name)
        process_celebrity_folder(
            current_celebrity_folder_path, celebrity_folder_name, FINAL_CLEANED_DATA_DIR,
            NUM_SAMPLE_IMAGES_FOR_REFERENCE, MIN_MATCHES_FOR_AN_ENCODING_TO_BE_CANDIDATE,
            MAX_REFERENCE_ENCODINGS_TO_USE, FACE_COMPARISON_TOLERANCE,
            REFERENCE_GATHERING_TOLERANCE, OPERATION_MODE, FACE_DETECTION_MODEL,
            UPSAMPLE_FACE_LOCATION, NUM_JITTERS_ENCODING
        )

    print("\nStreamlined bulk cleaning process complete!")
    print(f"Confirmed celebrity images are in subfolders within: {FINAL_CLEANED_DATA_DIR}")
    manual_review_root = os.path.join(os.path.dirname(FINAL_CLEANED_DATA_DIR), "_needs_manual_reference_check")
    if os.path.exists(manual_review_root) and any(os.scandir(manual_review_root)):
        print(f"Folders needing manual reference checks (if any) are in: {manual_review_root}")
    if DEBUG_SAVE_DOMINANT_FACE_IMAGE:
        debug_info_root = os.path.join(os.path.dirname(FINAL_CLEANED_DATA_DIR), "_debug_info", "_dominant_faces_debug")
        if os.path.exists(debug_info_root) and any(os.scandir(debug_info_root)):
            print(f"Check '{debug_info_root}' for visualized top reference faces (if any were generated).")