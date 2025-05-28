import cv2
import os
import sys
import numpy as np # Needed for DNN processing

# --- Configuration ---
INPUT_BASE_DIR = r"D:\MACHINE LEARNING\FaceRec\images_300" # Directory containing celebrity folders
OUTPUT_BASE_DIR = 'cropped_pics' # Directory where cropped faces will be saved

# --- DNN Face Detector Configuration ---
# Paths to the Caffe model files
PROTOTXT_PATH = 'deploy.prototxt.txt' # Path to the .prototxt file
MODEL_PATH = 'res10_300x300_ssd_iter_140000.caffemodel' # Path to the .caffemodel file
CONFIDENCE_THRESHOLD = 0.5 # Minimum probability to filter weak detections

TARGET_SIZE = (224, 224) # Desired output size (width, height) for cropped faces
PADDING_FACTOR = 0.2 # Add 20% padding around the detected face box (0 = no padding)
# MIN_FACE_SIZE is not directly used by DNN in the same way, but confidence handles small/noisy detections
SAVE_JPEG_QUALITY = 95 # Quality for saving JPEG images (0-100, higher is better)

# --- Initialization ---

# Check if DNN model files exist
if not os.path.exists(PROTOTXT_PATH) or not os.path.exists(MODEL_PATH):
    print(f"Error: DNN model files not found.")
    print(f"Ensure '{PROTOTXT_PATH}' and '{MODEL_PATH}' exist.")
    print("You can download them from OpenCV's resources.")
    sys.exit(1)

# Load the DNN face detector model
print("Loading DNN face detector model...")
net = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, MODEL_PATH)
if net.empty():
    print(f"Error: Could not load DNN model from '{PROTOTXT_PATH}' and '{MODEL_PATH}'")
    sys.exit(1)
print("DNN model loaded successfully.")

# Check if input directory exists
if not os.path.isdir(INPUT_BASE_DIR):
    print(f"Error: Input directory '{INPUT_BASE_DIR}' not found.")
    sys.exit(1)

# Create the base output directory if it doesn't exist
if not os.path.exists(OUTPUT_BASE_DIR):
    print(f"Creating output directory: '{OUTPUT_BASE_DIR}'")
    os.makedirs(OUTPUT_BASE_DIR)

print("Starting face detection and cropping process...")
print(f"Input Directory:  {os.path.abspath(INPUT_BASE_DIR)}")
print(f"Output Directory: {os.path.abspath(OUTPUT_BASE_DIR)}")
print(f"Target Size:      {TARGET_SIZE}")
print(f"Confidence Thresh:{CONFIDENCE_THRESHOLD}")
print("-" * 30)

# --- Processing ---
skipped_no_face = 0
skipped_multi_face = 0 # Will count cases where multiple faces pass confidence, and we pick the largest
processed_count = 0
error_count = 0

# Iterate through each celebrity folder in the input directory
for celebrity_name in os.listdir(INPUT_BASE_DIR):
    input_celebrity_dir = os.path.join(INPUT_BASE_DIR, celebrity_name)
    output_celebrity_dir = os.path.join(OUTPUT_BASE_DIR, celebrity_name)

    # Ensure it's actually a directory
    if not os.path.isdir(input_celebrity_dir):
        continue

    print(f"Processing folder: '{celebrity_name}'")

    # Create corresponding output celebrity folder
    if not os.path.exists(output_celebrity_dir):
        os.makedirs(output_celebrity_dir)

    # Iterate through each image file in the celebrity folder
    for filename in os.listdir(input_celebrity_dir):
        input_filepath = os.path.join(input_celebrity_dir, filename)
        output_filepath = os.path.join(output_celebrity_dir, filename)

        # Basic check for image file extensions
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            # print(f"  Skipping non-image file: {filename}")
            continue

        try:
            # Load the image in color
            image = cv2.imread(input_filepath)
            if image is None:
                print(f"  Warning: Could not read image: {input_filepath}. Skipping.")
                error_count += 1
                continue

            (orig_h, orig_w) = image.shape[:2]

            # Construct a blob from the image
            # Resize to 300x300 and apply mean subtraction (104.0, 177.0, 123.0) which are standard for this model
            blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
                                         (300, 300), (104.0, 177.0, 123.0))

            # Pass the blob through the network and obtain the detections
            net.setInput(blob)
            detections = net.forward()

            detected_faces = []
            for i in range(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]

                if confidence > CONFIDENCE_THRESHOLD: # Ensure this is '>'
                    # Extract the (x, y)-coordinates of the bounding box
                    # and compute the dimensions of the face
                    box = detections[0, 0, i, 3:7] * np.array([orig_w, orig_h, orig_w, orig_h])
                    (startX, startY, endX, endY) = box.astype("int")

                    # Ensure the bounding box is within the image bounds
                    startX = max(0, startX)
                    startY = max(0, startY)
                    endX = min(orig_w, endX)
                    endY = min(orig_h, endY)

                    w = endX - startX
                    h = endY - startY

                    # Filter out very small detections that might be false positives even with high confidence
                    if w > 10 and h > 10: # Ensure this is '>'
                        detected_faces.append((startX, startY, w, h, confidence))


            if len(detected_faces) == 0:
                # print(f"  Warning: No faces detected in {filename} with confidence > {CONFIDENCE_THRESHOLD}. Skipping.") # Ensure this is '>'
                skipped_no_face += 1
                continue

            # Handle multiple faces: choose the one with the largest area (w*h)
            # Or you could choose based on highest confidence: sorted(detected_faces, key=lambda f: f[4], reverse=True)
            if len(detected_faces) > 1: # Ensure this is '>'
                # print(f"  Info: Multiple faces ({len(detected_faces)}) detected in {filename}. Cropping the largest.")
                skipped_multi_face +=1
                detected_faces = sorted(detected_faces, key=lambda f: f[2]*f[3], reverse=True) # Sort by area w*h

            # Get coordinates of the chosen face
            (x, y, w, h, conf) = detected_faces[0]

            # Calculate padding
            pad_w = int(w * PADDING_FACTOR / 2)
            pad_h = int(h * PADDING_FACTOR / 2)

            # Calculate coordinates for cropping with padding, ensuring they stay within image bounds
            crop_x1 = max(0, x - pad_w)
            crop_y1 = max(0, y - pad_h)
            crop_x2 = min(orig_w, x + w + pad_w)
            crop_y2 = min(orig_h, y + h + pad_h)

            # Crop the original color image
            cropped_face = image[crop_y1:crop_y2, crop_x1:crop_x2]

            # Check if crop is valid
            if cropped_face.size == 0:
                print(f"  Warning: Cropped face has zero size for {filename} after padding. Skipping.")
                error_count += 1
                continue

            # --- Resize while maintaining aspect ratio and pad to TARGET_SIZE ---
            cropped_h, cropped_w = cropped_face.shape[:2]
            target_w, target_h = TARGET_SIZE

            # Calculate scaling factor to fit within target_w, target_h
            scale = min(target_w / cropped_w, target_h / cropped_h)

            # New dimensions after scaling
            new_w = int(cropped_w * scale)
            new_h = int(cropped_h * scale)

            # Determine interpolation method
            if new_w * new_h < cropped_w * cropped_h: # Shrinking
                interpolation = cv2.INTER_AREA
            else: # Enlarging or same size
                interpolation = cv2.INTER_CUBIC

            # Resize the cropped face maintaining aspect ratio
            resized_scaled_face = cv2.resize(cropped_face, (new_w, new_h), interpolation=interpolation)

            # Create a black canvas of TARGET_SIZE
            # OpenCV uses (width, height) for size, but numpy shape is (height, width, channels)
            canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)

            # Calculate top-left position to paste the resized face onto the center of the canvas
            x_offset = (target_w - new_w) // 2
            y_offset = (target_h - new_h) // 2

            # Paste the resized face onto the canvas
            canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_scaled_face
            resized_face = canvas # This is the final image to save

            # Save the resized face
            if output_filepath.lower().endswith(('.jpg', '.jpeg')):
                 cv2.imwrite(output_filepath, resized_face, [cv2.IMWRITE_JPEG_QUALITY, SAVE_JPEG_QUALITY])
            else:
                cv2.imwrite(output_filepath, resized_face)

            processed_count += 1
            if (processed_count + skipped_no_face + error_count) % 100 == 0:
                 print(f"  ... processed {processed_count + skipped_no_face + error_count} images so far ({processed_count} saved, {skipped_no_face} no-face, {error_count} errors)")


        except Exception as e:
            print(f"  Error processing file {input_filepath}: {e}")
            error_count += 1

# --- Completion Summary ---
print("-" * 30)
print("Processing Complete.")
print(f"Total images processed and saved: {processed_count}")
print(f"Images skipped (no face detected or low confidence): {skipped_no_face}")
print(f"Images where multiple faces were detected (processed largest): {skipped_multi_face}")
print(f"Errors encountered: {error_count}")
print(f"Cropped images saved in: '{os.path.abspath(OUTPUT_BASE_DIR)}'")