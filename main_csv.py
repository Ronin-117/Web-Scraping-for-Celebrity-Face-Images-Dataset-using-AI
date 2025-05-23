from data_scraper2 import download_images
import os
import pandas as pd
import logging
import time

# --- Configuration ---
IMAGES_FOLDER = "images"

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')



def get_celebrity_names_from_csv(csv_filepath: str, column_name: str = "full_name") -> list:

    try:
        df = pd.read_csv(csv_filepath)
        if column_name not in df.columns:
            logging.error(f"Column '{column_name}' not found in CSV file: {csv_filepath}")
            return []
        # Drop NA values and convert to string to ensure all names are processable
        celebrities = df[column_name].dropna().astype(str).tolist()
        logging.info(f"Successfully read {len(celebrities)} celebrity names from '{csv_filepath}'.")
        return celebrities
    except FileNotFoundError:
        logging.error(f"CSV file not found: {csv_filepath}")
        celebrities = []
    except Exception as e:
        logging.error(f"Error reading CSV file {csv_filepath}: {e}")
        celebrities = []
    
    return celebrities

def start_scraping(csv_filepath: str, images_per_folder: int):
    

    celebrities = get_celebrity_names_from_csv(csv_filepath)

    if not celebrities:
        logging.warning("No celebrity names found in the CSV or CSV could not be read. Exiting scraping.")
        return

    logging.info(f"--- Starting image scraping for {len(celebrities)} celebrities listed in '{csv_filepath}' ---")
    os.makedirs(IMAGES_FOLDER, exist_ok=True)     

    for i, celebrity_name in enumerate(celebrities):
        logging.info(f"Processing celebrity {i+1}/{len(celebrities)}: {celebrity_name}")

        celebrity_folder_name = celebrity_name.replace(" ", "_")
        celebrity_folder_path = os.path.join(IMAGES_FOLDER, celebrity_folder_name)

        if os.path.exists(celebrity_folder_path):
            logging.info(f"Folder for '{celebrity_name}' already exists at '{celebrity_folder_path}'. Skipping download.")
            continue

        download_images(
            search_terms=[
                f" {celebrity_name} images",
                f" {celebrity_name} awards",
                f" {celebrity_name} actor",
                f" {celebrity_name} portraits", # Corrected typo
                f" {celebrity_name} hd",
                f" {celebrity_name} photoshoot",
                f" {celebrity_name} Solo images",
                f" {celebrity_name} Face",
                f" {celebrity_name} Smiling",
            ],
            save_name=celebrity_name, # Use the name directly
            num_images=images_per_folder
        )       

    logging.info(f"--- Finished scraping images for all celebrities in '{csv_filepath}' ---")
    return True

if __name__ == "__main__":

    # Define the path to your CSV file and the number of images per celebrity
    CELEBRITY_CSV_FILE = "celebs_batch_A_1.csv"  # Ensure this file exists with a 'full_name' column
    IMAGES_PER_CELEBRITY = 1000  # Adjust as needed, original was 1300
    
    while True:
        try:
            logging.info("--- Starting Image Scraping Process ---")
            if start_scraping(csv_filepath=CELEBRITY_CSV_FILE, images_per_folder=IMAGES_PER_CELEBRITY):
                break
            logging.info("--- Image Scraping Process Finished ---")
        except Exception as e :
            logging.error(f"Image scraping stopped midway due to an unhandled error: {e}", exc_info=True)

    logging.info("--- All Tasks Finished ---")

