from data_scraper2 import download_images
from pydantic import BaseModel
from config import GEMINI_API_KEY, MODEL_ID
from google import genai
from google.genai import types
import json
import os

class Celebrity_data(BaseModel):
    Celebrities : list
    explanation : str

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
            contents=f"Generate a list of celebrities with their names.like this: 'celebrity : their name'. make sure the list names that are different to {celebrity_folders} ",
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                #response_schema=Celebrity_data,
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

def start_scraping(celebrity_count):
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
                num_images=2000
            )
                    


if __name__ == "__main__":
    start_scraping(celebrity_count=100)