from data_scarper import download_images
from pydantic import BaseModel
from config import GEMINI_API_KEY, MODEL_ID
from google import genai
from google.genai import types
import json

class Celebrity_data(BaseModel):
    Celebrities : list

client = genai.Client(api_key=GEMINI_API_KEY)

response = client.models.generate_content(
        model=MODEL_ID,
        contents=f"Generate a list of celebrities with their names.",
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            #response_schema=Celebrity_data.model_json_schema(),
        ),
    )

try:
    parsed_data = response.candidates[0].content.parts[0].text
    print(parsed_data)  # Print the raw JSON for inspection
    data = json.loads(parsed_data)
    celebrities = data["names"]  # Access the "names" key
    print("Celebrities:", celebrities)
except json.JSONDecodeError as e:  # Catch JSON parsing errors specifically
    print(f"Error parsing JSON: {e}")
    celebrities = []
except Exception as e:
    print(f"An unexpected error occurred: {e}")
    celebrities = []
#download_images("Elon Musk", num_images=10, save_folder="images", max_workers=8)