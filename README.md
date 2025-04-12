# Web Scraping for Images Dataset using AI

[![GitHub Repo stars](https://img.shields.io/github/stars/Ronin-117/Web-Scraping-for-Images-Dataset-using-AI?style=social)](https://github.com/Ronin-117/Web-Scraping-for-Images-Dataset-using-AI)

This project automates the process of creating image datasets of celebrities by scraping images from the web using DuckDuckGo Search. It leverages Google's Gemini AI to generate lists of celebrities, ensuring a diverse dataset and avoiding redundant scraping of already processed individuals.

**Repository:** [https://github.com/Ronin-117/Web-Scraping-for-Images-Dataset-using-AI](https://github.com/Ronin-117/Web-Scraping-for-Images-Dataset-using-AI)

## Features

*   **AI-Powered Celebrity Suggestions:** Uses Google Gemini to generate lists of celebrities to scrape, checking against already downloaded datasets to ensure uniqueness.
*   **Efficient Image Scraping:** Utilizes the `duckduckgo-search` library to find image URLs.
*   **Concurrent Downloads:** Employs multithreading (`ThreadPoolExecutor`) for faster image downloading.
*   **Duplicate Prevention:** Calculates MD5 hashes of downloaded images to avoid saving duplicates.
*   **Robust Downloading:**
    *   Uses `requests` with sessions and user-agent spoofing.
    *   Handles potential download errors and timeouts.
    *   Identifies and skips problematic domains.
    *   Determines appropriate file extensions based on URL or content type.
*   **Organized Output:** Saves images in separate folders named after each celebrity within a main `images` directory.
*   **Batch Processing & Politeness:** Downloads images in batches with pauses in between to avoid overwhelming servers (`data_scraper2.py`).
*   **Targeted Search:** Uses multiple search query variations for each celebrity to potentially gather a wider range of images.

## File Structure

```
.
├── images/                  # Default output directory for downloaded images
│   └── Celebrity_Name_1/
│   └── Celebrity_Name_2/
│       └── ...
├── main.py                # Main script to orchestrate AI name generation and scraping
├── data_scraper2.py       # Core image scraping and downloading logic (used by main.py)
├── data_scraper.py        # An alternative/earlier version of the scraper
├── config.py              # Configuration file for API keys
├── requirements.txt       # Python package dependencies
└── README.md              # This file
```

## Requirements

*   Python 3.7+
*   Google Gemini API Key

## Setup & Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Ronin-117/Web-Scraping-for-Images-Dataset-using-AI.git
    cd Web-Scraping-for-Images-Dataset-using-AI
    ```

2.  **Create and activate a virtual environment (Recommended):**
    ```bash
    # Windows
    python -m venv venv
    .\venv\Scripts\activate

    # macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure API Key:**
    *   Obtain an API key for Google Gemini from Google AI Studio.
    *   Open the `config.py` file.
    *   Replace the placeholder `"YOUR_GEMINI_API_KEY_HERE"` with your actual Gemini API key.
    *   **Important:** Add `config.py` to your `.gitignore` file to avoid accidentally committing your secret API key!
        ```
        # .gitignore
        config.py
        venv/
        __pycache__/
        *.pyc
        images/* # Optional: ignore downloaded images
        !images/.gitkeep # Optional: keep the images dir itself
        ```

## Usage

1.  **Run the main script:**
    ```bash
    python main.py
    ```

2.  **How it works:**
    *   The script checks the `images/` directory to see which celebrities have already been scraped.
    *   It asks the Google Gemini API to generate a list of new celebrity names, excluding those already present.
    *   For each new celebrity name, it calls the `download_images` function from `data_scraper2.py`.
    *   `download_images` uses various search terms (e.g., "Name", "Name actor", "Name hd") to find image URLs via DuckDuckGo.
    *   It downloads the specified number of unique images (default is 2000 per celebrity in `main.py`) concurrently.
    *   Images are saved into `images/Celebrity_Name/`.
    *   The process repeats until the target number of celebrity folders (defined by `celebrity_count` in `main.py`, default 100) is reached in the `images/` directory.

3.  **Direct Scraper Usage (Optional):**
    You can also use `data_scraper2.py` directly if you want to scrape images for a specific list of search terms without the AI component. See the example usage within the `if __name__ == "__main__":` block in `data_scraper2.py`.

## Configuration (`main.py` & `data_scraper2.py`)

*   **`main.py`:**
    *   `celebrity_count`: The total number of celebrity folders you want in the `images` directory before the script stops.
    *   `num_images`: The target number of images to download *per celebrity*.
*   **`data_scraper2.py`:**
    *   `save_folder`: The root directory where celebrity folders will be created (default: "images").
    *   `threads`: Number of concurrent download workers.
    *   `batch_size`: How many images to attempt fetching URLs for in each search cycle.
    *   `pause_time`: Seconds to wait between batches.

## Disclaimer

*   This script is intended for educational purposes and creating datasets for personal projects.
*   Please be mindful of website terms of service and copyright laws when downloading images.
*   Avoid excessively high download rates (adjust `threads` and `pause_time` if necessary) to prevent being blocked by image hosts or search engines. The current settings in `data_scraper2.py` include politeness measures (batching, pauses).

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request or open an Issue for suggestions, bug reports, or improvements.

## License

(Optional: Add a license file, e.g., MIT License, and mention it here.)
