import os
import requests
from duckduckgo_search import DDGS
from urllib.parse import urlparse
import mimetypes
import hashlib
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

def get_extension_from_url_or_content(url, content):
    # Try to guess extension from URL
    path = urlparse(url).path
    ext = os.path.splitext(path)[1]
    if ext and len(ext) <= 5:
        return ext.lstrip(".")

    # Try to guess extension from content type using HEAD request first
    try:
        head_response = requests.head(url, timeout=5)
        content_type_header = head_response.headers.get('content-type', '').split(";")[0]
        mime_ext = mimetypes.guess_extension(content_type_header)
        if mime_ext:
            return mime_ext.lstrip(".")
    except requests.exceptions.RequestException:
        # Ignore errors during HEAD request, proceed to content check if available
        pass

    # If content is available, try to guess from magic bytes (more robust but needs libraries like 'python-magic')
    # For simplicity, we'll stick to the previous methods or fallback.

    # Fallback
    return "jpg"

def download_single_image(url, timeout=10, delay=0.1):
    """Downloads a single image url. Returns (content, hash) or None."""
    time.sleep(delay) # Be polite
    try:
        response = requests.get(url, timeout=timeout)
        if response.status_code == 200:
            img_hash = hashlib.md5(response.content).hexdigest()
            return response.content, img_hash
    except Exception as e:
        print(f"Failed to download from {url}: {e}")
    return None, None


def download_images(person_name, num_images=50, save_folder="images", max_workers=10):
    folder = os.path.join(save_folder, person_name.replace(" ", "_"))
    os.makedirs(folder, exist_ok=True)

    print(f"Searching for high-quality images of {person_name}...")

    urls_to_process = []
    with DDGS() as ddgs:
        # Fetch slightly more URLs initially to account for download failures/duplicates
        results_iterator = ddgs.images(
            keywords=person_name,
            max_results=int(num_images * 1.5), # Fetch more to compensate for failures/duplicates
            safesearch="Off",
            size="Large"
        )
        print("Gathering image URLs...")
        for result in results_iterator:
            url = result.get("image")
            if url:
                urls_to_process.append(url)
            if len(urls_to_process) >= int(num_images * 1.5): # Stop gathering early if enough URLs found
                 break
        print(f"Found {len(urls_to_process)} potential image URLs.")


    seen_hashes = set()
    count = 0
    downloaded_count = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_url = {executor.submit(download_single_image, url): url for url in urls_to_process}

        print(f"Starting download process with {max_workers} workers...")
        for future in as_completed(future_to_url):
            if count >= num_images:
                # Attempt to cancel remaining futures (may not always be possible)
                # This helps to stop unnecessary downloads faster
                for f in future_to_url:
                    if not f.done():
                        f.cancel()
                break # Exit the loop once desired number is reached

            url = future_to_url[future]
            try:
                content, img_hash = future.result()
                if content and img_hash:
                    downloaded_count += 1
                    if img_hash not in seen_hashes:
                        seen_hashes.add(img_hash)
                        ext = get_extension_from_url_or_content(url, content) # Pass content if needed, though not strictly used now
                        file_path = os.path.join(folder, f"{person_name.replace(' ', '_')}_{count}.{ext}")
                        with open(file_path, "wb") as f:
                            f.write(content)
                        count += 1
                        print(f"Downloaded {count}/{num_images} (Processed: {downloaded_count}/{len(urls_to_process)})")
                    # else: # Optional: print duplicate skips
                    #     print(f"Skipped duplicate: {url}")

            except Exception as exc:
                 # This catches exceptions from the future.result() call itself if the task failed unexpectedly
                 print(f'{url} generated an exception: {exc}')


    print(f"✅ Done! Downloaded {count} high-quality images of {person_name} to '{folder}'")

