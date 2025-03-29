import os
import requests
import pandas as pd
from tqdm import tqdm
import cv2
import numpy as np

class DatasetDownloader:
    """
    A class to download images from the National Gallery of Art open dataset.

    - Downloads images resized to 224x224.
    - Skips any image that fails to download and logs the failure.
    - Saves all images inside the 'data/' folder.
    - Fetches image URLs from the dataset CSV.
    - Supports setting a download limit and timeout.
    """

    def __init__(self, save_dir="data", csv_url=None, limit=500, timeout=10):
        self.csv_url = csv_url or "https://raw.githubusercontent.com/NationalGalleryOfArt/opendata/refs/heads/main/data/published_images.csv"
        self.save_dir = save_dir
        self.limit = limit
        self.timeout = timeout
        self.failed_log = os.path.join(self.save_dir, "failed_downloads.txt")
        os.makedirs(self.save_dir, exist_ok=True)

    def fetch_csv(self):
        response = requests.get(self.csv_url)
        if response.status_code != 200:
            print("Failed to download CSV.")
            return None
        csv_path = os.path.join(self.save_dir, "published_images.csv")
        with open(csv_path, "w", encoding="utf-8") as file:
            file.write(response.text)
        return csv_path

    def get_image_urls(self, csv_path):
        df = pd.read_csv(csv_path)
        if "iiifurl" not in df.columns:
            print("Error: CSV file does not contain 'iiifurl' column.")
            return []
        return df["iiifurl"].dropna().tolist()[:self.limit]

    def download_and_resize_image(self, url, index):
        full_url = f"{url}/full/full/0/default.jpg"
        try:
            response = requests.get(full_url, stream=True, timeout=self.timeout)
            response.raise_for_status()
            image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
            img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Failed to decode image.")
            
            # Resize to 224x224 while preserving aspect ratio
            img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
            
            file_path = os.path.join(self.save_dir, f"image_{index}.jpg")
            cv2.imwrite(file_path, img)
        except requests.exceptions.Timeout:
            print(f"Skipped (Timeout): {full_url}")
            with open(self.failed_log, "a") as log_file:
                log_file.write(f"Timeout: {full_url}\n")
        except Exception as e:
            print(f"Skipped: {full_url} | Error: {e}")
            with open(self.failed_log, "a") as log_file:
                log_file.write(f"Skipped: {full_url} | Error: {e}\n")

    def download_images(self):
        csv_path = self.fetch_csv()
        if not csv_path:
            return
        image_urls = self.get_image_urls(csv_path)
        print(f"Downloading {len(image_urls)} images...")
        for i, url in enumerate(tqdm(image_urls, desc="Downloading")):
            self.download_and_resize_image(url, i)
        print("Download complete!")
        print(f"Check '{self.failed_log}' for any failed downloads.")

# Example usage:
downloader = DatasetDownloader(limit=500)
downloader.download_images()