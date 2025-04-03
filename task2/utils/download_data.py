import os
import requests
import pandas as pd
from tqdm import tqdm
import cv2
import numpy as np
import time

class DatasetDownloader:
    def __init__(self, save_dir="data", csv_url=None, limit=500, timeout=240, max_retries=3):
        self.csv_url = csv_url or "https://raw.githubusercontent.com/NationalGalleryOfArt/opendata/refs/heads/main/data/published_images.csv"
        self.save_dir = save_dir
        self.limit = limit
        self.timeout = timeout
        self.max_retries = max_retries
        self.failed_log = os.path.join(self.save_dir, "failed_downloads.txt")
        os.makedirs(self.save_dir, exist_ok=True)

    def fetch_csv(self):
        response = requests.get(self.csv_url, timeout=self.timeout)
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
        return df["iiifurl"].dropna().tolist()

    def download_and_resize_image(self, url, index):
        full_url = f"{url}/full/full/0/default.jpg"
        retries = 0

        while True:  
            try:
                response = requests.get(full_url, timeout=self.timeout)
                response.raise_for_status()

                image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
                img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

                if img is None:
                    raise ValueError("Failed to decode image.")

                img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
                file_path = os.path.join(self.save_dir, f"image_{index}.jpg")
                cv2.imwrite(file_path, img)
                return True
            
            except requests.exceptions.Timeout:
                print(f"Timeout for {full_url}. Retrying...")
                time.sleep(5)  
            
            except requests.exceptions.ConnectionError as e:
                print(f"Connection error: {e}. Retrying...")
                time.sleep(5)
            
            except Exception as e:
                print(f"Skipped: {full_url} | Error: {e}")
                with open(self.failed_log, "a") as log_file:
                    log_file.write(f"Skipped: {full_url} | Error: {e}\n")
                return False  

    def download_images(self):
        csv_path = self.fetch_csv()
        if not csv_path:
            return

        image_urls = self.get_image_urls(csv_path)
        total_images = len(image_urls)
        print(f"Total available images in dataset: {total_images}")

        if total_images < self.limit:
            print(f"Warning: Only {total_images} images available, but {self.limit} requested.")

        print(f"Attempting to download {self.limit} images...")

        downloaded_count = 0
        url_index = 0

        with tqdm(total=self.limit, desc="Downloading", unit="img") as pbar:
            while downloaded_count < self.limit and url_index < total_images:
                success = self.download_and_resize_image(image_urls[url_index], downloaded_count)
                if success:
                    downloaded_count += 1
                    pbar.update(1)
                url_index += 1

        print(f"Download complete! Successfully downloaded {downloaded_count}/{self.limit} images.")
        print(f"Check '{self.failed_log}' for any skipped downloads.")


