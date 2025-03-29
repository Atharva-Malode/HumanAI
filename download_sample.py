import requests

# Image URL
url = "https://api.nga.gov/iiif/27468a27-906e-4441-b019-95fc454544fc/full/!200,200/0/default.jpg"

# Filepath to save the image
save_path = "downloaded_image.jpg"

# Download the image
response = requests.get(url)

if response.status_code == 200:
    with open(save_path, "wb") as file:
        file.write(response.content)
    print(f"✅ Image downloaded successfully: {save_path}")
else:
    print(f"❌ Failed to download image. Status code: {response.status_code}")
