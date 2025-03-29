import cv2
from skimage.metrics import structural_similarity as ssim

def compute_ssim(img1_path, img2_path):
    """Compute SSIM between two images after resizing them to the same dimensions."""
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)  # Convert to grayscale
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

    # Check if images are loaded properly
    if img1 is None or img2 is None:
        print(f"❌ Error loading images: {img1_path} or {img2_path}")
        return None

    # Resize img2 to match img1's size
    img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))  # (width, height)

    # Compute SSIM
    score, _ = ssim(img1, img2, full=True)
    return score

# Example usage
ssim_score = compute_ssim(
    r"C:\Users\athar\OneDrive\Desktop\Atharva\Github\open_source\humandArt\downloaded_image.jpg",
    r"C:\Users\athar\OneDrive\Desktop\Atharva\Github\open_source\humandArt\task2\data\image_366.jpg"
)

if ssim_score is not None:
    print(f"SSIM Score: {ssim_score:.4f}")  # Print score up to 4 decimal places
