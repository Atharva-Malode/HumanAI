import os
import torch
import timm
import numpy as np
import cv2
from facenet_pytorch import MTCNN
from torchvision import transforms
from tqdm import tqdm
from PIL import Image

class FeatureExtractor:
    """
    Extracts features from images using DINO and ConvNeXt models. 
    Detects faces using MTCNN and processes images after resizing to 224x224.
    """
    def __init__(self, image_dir, features_dir="features", device=None):
        self.image_dir = image_dir
        self.features_dir = features_dir
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")

        self.dino_model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16').to(self.device).eval()
        self.convnext_model = timm.create_model('convnext_base', pretrained=True, num_classes=0).to(self.device).eval()
        self.face_detector = MTCNN(min_face_size=2, margin=10, thresholds=[0.4, 0.4, 0.4], device=self.device)

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        for subdir in ["with_face/dino", "with_face/convnext", "without_face/dino", "without_face/convnext"]:
            os.makedirs(os.path.join(self.features_dir, subdir), exist_ok=True)

    def detect_face(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boxes, _ = self.face_detector.detect(image_rgb)
        if boxes is not None and len(boxes) > 0:
            x_min, y_min, x_max, y_max = map(int, boxes[0])
            return image[y_min:y_max, x_min:x_max]
        return None

    def extract_features(self, image_array, model):
        if image_array is None or image_array.size == 0:
            return None
        image_pil = Image.fromarray(cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB))
        img_tensor = self.transform(image_pil).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return model(img_tensor).cpu().numpy().flatten()

    def process_images(self):
        image_files = [f for f in os.listdir(self.image_dir) if f.endswith(".jpg")]
        print(f"Processing {len(image_files)} images...")
        
        for image_name in tqdm(image_files, desc="Extracting Features"):
            image_path = os.path.join(self.image_dir, image_name)
            image = cv2.imread(image_path)
            if image is None:
                continue

            face = self.detect_face(image)
            if face is not None:
                dino_face = self.extract_features(face, self.dino_model)
                convnext_face = self.extract_features(face, self.convnext_model)
                if dino_face is not None:
                    np.save(os.path.join(self.features_dir, "with_face/dino", image_name.replace(".jpg", ".npy")), dino_face)
                if convnext_face is not None:
                    np.save(os.path.join(self.features_dir, "with_face/convnext", image_name.replace(".jpg", ".npy")), convnext_face)
            
            dino_no_face = self.extract_features(image, self.dino_model)
            convnext_no_face = self.extract_features(image, self.convnext_model)
            if dino_no_face is not None:
                np.save(os.path.join(self.features_dir, "without_face/dino", image_name.replace(".jpg", ".npy")), dino_no_face)
            if convnext_no_face is not None:
                np.save(os.path.join(self.features_dir, "without_face/convnext", image_name.replace(".jpg", ".npy")), convnext_no_face)

        print("Feature extraction completed.")

# Usage
image_dir = r"C:\Users\athar\OneDrive\Desktop\Atharva\Github\open_source\humandArt\task2\data"
extractor = FeatureExtractor(image_dir)
extractor.process_images()
