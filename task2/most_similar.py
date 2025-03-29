import os
import torch
import timm
import numpy as np
import cv2
from torchvision import transforms
from PIL import Image
from facenet_pytorch import MTCNN
from sklearn.metrics.pairwise import cosine_similarity

# Environment Variables (Avoid OpenMP issues)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

class ImageRetrieval:
    def __init__(self, model_type="both", device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_type = model_type.lower()

        # Load Models
        if self.model_type in ["dino", "both"]:
            self.dino_model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16').eval().to(self.device)
        if self.model_type in ["convnext", "both"]:
            self.convnext_model = timm.create_model('convnext_base', pretrained=True, num_classes=0).eval().to(self.device)
        
        self.face_detector = MTCNN(min_face_size=2, margin=10, thresholds=[0.4, 0.4, 0.4], device=self.device)
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def load_features(self, feature_dir):
        features, image_names = [], []
        for file in os.listdir(feature_dir):
            if file.endswith(".npy"):
                features.append(np.load(os.path.join(feature_dir, file)))
                image_names.append(file.replace(".npy", ".jpg"))
        return np.array(features).astype("float32"), image_names

    def find_highly_similar_images(self, database_features, image_names, threshold=0.90):
        similarity_matrix = cosine_similarity(database_features)
        similar_pairs = []
        num_images = len(image_names)
        
        for i in range(num_images):
            for j in range(i + 1, num_images):
                if similarity_matrix[i, j] >= threshold:
                    similar_pairs.append((image_names[i], image_names[j], similarity_matrix[i, j]))
        
        return similar_pairs

    def find_most_similar_images(self, with_face=False, threshold=0.90):
        print(f"Finding images with at least {threshold * 100}% similarity (Face Mode: {with_face})...")
        
        base_dir = "C:\\Users\\athar\\OneDrive\\Desktop\\Atharva\\Github\\open_source\\humandArt\\task2\\features"
        feature_type = "with_face" if with_face else "without_face"
        
        dino_feature_dir = os.path.join(base_dir, feature_type, "dino")
        convnext_feature_dir = os.path.join(base_dir, feature_type, "convnext")
        
        results = {}
        
        if self.model_type in ["dino", "both"]:
            dino_features, dino_images = self.load_features(dino_feature_dir)
            results["dino"] = self.find_highly_similar_images(dino_features, dino_images, threshold)

        if self.model_type in ["convnext", "both"]:
            convnext_features, convnext_images = self.load_features(convnext_feature_dir)
            results["convnext"] = self.find_highly_similar_images(convnext_features, convnext_images, threshold)
        
        return results

# Example Usage
if __name__ == "__main__":
    retrieval = ImageRetrieval(model_type="both")
    results = retrieval.find_most_similar_images(with_face=True, threshold=0.90)
    print("Highly Similar Image Pairs:", results)