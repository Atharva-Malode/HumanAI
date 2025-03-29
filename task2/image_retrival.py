import os
import torch
import timm
import numpy as np
import cv2
from torchvision import transforms
from PIL import Image
from facenet_pytorch import MTCNN
from sklearn.neighbors import NearestNeighbors

# Environment Variables (Avoid OpenMP issues)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

class ImageRetrieval:
    """
    A class for retrieving similar images using extracted features from DINO and ConvNeXt models.
    Supports both full-image and face-based retrieval.
    """
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

    def find_similar_images(self, query_features, database_features, image_names, k=5):
        knn = NearestNeighbors(n_neighbors=k, metric='cosine')
        knn.fit(database_features)
        _, indices = knn.kneighbors(query_features)
        return [image_names[i] for i in indices[0]]

    def extract_features(self, img, model):
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return model(img_tensor).cpu().numpy().flatten().reshape(1, -1)

    def detect_face(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            return None
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boxes, _ = self.face_detector.detect(image_rgb)
        if boxes is not None and len(boxes) > 0:
            x_min, y_min, x_max, y_max = map(int, boxes[0])
            face = image[y_min:y_max, x_min:x_max]
            return Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
        return None

    def retrieve_similar_images(self, query_image_path, with_face=False, k=5):
        print(f"Retrieving {k} similar images for {query_image_path} (Face Mode: {with_face})...")

        base_dir = "C:\\Users\\athar\\OneDrive\\Desktop\\Atharva\\Github\\open_source\\humandArt\\task2\\features"
        feature_type = "with_face" if with_face else "without_face"
        
        dino_feature_dir = os.path.join(base_dir, feature_type, "dino")
        convnext_feature_dir = os.path.join(base_dir, feature_type, "convnext")

        query_img = self.detect_face(query_image_path) if with_face else Image.open(query_image_path).convert("RGB")
        
        if with_face and query_img is None:
            print("No face detected. Skipping retrieval.")
            return {"dino": [], "convnext": []}
        
        results = {}
        
        if self.model_type in ["dino", "both"]:
            query_features_dino = self.extract_features(query_img, self.dino_model)
            dino_features, dino_images = self.load_features(dino_feature_dir)
            results["dino"] = self.find_similar_images(query_features_dino, dino_features, dino_images, k)

        if self.model_type in ["convnext", "both"]:
            query_features_convnext = self.extract_features(query_img, self.convnext_model)
            convnext_features, convnext_images = self.load_features(convnext_feature_dir)
            results["convnext"] = self.find_similar_images(query_features_convnext, convnext_features, convnext_images, k)
        
        return results

# Example Usage
if __name__ == "__main__":
    query_image = "C:\\Users\\athar\\OneDrive\\Desktop\\Atharva\\Github\\open_source\\humandArt\\task2\\data\\image_106.jpg"
    retrieval = ImageRetrieval(model_type="both")
    results = retrieval.retrieve_similar_images(query_image, with_face=True)
    print("Results:", results)