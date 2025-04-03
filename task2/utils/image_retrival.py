import os
import torch
import timm
import numpy as np
import cv2
from torchvision import transforms
from PIL import Image
from facenet_pytorch import MTCNN
from sklearn.neighbors import NearestNeighbors

class ImageRetrieval:
    """
    Image Retrieval class that extracts features using DINO and ConvNeXt models,
    detects faces (if enabled), and finds similar images based on feature embeddings.
    """
    def __init__(self, feature_dir, model_name="both", use_face=False, device=None):
        self.feature_dir = feature_dir
        self.model_name = model_name.lower()
        self.use_face = use_face
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")

        self.models = {}
        if self.model_name in ["dino", "both"]:
            self.models["dino"] = torch.hub.load('facebookresearch/dino:main', 'dino_vits16').eval().to(self.device)
        if self.model_name in ["convnext", "both"]:
            self.models["convnext"] = timm.create_model('convnext_base', pretrained=True, num_classes=0).eval().to(self.device)

        self.face_detector = MTCNN(min_face_size=2, margin=10, thresholds=[0.4, 0.4, 0.4], device=self.device)

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def load_features(self, model_type):
        feature_subdir = os.path.join(self.feature_dir, "with_face" if self.use_face else "without_face", model_type)
        features, image_names = [], []
        for file in os.listdir(feature_subdir):
            if file.endswith(".npy"):
                features.append(np.load(os.path.join(feature_subdir, file)))
                image_names.append(file.replace(".npy", ".jpg"))
        return np.array(features).astype("float32"), image_names

    def extract_features(self, img, model):
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return model(img_tensor).cpu().numpy().flatten().reshape(1, -1)

    def detect_face(self, image_path):
        image = Image.open(image_path).convert("RGB")
        image = image.resize((224, 224))
        image_cv2 = np.array(image)
        image_cv2 = cv2.cvtColor(image_cv2, cv2.COLOR_RGB2BGR)

        boxes, _ = self.face_detector.detect(image_cv2)
        if boxes is not None and len(boxes) > 0:
            x_min, y_min, x_max, y_max = map(int, boxes[0])
            face = image_cv2[y_min:y_max, x_min:x_max]
            face_pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
            return face_pil  
    
        return None


    def find_similar_images(self, query_image_path, k=5):
        query_img = self.detect_face(query_image_path) if self.use_face else Image.open(query_image_path).convert("RGB")
        if self.use_face and query_img is None:
            return {}

        results = {}
        for model_name, model in self.models.items():
            query_features = self.extract_features(query_img, model)
            db_features, image_names = self.load_features(model_name)

            knn = NearestNeighbors(n_neighbors=k, metric='cosine')
            knn.fit(db_features)
            _, indices = knn.kneighbors(query_features)

            results[model_name] = [image_names[i] for i in indices[0]]

        return results

