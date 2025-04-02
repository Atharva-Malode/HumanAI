import cv2
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms, models

class WikiArtDataset(Dataset):
    """
    Dataset class for WikiArt paintings, extracting both brushstroke texture (Gabor filters) 
    and high-level semantic features (ConvNeXt) to form an input sequence for a CNN-RNN model.
    """
    
    def __init__(self, csv_file, num_kernels=8, transform=None):
        self.data = pd.read_csv(csv_file)
        print(f"[INFO] Loaded {len(self.data)} samples from {csv_file}")
        
        num_classes = self.data.iloc[:, 1].nunique()
        print(f"[INFO] Dataset contains {num_classes} unique classes")
        
        self.num_kernels = num_kernels
        self.transform = transform if transform else self.default_transform()
        
        print("[INFO] Loading ConvNeXt model...")
        self.convnext = models.convnext_base(weights='IMAGENET1K_V1')
        self.convnext = torch.nn.Sequential(*list(self.convnext.children())[:-2])
        self.convnext.eval()
        print("[INFO] ConvNeXt model loaded successfully")
        
        print("[INFO] Testing feature extraction pipeline...")
        if len(self.data) > 0:
            self._test_feature_pipeline()
    
    def _test_feature_pipeline(self):
        try:
            test_path = self.data.iloc[0, 0]
            test_img = cv2.imread(test_path)
            
            if test_img is None:
                print(f"[WARNING] Could not load test image: {test_path}")
                return
                
            test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
            
            gabor = self.apply_gabor_filters(test_img)
            convnext = self.extract_convnext_features(Image.fromarray(test_img))
            sequence = self.prepare_feature_sequence(test_img)
            
            print(f"[INFO] Gabor features shape: {gabor.shape}")
            print(f"[INFO] ConvNeXt features shape: {convnext.shape}")
            print(f"[INFO] Final sequence shape: {sequence.shape}")
            print(f"[INFO] Feature extraction pipeline verified successfully!")
        except Exception as e:
            print(f"[ERROR] Feature pipeline test failed: {str(e)}")
    
    def default_transform(self):
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def apply_gabor_filters(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        resized_gray = cv2.resize(gray, (7, 7))
        
        feature_maps = []
        for theta in np.linspace(0, np.pi, self.num_kernels):
            kernel = cv2.getGaborKernel((5, 5), 1.0, theta, 5, 0.5, 0, ktype=cv2.CV_32F)
            filtered = cv2.filter2D(resized_gray, cv2.CV_8UC3, kernel)
            feature_maps.append(filtered)
        
        return np.array(feature_maps) 
    
    def extract_convnext_features(self, image):
        image_tensor = self.transform(image).unsqueeze(0)
        with torch.no_grad():
            features = self.convnext(image_tensor)
        return features.squeeze(0).cpu().numpy()
    
    def prepare_feature_sequence(self, image):
        gabor_features = self.apply_gabor_filters(image)
        convnext_features = self.extract_convnext_features(Image.fromarray(image))
        
        _, H, W = convnext_features.shape
        sequence_length = H * W
        
        gabor_seq = gabor_features.reshape(self.num_kernels, -1).T
        convnext_seq = convnext_features.reshape(1024, -1).T
        
        combined_seq = np.concatenate([gabor_seq, convnext_seq], axis=1)
        
        return torch.tensor(combined_seq, dtype=torch.float32)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image_path = self.data.iloc[idx, 0]
        label = self.data.iloc[idx, 1]
        
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise FileNotFoundError(f"Could not load image: {image_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            sequence = self.prepare_feature_sequence(image)
            return sequence, torch.tensor(label, dtype=torch.long)
        except Exception as e:
            print(f"[WARNING] Error processing image {image_path}: {str(e)}")
            return self.__getitem__((idx + 1) % len(self.data))
