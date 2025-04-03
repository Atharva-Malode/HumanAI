import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm
from PIL import Image

class TSNEVisualizer:
    def __init__(self, feature_dir, image_dir, model_name="dino", save_path="tsne_visualization.png"):
        """
        Initializes the t-SNE visualizer.
        
        Parameters:
        - feature_dir (str): Directory containing .npy feature files.
        - image_dir (str): Directory containing image files.
        - model_name (str): Feature extraction model used ("dino" or "convnext").
        - save_path (str): Path to save the generated t-SNE plot.
        """
        self.feature_dir = feature_dir
        self.image_dir = image_dir
        self.model_name = model_name.lower()
        self.save_path = save_path
        
        self.feature_vectors, self.image_names = self.load_features()
        self.reduced_features = None
    
    def load_features(self):
        """Loads feature vectors from .npy files."""
        features = []
        image_names = []
        
        for file in os.listdir(self.feature_dir):
            if file.endswith(".npy"):
                feature_path = os.path.join(self.feature_dir, file)
                features.append(np.load(feature_path))
                image_names.append(file.replace(".npy", ".jpg")) 
        
        return np.array(features), image_names
    
    def apply_tsne(self, perplexity=30, random_state=42):
        """Applies t-SNE to reduce feature dimensions to 2D."""
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state)
        self.reduced_features = tsne.fit_transform(self.feature_vectors)
        
    def plot_tsne(self, max_size=0.05):
        """Plots the t-SNE visualization with images overlaid and saves it in high resolution."""
        if self.reduced_features is None:
            raise ValueError("t-SNE has not been applied. Call apply_tsne() first.")
        
        # Normalize coordinates
        x_min, x_max = np.min(self.reduced_features, axis=0), np.max(self.reduced_features, axis=0)
        tx = (self.reduced_features[:, 0] - x_min[0]) / (x_max[0] - x_min[0])
        ty = (self.reduced_features[:, 1] - x_min[1]) / (x_max[1] - x_min[1])
        
        fig, ax = plt.subplots(figsize=(20, 20))  # Maximized figure size
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"t-SNE Visualization ({self.model_name.upper()} Features)", fontsize=20)
        
        for image_name, x, y in tqdm(zip(self.image_names, tx, ty), desc="Plotting images", total=len(self.image_names)):
            image_path = os.path.join(self.image_dir, image_name)
            
            if os.path.exists(image_path):
                img = Image.open(image_path)
                img = img.resize((80, 80))  
                
                img_box = ax.inset_axes([x, y, max_size, max_size], transform=ax.transData)
                img_box.imshow(img)
                img_box.set_xticks([])
                img_box.set_yticks([])
        
        # Save the figure before displaying
        plt.tight_layout(pad=0)  # Remove unnecessary padding
        fig.savefig(self.save_path, bbox_inches="tight", dpi=600) 
        print(f"t-SNE visualization saved as: {self.save_path}")
        
        plt.show()  

