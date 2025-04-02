# ArtExtract

I am **Atharva Malode**, a research fellow at **CSIR-NEERI**, a Government of India organization for environmental research. I enjoy working in the research and machine learning domain.  

Here are my tasks for the project **ArtExtract: Painting in Painting**.

## Task 1: Convolutional-Recurrent Architectures

### Task Description
Build a model based on convolutional-recurrent architectures for classifying **Style, Artist, Genre**, and other attributes. The goal was to select the most appropriate approach, discuss the strategy, and implement it using the [ArtGAN WikiArt dataset](https://github.com/cs-chan/ArtGAN/blob/master/WikiArt%20Dataset/README.md).

### Approach Summary  
The approach combines traditional feature extraction with deep learning techniques:  

1. **Gabor Filters**: Used to extract texture-based features, capturing brush stroke patterns and stylistic elements in the paintings.  
2. **ConvNeXt (Pretrained on ImageNet)**: Extracts high-level visual features from paintings, complementing the handcrafted features from Gabor filters.  
3. **Bidirectional LSTM**: Processes the combined feature sequence from Gabor filters and ConvNeXt, capturing complex relationships in the extracted artistic features.  


The mathematical foundation for the Gabor filter is defined as:

$$g(x,y;\lambda,\theta,\psi,\sigma,\gamma) = \exp\left(-\frac{x'^2+\gamma^2y'^2}{2\sigma^2}\right)\exp\left(i\left(2\pi\frac{x'}{\lambda}+\psi\right)\right)$$

Where:
- $x' = x\cos\theta + y\sin\theta$
- $y' = -x\sin\theta + y\cos\theta$
- $\lambda$ represents wavelength
- $\theta$ represents orientation
- $\psi$ is phase offset
- $\sigma$ is standard deviation
- $\gamma$ is spatial aspect ratio

### Evaluation Metrics  
To evaluate the model's performance, **accuracy** was used as the primary metric. A detailed explanation of the model implementation and its results can be found in the **Task 1 README file**.  

### Results  
The model achieved the following classification accuracy:  
- **Artist Classification**: **0.63**  
- **Genre Classification**: **0.70**  
- **Style Classification**: **0.40**  

*To handle class imbalance in style classification, class-weighted loss was assigned to underrepresented styles.*  

Confusion matrices for all three classification tasks can be found in the **results** folder.  


### Folder Structure
```
Task1/
│── results/              # Results for all models  
│   ├── artist/ ── [confusion_matrix.png, scores.txt, classification_report.txt]  
│   ├── genre/  ── [confusion_matrix.png, scores.txt, classification_report.txt]  
│   ├── style/  ── [confusion_matrix.png, scores.txt, classification_report.txt]  
│
│── utils/                # Files used for training and evaluation  
│   ├── [data_preprocessing.py, model_architecture.py, training_helpers.py]  
│
│── weight/               # Saved model weights  
│   ├── artist/ ── [epoch_1.pth, epoch_2.pth, ...]  
│   ├── genre/  ── [epoch_1.pth, epoch_2.pth, ...]  
│   ├── style/  ── [epoch_1.pth, epoch_2.pth, ...]  
│   ├── final_weight/ ── [artist.pth, genre.pth, style.pth]  
│
│── Wikiart Dataset/      # Original data CSV files  
│── evaluation.ipynb      # Model evaluation is done here  
│── train.py              # File used for training  
└── README.md             # Implementation documentation  
```

## Task 2: Similarity

### Task Description
Develop a model to find similarities in paintings, such as identifying **portraits with a similar face or pose**. The approach and methodology were discussed, and the model was implemented using the [National Gallery Of Art open dataset](https://github.com/NationalGalleryOfArt/opendata).

### Approach Summary  
The similarity search was performed using feature extraction methods and various similarity metrics. The pipeline involved the following steps:  

1. **Feature Extraction**:  
   - **ConvNeXt (trained on ImageNet)** and **DINO** were used for feature extraction.  
   - The model selection (ConvNeXt or DINO) was configurable and had to be specified before extraction.  
   - If **face-based similarity** was enabled, **Mask R-CNN** was used to detect faces before feature extraction.  

2. **Similarity Computation**:  
   - The extracted features were compared using similarity metrics like **cosine similarity** and **SSIM (Structural Similarity Index Measure)**.  

3. **Visualization**:  
   - The similarity relationships between images were mapped using **t-SNE**, providing a 2D visual representation.  
   - The visualization was generated using the **t-SNE visualizer function** from the `utils` module.  

#### Image Similarity (T-distributed Stochastic Neighbor Embedding)  
![t-SNE Visualization](assets/task2/tsne_visualization.png)  


# Feature Extraction and Similarity Metrics

## 1. Feature Extraction
- **ConvNeXt**: A modern convolutional network architecture for extracting visual features  
- **DINO**: Self-supervised vision transformer for robust feature extraction  

## 2. Similarity Metrics
- **SSIM (Structural Similarity Index)**: Measures the perceived similarity between two images  
- **RMSE (Root Mean Square Error)**: Measures the absolute differences between images  
- **LPIPS (Learned Perceptual Image Patch Similarity)**: Measures perceptual differences using deep features  

### SSIM Definition  

$$\text{SSIM}(x, y) = \frac{(2\mu_x\mu_y + c_1)(2\sigma_{xy} + c_2)}{(\mu_x^2 + \mu_y^2 + c_1)(\sigma_x^2 + \sigma_y^2 + c_2)}$$  

Where:  
- $\mu_x$ is the average of x  
- $\mu_y$ is the average of y  
- $\sigma_x^2$ is the variance of x  
- $\sigma_y^2$ is the variance of y  
- $\sigma_{xy}$ is the covariance of x and y  
- $c_1$ and $c_2$ are constants to stabilize division  

## Evaluation Results
For the evaluation, 50 images were selected and analyzed using both feature extraction methods:  
- **Average SSIM - ConvNeXt**: 0.2045  
- **Average SSIM - DINO**: 0.2045  

The other metrics used and their results are provided in detail in the Task 2 README file.


### Folder Structure
```
Task2/
│── features/             # features in npy for fast processing  
│
│── utils/                # Files used for data and features extraction  
│   ├── [download_data.py, extract_features.py, image_retrival.py, visualize.py]  
│
│── tutorial.ipynb/       # notebook to test the work   
│
└── README.md             # Implementation documentation  
```
---

## Future Work
There are several directions for improving and extending this work:  
1. **Temporal Data Utilization**: Obtain the year for all paintings and incorporate temporal data to enhance Task 1, making it more historically aware.  
2. **Domain-Specific Feature Extraction**: Instead of using general ImageNet-based models, fine-tune a CNN specifically on paintings to extract more domain-relevant features.  
3. **Efficient Similarity Search**: Replace KNN with **FAISS** for Task 2 to enable faster and more scalable image retrieval.  
4. **Improved Face Feature Extraction**: Further fine-tune the face detection model on paintings to better capture fine details like noses, ears, and other facial features unique to artwork.  

This was an exciting challenge, and I truly enjoyed implementing and learning from it! Thank you again for this opportunity. 🚀  
