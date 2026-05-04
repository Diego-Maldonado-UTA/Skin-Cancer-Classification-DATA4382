# Comparing Machine Learning and Deep Learning on Skin Cancer Classification

## Motivation and Overview
Skin cancer is one of the most common types of cancer diagnosed worldwide. It is also one of the most treatable if detected early enough. Convolutional neural networks (CNNs) have achieved strong performance on clean, balanced datasets, but struggle in realistic medical settings where data is limited and class imbalance is severe. These models also lack interpretability and transparency in their decision making, a critical limitation in the medical field. This project evaluates an alternative pipeline on a small, highly imbalanced dataset that reflects realistic medical datasets. Using handcrafted features based on the ABCD dermatological criteria, this project trained an XGBoost model that acheieved a macro-averaged F1-score of 79%, compared to 52% achieved by a ResNet50 model, while providing interpretable feature importance.

Authors: Diego Maldonado, Sidhantaa Sarna, Tiffany De La Cruz

---
 
## Data
 
| Attribute     | Details                        |
|---------------|--------------------------------|
| **Source**    | [Skin Cancer ISIC Dataset](https://www.kaggle.com/datasets/nodoubttome/skin-cancer9-classesisic)      |
| **Type**      | JPG images   |
| **Size**      | 2,357 images, 9 classes, 782 MB

- - -
## Exploratory Data Analysis

### Class Imbalance
![Visualization 1](images/class_distribution.png)

The graph above visualizes the high class imbalance in the dataset, the main problem with the dataset. In such a small dataset, this imbalance is exacerbated. However, this initial observation inspired our alternative approach. Studies have shown that despite their complexity, deep learning models still underperform on small and imbalanced tabular datasets compared to tree-based models. (Grinsztajn et al., 2022)  

<br>

### Visual Simularity & Artifacts
![Visualization 2](images/sample_plot.png)
A sample plot of images from each class tells us several things. Firstly, many of the classes closely resemble each other, making a classification task even more difficult. Additionally, there are artifacts such as hair follicles and edges of dermoscopic that would distract the CNN. 

<br>

### Varying Image Sizes
<p float="left">
  <img src=images/avg_size.png width="479" />
  <img src=images/size_plot.png width="415" /> 
</p>

These two plots highlight the variability in image sizes in the dataset. Melanoma, nevus, and sebhorreic keratosis had significantly bigger images compared to the other classes. Initial CNN modeling attempts directly resized the images down to 224x224. compressing the larger images and distorting lesion boundaries.

## Data Preprocessing
After applying a stratified split in both pipelines, the following preprocessing steps were taken:

<ins>Deep Learning</ins>
- Apply DullRazor algorithm to remove hair follicles (Lee et al., 1998)
- Resize shortest side of the image to 256
- Center crop an area of 224x224
- Normalizing images 

<ins>Machine Learning</ins>
- Extract raw features from each image
- Engineer ABCD, color moment, color contrast, and color ratio features 
- Save results in a CSV file for later training

## Modeling Choice and Training
Models were chosen based on several factors. MobileNetV2 was chosen as a baseline model given its light-weight architecture and previouse use in skin cancer classification. Advanced models included ResNet50, Random Forest, and XGBoost. ResNet50 is a significantly heavier model and demonstrates how a heavier model performs on a realistic dataset. XGBoost and Random Forest are able to handle class imbalance better than other models and have been used extensively in classical computer vision before the widespread adoption of CNNs. 

### Tools
