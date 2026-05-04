# Comparing Machine Learning and Deep Learning on Skin Cancer Classification

## Motivation and Overview
Skin cancer is one of the most common types of cancer diagnosed worldwide. It is also one of the most treatable if detected early enough. Convolutional neural networks (CNNs) have achieved strong performance on clean, balanced datasets, but struggle in realistic medical settings where data is limited and class imbalance is severe. These models also lack interpretability and transparency in their decision making, a critical limitation in the medical field. This project evaluates an alternative pipeline on a small, highly imbalanced dataset that reflects realistic medical datasets. Using handcrafted features based on the ABCD dermatological criteria, this project trained an XGBoost model that acheieved a macro-averaged F1-score of 79%, compared to 52% achieved by a ResNetV2 model, while providing interpretable feature importance.

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
Initial analysis of the images reveals several insights:

![Visualization 1](images/class_distribution.png)
The graph above visualizes the high class imbalance in the dataset, the main problem with the dataset. In such a small dataset, this imbalance is exacerbated. However, this initial observation inspired our alternative approach. Studies have shown that despite their complexity, deep learning models still underperform on small and imbalanced tabular datasets compared to tree-based models. (Grinsztajn et al., 2022)  

<br>

![Visualization 2](images/sample_plot.png)
A sample plot of images from each class tells us several things. Firstly, many of the classes closely resemble each other, making a classification task even more difficult. Additionally, there are artifacts such as hair follicles and the edges of the dermoscopic camera that need to be removed before training. 

<br>


<p float="left">
  <img src=images/avg_size.png width="400" />
  <img src=images/size_plot.png width="350" /> 
</p>