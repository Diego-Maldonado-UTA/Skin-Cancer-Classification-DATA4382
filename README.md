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
| **Size**      | 2,357 images, 9 classes|
 

---

![Image Samples](images/sample_plot.png)
Sample plot of all 9 classes

## Data Preprocessing 
Although both pipelines had a stratified split applied, they differed in several ways.

Deep Learning:
- Apply [DullRazor](https://pubmed.ncbi.nlm.nih.gov/9437554/) algorithm to remove hair follicles
- Resize shortest side to 256 and center crop 224x224
- Normalize pixel data

Machine Learning:
- Combine class labels
- Extract raw features 
- Engineer ABCD features
- Add features to a CSV file and save