# Import libraries
import pandas as pd
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import seaborn as sns

# Directory containing the dataset
dataset_dir = 'Dataset'

# Load dataset
def load_dataset(dataset_dir):
    categories = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']
    data = []
    
    for category in categories:
        path = os.path.join(dataset_dir, category)
        class_num = categories.index(category)
        
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img))
                resized_img = cv2.resize(img_array, (128, 128))  # Resize images for consistency
                data.append([resized_img, class_num])
            except Exception as e:
                pass
    
    return data

data = load_dataset(dataset_dir)

# Convert to DataFrame for easier exploration
df = pd.DataFrame(data, columns=['image', 'label'])

# Display some example images
def display_examples(df, categories, num_examples=5):
    fig, axes = plt.subplots(len(categories), num_examples, figsize=(15, 15))
    
    for i, category in enumerate(categories):
        category_df = df[df['label'] == i]
        for j in range(num_examples):
            ax = axes[i, j]
            ax.imshow(cv2.cvtColor(category_df.iloc[j]['image'], cv2.COLOR_BGR2RGB))
            ax.axis('off')
            if j == 0:
                ax.set_title(category, size=15)
    
    plt.tight_layout()
    plt.show()

categories = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']
display_examples(df, categories)

# Exploratory Data Analysis
def exploratory_data_analysis(df):
    label_counts = df['label'].value_counts().sort_index()
    sns.barplot(x=label_counts.index, y=label_counts.values)
    plt.xticks(ticks=label_counts.index, labels=categories)
    plt.xlabel('Flower Categories')
    plt.ylabel('Number of Images')
    plt.title('Distribution of Flower Images in the Dataset')
    plt.show()

exploratory_data_analysis(df)

# Documentation of KNN Method
knn_documentation = """
K-Nearest Neighbors (KNN) Algorithm:
- KNN is a simple, supervised machine learning algorithm that can be used for classification and regression tasks.
- It operates by finding the K closest data points (neighbors) in the feature space to a given input and assigning the majority class (for classification) or averaging the values (for regression) of these neighbors.
- Key concepts include:
  1. Choice of 'K': The number of neighbors to consider.
  2. Distance Metric: Common choices include Euclidean, Manhattan, and Minkowski distances.
  3. Voting: For classification, the majority class of the K neighbors is assigned to the input.

Challenges in Flower Classification with KNN:
- Variability in image poses, lighting conditions, and backgrounds can impact the algorithm's performance.
- High-dimensional feature spaces require efficient distance calculations and handling of potential overfitting with high values of K.
"""

print(knn_documentation)
