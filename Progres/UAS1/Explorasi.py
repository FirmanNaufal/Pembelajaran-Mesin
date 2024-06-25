import os
import cv2
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

dataset_dir = "Dataset"

class_counts = {}

def display_sample_images(sample_size=5):
    try:
        for cls in os.listdir(dataset_dir):
            cls_dir = os.path.join(dataset_dir, cls)
            if os.path.isdir(cls_dir):
                image_files = [f for f in os.listdir(cls_dir) if f.endswith('.jpg')]
                sample_images = image_files[:sample_size]
                for img_file in sample_images:
                    img_path = os.path.join(cls_dir, img_file)
                    image = cv2.imread(img_path)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    plt.imshow(image)
                    plt.title(cls)
                    plt.axis('off')
                    plt.show()
    except Exception as e:
        logging.error(f"Error in displaying sample images: {e}")

def count_images_per_class():
    try:
        for cls in os.listdir(dataset_dir):
            cls_dir = os.path.join(dataset_dir, cls)
            if os.path.isdir(cls_dir):
                image_files = [f for f in os.listdir(cls_dir) if f.endswith('.jpg')]
                class_counts[cls] = len(image_files)
        logging.info("Image count per class computed successfully.")
    except Exception as e:
        logging.error(f"Error in counting images per class: {e}")

def plot_class_distribution():
    try:
        plt.bar(class_counts.keys(), class_counts.values())
        plt.xlabel('Kategori Bunga')
        plt.ylabel('Jumlah Gambar')
        plt.title('Distribusi Kelas dalam Dataset')
        plt.xticks(rotation=45)
        plt.show()
    except Exception as e:
        logging.error(f"Error in plotting class distribution: {e}")

count_images_per_class()
display_sample_images()
plot_class_distribution()
