import os
import cv2
import numpy as np
import logging
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ImagePreprocessor:
    def __init__(self, dataset_dir, save_dir):
        self.dataset_dir = dataset_dir
        self.save_dir = save_dir
        self.data = []
        self.labels = []
        self.image_size = (100, 100)
    
    def resize_image(self, image):
        resized_image = cv2.resize(image, self.image_size)
        return resized_image
    
    def normalize_image(self, image):
        normalized_image = image / 255.0
        return normalized_image
    
    def augment_image(self, image):
        augmented_images = [
            cv2.flip(image, 1),
            cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE),
            cv2.rotate(image, cv2.ROTATE_180),
            cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        ]
        return augmented_images
    
    def preprocess_image(self, image, label):
        resized_image = self.resize_image(image)
        normalized_image = self.normalize_image(resized_image)
        augmented_images = self.augment_image(normalized_image)
        return augmented_images, label
    
    def save_processed_data(self):
        np.save(os.path.join(self.save_dir, 'processed_data.npy'), self.data)
        np.save(os.path.join(self.save_dir, 'labels.npy'), self.labels)
        logging.info("Hasil pra-pemrosesan berhasil disimpan.")
    
    def load_data(self):
        try:
            for folder in os.listdir(self.dataset_dir):
                folder_path = os.path.join(self.dataset_dir, folder)
                if os.path.isdir(folder_path):
                    for file in os.listdir(folder_path):
                        file_path = os.path.join(folder_path, file)
                        if file.endswith('.jpg'):
                            image = cv2.imread(file_path)
                            label = folder
                            augmented_images, label = self.preprocess_image(image, label)
                            for aug_image in augmented_images:
                                self.data.append(aug_image)
                                self.labels.append(label)
            self.data = np.array(self.data)
            self.labels = np.array(self.labels)
        except Exception as e:
            logging.error(f"Error in loading data: {e}")

if __name__ == "__main__":
    dataset_dir = "Dataset"
    save_dir = "SaveData"

    preprocessor = ImagePreprocessor(dataset_dir, save_dir)
    preprocessor.load_data()
    preprocessor.save_processed_data()
