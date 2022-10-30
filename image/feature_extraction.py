import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
import numpy as np
from scipy.spatial import distance

class FeatureModel:
    """ Simple class to compare images similarities with EfficientNet-Lite (B0) model feature extraction"""

    def __init__(self):
        self.IMAGE_SHAPE = (224,224)
        model_url = "https://tfhub.dev/tensorflow/efficientnet/lite0/feature-vector/2"
        layer = hub.KerasLayer(model_url, input_shape=(self.IMAGE_SHAPE + (3,)))
        self.model = tf.keras.Sequential([layer])

    def extract_features(self, file):
        """
        Extract features from image file using NN model.

        Args:
            file (str): path to image file to be extracted.

        Returns:
            float []: flattened features of image.
        """
        file = Image.open(file).convert('L').resize(self.IMAGE_SHAPE)  #1
        file = np.stack((file,)*3, axis=-1)                       #2
        file = np.array(file)/255.0                               #3

        features = model.predict(file[np.newaxis, ...])
        features_np = np.array(features)
        flattended_features = features_np.flatten()
        return flattended_features
    
        
    def compare_images(self, path_1, path_2, metric = 'cosine'):
        """
        Compare two images using features extracted by NN model given a metric.

        Args:
            path_1 (str): path to image compared.
            path_2 (str): path to image compared.
            metric (str): distance metric (scipy cdist) used to calculate between image features (default is cosine). 

        Returns:
            numpy.ndarray : distance between images.
        """
        features_1 = self.extract_features(path_1)
        features_2 = self.extract_features(path_2)
        dis = distance.cdist([features_1], [features_2], metric)[0]
        return dis

