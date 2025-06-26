# Correct ways to import ImageDataGenerator in newer versions

# Option 1: Recommended for newer versions
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Option 2: Alternative import path
# from keras.preprocessing.image import ImageDataGenerator

# Option 3: Most current approach (TensorFlow 2.6+)
# from tensorflow.keras.utils import image_dataset_from_directory

# Example usage
datagen = ImageDataGenerator(
    horizontal_flip=True,
    vertical_flip=True,
    zoom_range=0.1,
    shear_range=0.1,
    width_shift_range=0.2,
    height_shift_range=0.2,
    rotation_range=90
)

print("ImageDataGenerator imported successfully!")
print("Created datagen with augmentation parameters")
