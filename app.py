import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
import numpy as np
from numpy.linalg import norm
import os
from tqdm import tqdm
import pickle

# Build ResNet50 model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

model = tensorflow.keras.Sequential([
    base_model,
    GlobalMaxPooling2D()
])

# Feature extraction function
def extract_features(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img, verbose=0).flatten()
    normalized_result = result / norm(result)
    return normalized_result

# ✅ Correct images directory (inside fashion-dataset/)
image_dir = "/Users/shivangisingh/.cache/kagglehub/datasets/paramaggarwal/fashion-product-images-dataset/versions/1/fashion-dataset/images"

if not os.path.exists(image_dir):
    raise FileNotFoundError(f"❌ Image directory not found: {image_dir}")

# Collect filenames
filenames = [os.path.join(image_dir, file) for file in os.listdir(image_dir) if file.endswith(".jpg")]
print("Total images found:", len(filenames))

# Extract features
feature_list = []
for file in tqdm(filenames, desc="Extracting features"):
    try:
        feature_list.append(extract_features(file, model))
    except Exception as e:
        print(f"⚠️ Skipping {file} due to error: {e}")

# Save embeddings + filenames
pickle.dump(feature_list, open("embeddings.pkl", "wb"))
pickle.dump(filenames, open("filenames.pkl", "wb"))

print("✅ Feature extraction completed. Saved 'embeddings.pkl' and 'filenames.pkl'.")
