import kagglehub
import os
import pandas as pd

# 1. Download dataset (only once, then it's cached locally)
path = kagglehub.dataset_download("paramaggarwal/fashion-product-images-dataset")
print("Dataset downloaded to:", path)

# 2. Load metadata (styles.csv)
csv_path = os.path.join(path, "styles.csv")
df = pd.read_csv(csv_path, on_bad_lines="skip")
print("Metadata shape:", df.shape)
print("First 5 records:\n", df.head())

# 3. Point to images directory
image_dir = os.path.join(path, "images")
print("Number of images available:", len(os.listdir(image_dir)))

# 4. Example: get path of first image
sample_id = df.iloc[0]["id"]
sample_img_path = os.path.join(image_dir, f"{sample_id}.jpg")
print("Sample image path:", sample_img_path, "| Exists:", os.path.exists(sample_img_path))
