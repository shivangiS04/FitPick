import kagglehub
import os
import pandas as pd

# 1. Try KaggleHub download (cached locally)
path = kagglehub.dataset_download("paramaggarwal/fashion-product-images-dataset")
print("Dataset downloaded to (kagglehub):", path)

# 2. Check where styles.csv exists
csv_path = os.path.join(path, "styles.csv")
image_dir = os.path.join(path, "images")

if not os.path.exists(csv_path):
    # fallback to Kaggle CLI download folder
    print("⚠️ styles.csv not found in kagglehub cache. Falling back to ./data/")
    csv_path = os.path.join("data", "styles.csv")
    image_dir = os.path.join("data", "images")

# 3. Load metadata
df = pd.read_csv(csv_path, on_bad_lines="skip")
print("Metadata shape:", df.shape)
print("First 5 records:\n", df.head())

# 4. Images directory
print("Number of images available:", len(os.listdir(image_dir)))

# 5. Example: first image
sample_id = df.iloc[0]["id"]
sample_img_path = os.path.join(image_dir, f"{sample_id}.jpg")
print("Sample image path:", sample_img_path, "| Exists:", os.path.exists(sample_img_path))
