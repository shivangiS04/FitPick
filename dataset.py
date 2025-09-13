import kagglehub
import os
import pandas as pd

# 1. Download dataset (cached locally if already present)
path = kagglehub.dataset_download("paramaggarwal/fashion-product-images-dataset")
print("Path to dataset files:", path)

# 2. Try to locate styles.csv and images folder inside KaggleHub path
csv_path = os.path.join(path, "styles.csv")
image_dir = os.path.join(path, "images")

if not os.path.exists(csv_path):
    # In case KaggleHub doesn't have styles.csv, check subfolders
    found_csv = None
    for root, _, files in os.walk(path):
        if "styles.csv" in files:
            found_csv = os.path.join(root, "styles.csv")
            image_dir = os.path.join(root, "images")  # assume images nearby
            break
    if found_csv:
        csv_path = found_csv
    else:
        # Fallback to ./data (manual download needed here)
        print("⚠️ styles.csv not found in KaggleHub path. Falling back to ./data/")
        csv_path = os.path.join("data", "styles.csv")
        image_dir = os.path.join("data", "images")

# 3. Load metadata
df = pd.read_csv(csv_path, on_bad_lines="skip")
print("Metadata shape:", df.shape)
print("First 5 records:\n", df.head())

# 4. Images directory
if os.path.exists(image_dir):
    print("Number of images available:", len(os.listdir(image_dir)))
else:
    print("⚠️ Images directory not found:", image_dir)

# 5. Example: first image
sample_id = df.iloc[0]["id"]
sample_img_path = os.path.join(image_dir, f"{sample_id}.jpg")
print("Sample image path:", sample_img_path, "| Exists:", os.path.exists(sample_img_path))
