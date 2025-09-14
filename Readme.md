# FitPick 

**AI-Powered Fashion Recommendation System**

FitPick is a **content-based image recommendation system** for fashion products. It uses a **pre-trained ResNet50 model** to extract deep visual features from images, then applies **nearest neighbor search** to find visually similar products.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![Git LFS](https://img.shields.io/badge/Git_LFS-Enabled-lightgray)

---

##  Features
* Extracts **deep visual embeddings** from fashion product images using ResNet50.
* Stores embeddings in `.pkl` files for **fast retrieval**.
* Finds **top-N similar items** given a query image.
* Interactive UI with **Streamlit**.
* Dataset sourced from [Fashion Product Images Dataset (Kaggle)](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset).

---

## 📂 Project Structure

```bash
FitPick/
│── app.py              # Streamlit app (UI for recommendations)
│── test.py             # Backend test script for recommendation
│── embeddings.pkl      # Precomputed image embeddings
│── filenames.pkl       # Image file paths
│── requirements.txt    # Dependencies
│── README.md           # Project documentation
│── .gitattributes      # Git LFS tracking rules
```


---

##  Dataset
We use the **Fashion Product Images Dataset** (~44k images).

Download with `kagglehub`:
```python
import kagglehub

path = kagglehub.dataset_download("paramaggarwal/fashion-product-images-dataset")
print("Dataset path:", path)

```

## ▶ Usage
1.  **Run backend script (`test.py`):**
    ```bash
    python test.py
    ```
    * Loads a sample query image (`sample/shirt.jpg`).
    * Extracts its feature vector.
    * Finds top-5 visually similar items from the dataset.
    * Displays them one by one using OpenCV.

2.  **Run Streamlit app:**
    ```bash
    streamlit run app.py
    ```
    * Upload any fashion image.
    * See recommended similar items instantly.

##  How It Works
* **Feature Extraction:** Each image is passed through a pre-trained **ResNet50 model** (on ImageNet). The output feature maps are pooled into a 2048-dimensional vector.
* **Normalization:** Embeddings are normalized to a unit length.
* **Similarity Search:** The system uses **K-Nearest Neighbors (KNN)** with Euclidean distance to find the nearest neighbors in the embedding space.
* **Recommendation:** It returns the top-N most visually similar products.

---

###  Example Workflow
1.  A user uploads a T-shirt image.
2.  The system extracts its embedding.
3.  It finds nearest neighbors in the embedding space.
4.  It displays similar T-shirts.

---

##  Requirements
* Python 3.8+
* TensorFlow / Keras
* NumPy
* scikit-learn
* OpenCV
* Streamlit
* kagglehub
* tqdm

**Install with:**
```bash
pip install tensorflow numpy scikit-learn opencv-python streamlit kagglehub tqdm
```
##  Future Improvements
* Support multi-modal search (image + text query).
* Use FAISS for faster similarity search at scale.
* Build a web dashboard for browsing recommendations.
* Deploy with Docker / cloud hosting.

##   Acknowledgements
* Fashion Product Images Dataset (Kaggle)
* ResNet50 for feature extraction
* Streamlit for the UI
