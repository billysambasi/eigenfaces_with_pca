# Eigenfaces with PCA

A face recognition system built from scratch using Principal Component Analysis (PCA) on the Labeled Faces in the Wild (LFW) dataset.

---

## Project Overview

This project explores the mathematics behind eigenfaces — a classical approach to face recognition. It covers the full pipeline from raw images to classification, with a focus on understanding eigenvalues, eigenvectors, and PCA both theoretically and practically.

---

## Dataset

- **Source:** Labeled Faces in the Wild (LFW) via `sklearn.datasets.fetch_lfw_people`
- **Samples:** 4324 face images
- **Image size:** 50×37 pixels (resized to 40% of original)
- **Classes:** 158 people

---

## Project Pipeline

### 1. Data Loading & Visualization
- Loaded the LFW dataset with `min_faces_per_person=10` and `resize=0.4`
- Visualized sample faces and their labels

### 2. Flattening
- Reshaped each 2D image (50×37) into a 1D vector of 1850 pixels
- Resulting shape: `(4324, 1850)`

### 3. Mean-Centering
- Computed the mean face across all 4324 images
- Subtracted the mean face from every image to center the data
- Visualized the mean face

### 4. PCA & Eigenfaces
- Applied PCA with 100 components to extract the top eigenfaces
- Visualized the top 10 eigenfaces
- Each eigenface represents a direction of maximum variance in face space

### 5. Face Reconstruction
- Reconstructed faces using increasing numbers of components (k=5, 20, 50, 100, 1850)
- Demonstrated how reconstruction quality improves with more components

### 6. Explained Variance
- Plotted cumulative explained variance vs number of components
- Identified the point of diminishing returns (~50 components)
- Marked the 95% variance threshold

### 7. Manual Eigenvalue/Eigenvector Computation
- Computed the covariance matrix manually using `numpy.cov`
- Extracted eigenvalues and eigenvectors using `numpy.linalg.eig`
- Sorted by descending eigenvalue
- Verified results match sklearn's PCA components

### 8. Face Recognition
- Split data into 80% train / 20% test
- Projected faces into eigenface space (100 features)
- Trained KNN classifier on PCA-projected faces
- Compared against KNN on raw pixels (1850 features)

---

## Results

| Metric | PCA + KNN | Raw Pixel KNN |
|--------|-----------|---------------|
| Accuracy | 20% | 21% |
| Weighted F1 | 0.18 | 0.18 |
| Features used | 100 | 1850 |
| Feature reduction | 94% | 0% |

PCA matched raw pixel accuracy while using 94% fewer features, demonstrating its power for dimensionality reduction.

---

## Key Insights

- **Eigenvalues** tell us how much variance each direction captures
- **Eigenvectors** represent the directions of maximum variance (eigenfaces)
- **Mean-centering** is critical before applying PCA
- **PCA is not always about improving accuracy** — it is about representing data efficiently while preserving the most important information
- Both models struggled due to heavy class imbalance in the dataset

---

## Requirements

```
numpy
matplotlib
scikit-learn
```

Install with:

```bash
pip install numpy matplotlib scikit-learn
```

---

## How to Run

Open `index.ipynb` in Jupyter Notebook or JupyterLab and run all cells sequentially.

---

## Project Structure

```
eigenfaces_with_pca/
├── index.ipynb   # Main project notebook
└── README.md     # Project documentation
```
