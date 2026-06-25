# PCA from Scratch Using Math

A from-scratch implementation of Principal Component Analysis (PCA) using only NumPy's linear algebra functions, no `sklearn.decomposition.PCA` involved. The notebook builds the covariance matrix, solves for eigenvalues and eigenvectors by hand, and projects a 3D synthetic dataset down to 2D, then checks what that reduction costs a KNN classifier.

## Why this exists

Most PCA tutorials call `PCA(n_components=2)` and move on. This repo opens that black box. Every step here, mean centering, the covariance matrix, the eigen decomposition, the projection, is written out explicitly so the math behind "dimensionality reduction" is visible instead of assumed.

## What's inside

| File | Description |
|------|-------------|
| `PCA.ipynb` | The full implementation: synthetic 3D dataset, manual PCA pipeline, 3D and 2D visualizations, and a KNN accuracy comparison before/after reduction |

## The math behind PCA

PCA finds the directions in your data that capture the most variance, then re-expresses the data along those directions. Here's the full derivation, step by step.

### 1. Mean centering (and scaling)

PCA is sensitive to the scale of each feature, so the data gets standardized first. For each feature column $j$:

$$
x_{ij}^{\text{scaled}} = \frac{x_{ij} - \mu_j}{\sigma_j}
$$

where $\mu_j$ is the mean and $\sigma_j$ is the standard deviation of feature $j$. This is what `StandardScaler` does in the notebook. Centering matters most here: PCA looks for directions of maximum variance, and an off-center dataset biases that search toward the mean itself instead of the actual spread of the data.

### 2. The covariance matrix

For a dataset $X$ with $n$ samples and $d$ features, the covariance matrix $\Sigma$ is a $d \times d$ matrix where each entry measures how two features vary together:

$$
\Sigma_{jk} = \frac{1}{n-1} \sum_{i=1}^{n} (x_{ij} - \bar{x}_j)(x_{ik} - \bar{x}_k)
$$

In matrix form:

$$
\Sigma = \frac{1}{n-1} X^T X
$$

(with $X$ already mean-centered). The diagonal holds each feature's variance; the off-diagonal entries hold the covariances between feature pairs. This matrix is symmetric and positive semi-definite, which is what guarantees real, non-negative eigenvalues in the next step.

### 3. Eigenvalues and eigenvectors

This is where the variance directions get extracted. An eigenvector $v$ of $\Sigma$ and its corresponding eigenvalue $\lambda$ satisfy:

$$
\Sigma v = \lambda v
$$

Solved from the characteristic equation:

$$
\det(\Sigma - \lambda I) = 0
$$

Each eigenvector points in a direction that the covariance matrix only stretches or shrinks, never rotates. The eigenvalue $\lambda$ tells you how much variance lies along that direction. `np.linalg.eig(cov_matrix)` in the notebook solves both equations directly.

### 4. Selecting the principal components

Sort the eigenvalue/eigenvector pairs by eigenvalue, descending, and keep the top $k$:

$$
\lambda_1 \geq \lambda_2 \geq \dots \geq \lambda_d
$$

The eigenvectors $v_1, \dots, v_k$ paired with the $k$ largest eigenvalues are the principal components. How much of the original variance they preserve is the explained variance ratio:

$$
\text{Explained Variance Ratio}_i = \frac{\lambda_i}{\sum_{j=1}^{d} \lambda_j}
$$

The notebook keeps the top 2 eigenvectors to take the data from 3D down to 2D.

### 5. Projecting the data

Stack the selected eigenvectors into a matrix $W$ (shape $k \times d$), then project the original data onto them:

$$
Z = X W^T
$$

$Z$ is the new dataset in $k$ dimensions, each column a principal component score. This is the `np.dot(df, pc.T)` step in the notebook.

## Walkthrough (matches the notebook)

**Step 0 — Generate data.** `make_blobs` creates 500 points across 3 clusters in 3D, so the result is visible and there's a ground-truth label (`target`) to test classification against.

**Step 1 — Mean center / scale.** `StandardScaler` centers and scales `X_train` and `X_test`.

**Step 2 — Build the covariance matrix.** `np.cov` computes $\Sigma$ from the three feature columns.

**Step 3 — Eigen decomposition.** `np.linalg.eig(cov_matrix)` returns the eigenvalues and eigenvectors of $\Sigma$.

**Step 4 — Select principal components.** The two eigenvectors with the highest eigenvalues become `pc`, the matrix $W$ from the formula above.

**Step 5 — Project.** `np.dot(df.iloc[:, :3], pc.T)` produces `PC1` and `PC2`, the 2D representation of the original 3D data.

## Results

| Setup | Dimensions | KNN Accuracy |
|-------|------------|--------------|
| Original data | 3D | Baseline (see notebook output) |
| After PCA | 2D | Slightly lower than baseline |

The drop in accuracy isn't a bug, it's the cost of throwing away one axis of information. The point isn't to improve accuracy here; it's to show that you can compress 3 columns into 2 and still keep most of the structure a classifier needs. That trade-off matters more as dimensionality grows: scale this up to something like MNIST, where each image is 784 pixels (28×28), and PCA's ability to compress hundreds of columns into a handful, without retraining your intuition for what "the data" means, is the entire reason it's a standard preprocessing step.

## Getting started

```bash
git clone https://github.com/KevalSolanki77/PCA_from_scartch_Using_Math.git
cd PCA_from_scartch_Using_Math
pip install numpy pandas scikit-learn plotly
jupyter notebook PCA.ipynb
```

Or open it directly in [Google Colab](https://colab.research.google.com/) by uploading `PCA.ipynb`.

## Requirements

- numpy
- pandas
- scikit-learn
- plotly

## License

MIT — see [LICENSE](LICENSE) for details.
