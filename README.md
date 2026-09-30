# Machine Learning Experiments

This repository contains notebook-based experiments for classifying ICD codes from prepared patient medical record (PMR) feature datasets. It compares K-Nearest Neighbors (KNN) approaches with different distance metrics and a LightGBM multiclass model.

## Experiments

| Notebook | Description |
| --- | --- |
| `knn_single_euclidean.ipynb` | KNN classification using Euclidean distance. The notebook explores multiple prepared datasets and feature-set sizes. |
| `knn_single_jaccard.ipynb` | KNN classification using Jaccard distance, with experiments across prepared datasets and feature sets. |
| `knn_double_euclidean_jaccard.ipynb` | Custom KNN experiments combining Euclidean and Jaccard distance calculations. The notebook uses CuPy and is configured for GPU execution. |
| `lightgbm.ipynb` | Multiclass classification experiments using LightGBM, including feature-set variants and GPU-oriented model configurations. |

The notebooks include data preparation, label encoding, model evaluation, and prediction analysis. Depending on the notebook, evaluation includes accuracy, precision, recall, F1 score, and confusion matrices.

## Repository Structure

```text
.
├── knn_double_euclidean_jaccard.ipynb
├── knn_single_euclidean.ipynb
├── knn_single_jaccard.ipynb
├── lightgbm.ipynb
└── README.md
```

## Data

The CSV datasets are not included in this repository. The notebooks expect prepared training and test CSV files, with feature columns and an ICD target column such as `tp_ICD_cl0`.

The data-loading cells currently use Google Colab paths under:

```text
/content/drive/MyDrive/IF6099-TESIS/References/Bootcamp/Final Experiment/dataset/
```

The expected subfolders and filenames differ by experiment and feature-set variant. Place the dataset files at the paths expected by the notebook, or update the relevant `pd.read_csv(...)` cells to point to your own files. Keep the train and test splits consistent when comparing models.

## Requirements

- Python 3
- Jupyter Notebook/Lab or Google Colab
- `pandas`, `numpy`, `scikit-learn`, and `matplotlib`
- `lightgbm` for `lightgbm.ipynb`
- `tqdm` and a CUDA-compatible `cupy` installation for `knn_double_euclidean_jaccard.ipynb`

For a basic local notebook environment, install the common packages with:

```bash
python -m pip install jupyterlab pandas numpy scikit-learn matplotlib lightgbm tqdm
```

Install CuPy separately using the package that matches your CUDA version. GPU-enabled LightGBM also requires a LightGBM build and GPU/OpenCL environment that support the notebook's `device_type="gpu"` configuration. GPU availability and setup depend on the local machine or hosted runtime.

## Running the Notebooks

### Google Colab

1. Open a notebook in Google Colab.
2. Make the required CSV datasets available in Google Drive at the expected paths, or update the data-loading cells.
3. Run the notebook cells from top to bottom. When prompted, authorize Google Drive access.
4. For the double-distance KNN and GPU-configured LightGBM experiments, select a runtime with compatible GPU support.

### Local Jupyter

1. Install the requirements and start JupyterLab:

   ```bash
   jupyter lab
   ```

2. Open the notebook you want to run.
3. Update its data paths to match the local dataset location, then run the cells in order.
4. Configure the required GPU dependencies before running notebooks that use CuPy or GPU-enabled LightGBM.

## Notes

- These are exploratory notebooks rather than a packaged application or command-line pipeline.
- Dataset variants contain different numbers of target classes and feature columns, so results are only directly comparable when the data split and preprocessing are aligned.
- Some notebooks contain Google Colab-specific code and absolute paths that need adjustment for local execution.
- The combined-distance KNN notebook explicitly expects GPU use; GPU requirements do not necessarily apply to the single-distance KNN notebooks.