# Brain Tumor MRI Classification with InceptionV3

A TensorFlow/Keras notebook that classifies brain MRI images into four categories using transfer learning with **InceptionV3**. It was written for Kaggle but can be adapted to other environments.

## Classes

| Class | Description |
|-------|-------------|
| `glioma` | Glioma tumor |
| `meningioma` | Meningioma tumor |
| `pituitary` | Pituitary tumor |
| `notumor` | No tumor |

## Dataset

- Source: a Kaggle brain tumor MRI dataset (`abdullahalaminifaz/tumor-classification-data`).
- Expected folder structure:
  ```
  <dataset root>/
  ├── Training/
  │   ├── glioma/
  │   ├── meningioma/
  │   ├── notumor/
  │   └── pituitary/
  └── Testing/
      ├── glioma/
      ├── meningioma/
      ├── notumor/
      └── pituitary/
  ```
- Training: 1,400 images per class (5,600 total). Testing: 400 images per class (1,600 total).
- The training set is split 80/20 (stratified, seed 42) into **4,480 train** and **1,120 validation** images.

## Pipeline Overview

1. **Setup**: imports, random seeds (42), GPU detection and memory growth.
2. **Data discovery**: automatically locates the `Training`/`Testing` folders under `/kaggle/input` and builds file/label DataFrames.
3. **Data pipeline**: `tf.data` with image decoding, resizing to 299×299, batching (32), and prefetching.
4. **Augmentation**: random horizontal flip, rotation (0.08), zoom (0.10), and contrast (0.10), applied inside the model.
5. **Model**: InceptionV3 (ImageNet weights, no top) → GlobalAveragePooling → Dropout(0.35) → Dense(4, softmax).
6. **Phase 1 – Feature extraction**: base frozen, Adam lr = 1e-3, up to 10 epochs.
7. **Phase 2 – Fine-tuning**: unfreezes the last ~30% of InceptionV3 layers (BatchNorm layers kept frozen), Adam lr = 1e-5, up to 10 epochs.
8. **Callbacks**: `ModelCheckpoint` (best val loss), `EarlyStopping` (patience 5, restores best weights), `ReduceLROnPlateau` (factor 0.2, patience 2).
9. **Evaluation**: loss/accuracy on the Testing folder, classification report, and confusion matrix.
10. **Inference helpers**: single-image prediction, whole-test-folder predictions saved to CSV, and a reusable `predict_folder()` for any unlabelled folder.

## Results

Observed in the notebook run:

| Metric | Value |
|--------|-------|
| Final validation accuracy | ~96.4% |
| Test accuracy | ~93.1% |
| Test loss | ~0.40 |

Per-class test results: pituitary and no-tumor are the strongest; glioma has the lowest recall (~0.80), most often confused with meningioma or no-tumor.

## Requirements

- Python 3.x
- TensorFlow 2.x (run with 2.21.0)
- numpy, pandas, matplotlib, scikit-learn, Pillow

```bash
pip install tensorflow numpy pandas matplotlib scikit-learn pillow
```

A GPU is recommended; the notebook also runs on CPU, more slowly.

## Usage

### On Kaggle
1. Open the notebook and add the brain tumor MRI dataset via **Add Input**.
2. Run all cells. Internet access is needed to download ImageNet weights (otherwise the code falls back to random initialization, which will hurt accuracy).

### Elsewhere
The data-discovery cell raises an error if `/kaggle/input` is missing. Edit it to set `DATA_ROOT`, `TRAIN_DIR`, and `TEST_DIR` manually. To use local weights, set `LOCAL_WEIGHTS_PATH` in the model cell.

### Predicting
- **Single image**: set `NEW_IMAGE_PATH` in the prediction cell and run it.
- **Unlabelled folder**:
  ```python
  results = predict_folder("/path/to/scans")
  display(results.head())
  ```
  Results are saved to `folder_predictions.csv`.

## Outputs

- `model_output/best_inceptionv3.keras`: best checkpoint (lowest validation loss).
- `model_output/whole_testing_folder_predictions.csv`: per-image predictions with confidence and correctness.
- `folder_predictions.csv`: output of `predict_folder()`.

## Notes and Limitations

- Training and validation images are split by file, so near-duplicate or same-patient slices could appear in both sets, which may inflate validation scores. Test accuracy is the more reliable estimate.
- The final cells re-use `model` and a few variables from earlier cells; run the notebook top to bottom.
- This project is for research and learning purposes only. It is **not** a medical device and must not be used for clinical diagnosis.
