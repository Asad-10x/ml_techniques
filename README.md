# ml_techniques

This repository showcases techniques and best practices for dataset preprocessing, machine learning algorithms, and model training. It is organized with a clean, logical directory structure for maximum clarity and reproducibility.

---

## Directory Structure

```
src/
├── app.py
├── datasets
│   ├── adult.csv
│   ├── enriched_student_academic_performance_dataset.csv
│   ├── preprocessed_SAP_ds.csv
│   └── processed_adult.csv
├── models
│   ├── ds1
│   │   ├── best_model_random_forest.pkl
│   │   ├── label_encoder.pkl
│   │   ├── minmax_scaler.pkl
│   │   ├── standard_scaler.pkl
│   │   └── top_features.pkl
│   └── ds2
│       ├── daves_bouldin_model.pkl
│       ├── label_encoder.pkl
│       └── top_features.pkl
├── processing
│   ├── ds1
│   │   ├── ds1_pre-processing.ipynb
│   │   ├── prediction.txt
│   │   └── sample.py
│   ├── ds2
│   │   ├── ds2_preprocessing.ipynb
│   │   └── ds2_Student_Academic_Performance_Report
│   └── Feature Extraction
│       └── feature_extraction.ipynb
├── requirements.txt
├── templates
│   ├── index.html
│   └── script.js
└── utils.py
```

---

## Project Overview

- **ds1 (Adult Income Dataset):**  
  Processing notebooks and scripts for the UCI Adult Income dataset are found in `src/processing/ds1/ds1_pre-processing.ipynb`.  
  Outputs: processed datasets and trained models stored in `src/datasets/` and `src/models/ds1/`.

- **ds2 (Student Academic Performance Dataset):**  
  Processing and analysis for student academic performance in `src/processing/ds2/ds2_preprocessing.ipynb`.  
  Outputs: processed datasets and models in `src/datasets/` and `src/models/ds2/`.

- **Feature Extraction (Wine Dataset):**  
  Demonstrates advanced feature extraction using an autoencoder on the Wine dataset in `src/processing/Feature Extraction/feature_extraction.ipynb`.

---

## How to Run

1. **Install Requirements**
   ```
   pip install -r src/requirements.txt
   ```

2. **Process Data & Train Models**  
   Run the relevant notebook(s) in the `src/processing/` folders (`ds1`, `ds2`, or `Feature Extraction`) to generate processed datasets and trained models.

3. **Start the Application**
   ```
   cd src
   python app.py
   ```
   The web UI will be accessible at [http://127.0.0.1:{port}](http://127.0.0.1:{port}) (default port as defined in `app.py`).

---

## Notes

- The `templates/` folder contains the UI files (`index.html`, `script.js`) for the web interface.
- Utility functions are defined in `src/utils.py`.
- All data and model artifacts are stored in the respective `datasets/` and `models/` subfolders.
- For best results, follow the directory structure and execution order as described above.

---

## Datasets Used

- Adult Income Dataset (UCI)
- Student Academic Performance Dataset
- Wine Dataset (scikit-learn)

---

## Contributions

Feel free to fork, open issues, or submit pull requests to enhance functionality or add new ML techniques!

---