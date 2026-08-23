# Distributed Big Data Sentiment Mining

An end to end PySpark pipeline that classifies Amazon Automotive product reviews as Positive or Negative, built and evaluated across 19.9 million reviews and 8.3 GB of raw JSON.

The project covers the full lifecycle: distributed ingestion, text preprocessing and TF-IDF feature engineering, four tuned classifiers, Spark performance profiling, model evaluation and robustness testing, and a set of Tableau dashboards built on exported aggregates.

**Dashboards:** [Tableau Public](https://public.tableau.com/app/profile/mohammadadnaniqbal/vizzes)

---

## Dataset

| | |
|---|---|
| Source | [Amazon Reviews 2023, McAuley Lab](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023) |
| Category | Automotive |
| Raw size | 19,955,450 rows, 10 columns, 8.13 GB uncompressed |
| Target | sentiment — 1.0 for 4-5 stars, 0.0 for 1-3 stars |
| Licence | Non commercial academic research use |

The dataset contains no personally identifiable information. User IDs are anonymous system generated identifiers with no link back to real accounts.

### Rows through the pipeline

| Stage | Rows |
|---|---|
| Raw dataset | 19,955,450 |
| After deduplication on user, product, text and timestamp | 19,723,226 |
| After removing reviews left empty by text cleaning | 19,707,232 |

The final labelled set splits roughly 78% Positive to 22% Negative. That imbalance is why AUC and the weighted per class metrics carry more weight here than accuracy, since a model predicting Positive every time would already score 78%.

---

## Repository structure

```
├── 01_data_ingestion.ipynb                    Distributed load, schema inspection, initial exploration
├── 02_preprocessing_and_feature_pipeline.ipynb  Cleaning, labelling, feature engineering, TF-IDF pipeline
├── 03_model_training.ipynb                    Four classifiers tuned with CrossValidator
├── 04_spark_tuning.ipynb                      Resource config, partitioning, caching benchmarks
├── 05_model_evaluation.ipynb                  Metrics, ROC/PR curves, noise robustness, SHAP
├── 06_dashboard_data_export.ipynb             Aggregate CSV extracts for Tableau
├── results/                                   Charts and figures
└── tableau/
    ├── data/                                  Fourteen CSV extracts backing the dashboards
    └── dashboard_screenshots/
```

---

## Feature engineering

Text from the review title and body is cleaned, tokenised, stripped of stop words, hashed to 65,536 TF-IDF dimensions, then combined with five engineered numeric features:

- review_length and title_length — character counts after cleaning
- is_long_review — flag for reviews over 200 characters
- helpful_vote — community helpfulness count
- verified_int — whether the purchase was verified

All six stages run inside a single Spark ML Pipeline, fitted on the training split only so that IDF statistics and scaling parameters never see test data. StandardScaler runs with withMean=False to preserve sparsity across the 65,541 dimension vectors.

---

## Models

All four tuned with CrossValidator and ParamGridBuilder over 2 folds, optimising AUC.

| Model | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| **Logistic Regression** | 0.8697 | 0.8644 | 0.8697 | 0.8576 | **0.9200** |
| Gradient Boosted Trees | 0.8113 | 0.8270 | 0.8113 | 0.7502 | 0.7831 |
| Random Forest | 0.7875 | 0.6202 | 0.7875 | 0.6939 | 0.5705 |
| Naive Bayes | 0.7579 | 0.8097 | 0.7579 | 0.7741 | 0.6407 |

Logistic Regression wins clearly, and the reason is structural rather than incidental. Sentiment in product reviews is carried by a relatively small set of strong signal words, which a linear model over sparse high-dimensional TF-IDF vectors captures directly. The tree ensembles have to approximate the same signal through axis-aligned splits across tens of thousands of mostly-zero dimensions, which is a poor fit for the geometry of the data. It was also the fastest of the four to train.

### Robustness

Each model was re-scored on a test set perturbed with Gaussian noise on every non-zero feature value, measuring how far its metrics drifted.

| Model | AUC drop | Accuracy drop |
|---|---|---|
| Naive Bayes | 0.001 | 0.000 |
| Random Forest | 0.004 | 0.000 |
| Logistic Regression | 0.007 | 0.002 |
| Gradient Boosted Trees | 0.071 | 0.007 |

Logistic Regression combines the strongest headline performance with near-flat degradation under noise. Gradient Boosted Trees is the most brittle, losing an order of magnitude more AUC than the others.

---

## Spark performance work

04_spark_tuning.ipynb profiles the execution side of the pipeline: driver memory sized against the 8.3 GB dataset and its high-dimensional sparse vectors, shuffle partitions tuned for even distribution, repartition versus coalesce compared directly, and MEMORY_AND_DISK caching benchmarked against an uncached baseline with the effect confirmed in the physical execution plan.

---

## Tech stack

Python 3.12 · PySpark 4.0 · Java 17 · scikit-learn · SHAP · matplotlib · Google Colab · Tableau Public · Hugging Face Datasets

---

## Limitations and next steps

**Model training ran on a sample** CrossValidator fits every parameter combination on every fold, which on the full 15.8M row training set would run for hours per algorithm. Training used a fraction of that set so all four algorithms could be compared under identical conditions within a single Colab session. Preprocessing and feature engineering ran on the full dataset, the model metrics above should be read as a relative ranking between algorithms rather than as their ceiling on this data.

**Text cleaning uses a Python UDF.** regexp_replace and lower would do the same work natively in the JVM and avoid serialising every row across the Python boundary. The UDF was retained for readability, but on a production run the native functions would be substantially faster.

**Worth exploring next:** word n-grams rather than unigrams alone, class weighting or resampling to address the 78/22 imbalance directly, and training the best model on the full dataset on a real cluster to establish its actual performance ceiling.

---

## Reproducing

Each notebook runs standalone in Google Colab and installs its own dependencies. Notebooks 2 through 6 read the Parquet output written by notebook 2, so run them in order on a first pass. 04_spark_tuning.ipynb requires an ngrok authtoken to expose the Spark UI, replace the placeholder with your own.
