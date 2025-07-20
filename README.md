# 🧬 Pancreatic Cancer Survival Prediction Project

## Introduction

Pancreatic cancer is one of the most lethal malignancies worldwide, with a five-year survival rate of less than 10%. This is largely due to late-stage diagnosis and the aggressive nature of the disease. Predicting survival at diagnosis remains a major challenge in oncology and limits clinicians’ ability to personalize care and allocate treatments effectively.

With the increasing availability of clinical and genomic data, machine learning offers a promising approach to improving the accuracy of survival prediction and supporting evidence-based decision-making in pancreatic cancer management.

---

## Problem Statement

Despite the rich availability of clinical and genomic data, survival prediction in pancreatic cancer is still largely based on general staging systems. There is a lack of robust tools that can personalize prognosis for individual patients.

This project aims to develop a machine learning model that predicts survival outcomes for pancreatic cancer patients using a curated clinical-genomic dataset. The ultimate goal is to improve risk stratification and aid clinical decision-making.

---

## Project Objectives

- Extract and isolate pancreatic cancer cases from the broader MSK-CHORD dataset (25,000+ patients).
- Clean and preprocess clinical, pathological, and genomic features 
- Build and evaluate machine learning models to classify survival outcomes (Alive vs. Deceased).
- Interpret feature importance to identify potential prognostic biomarkers.
- Demonstrate how the model could support personalized treatment and early risk counseling.


## Dataset: MSK-CHORD (Nature 2024)

The dataset is sourced from the MSK-CHORD 2024 clinical-genomic database, comprising over 25,000 tumors from 24,950 patients sequenced via **MSK-IMPACT**.

- Genomic + Clinical data
- Features include age, stage, TMB, tumor purity, mutation count, sample class, and more
- Subset: 3,109 pancreatic cancer patient records
- License: [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/)
- Data Source – cBioPortal](https://www.cbioportal.org/study/summary?id=msk_chord_2024)


## Data Preprocessing

- Filtered the dataset to only include records where is Pancreatic
- Handled missing values and selected relevant clinical and molecular features
- Encoded categorical values and scaled numeric columns
- Defined target variable which is the Overall Survival Status

## Exploratory Data Analysis (EDA)

- Distribution of survival status and tumor stage
- Age, Tumor Mutational Burden (TMB), Tumor Purity trends
- Correlation heatmaps of numeric features
- Visual exploration of top predictors by class

## Potential Impact

- Improved risk stratification at diagnosis
- Enhanced treatment planning
- Better patient counseling and clinical trial selection




