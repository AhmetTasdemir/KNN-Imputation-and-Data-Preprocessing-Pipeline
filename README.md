# KNN Imputation and Data Preprocessing Pipeline

## Overview

This notebook demonstrates the process of handling missing values in a dataset using K-Nearest Neighbors (KNN) imputation. Additionally, it scales the dataset, performs hyperparameter tuning using GridSearchCV, and conducts various statistical analyses to ensure data consistency after imputation.

## Dependencies

Ensure you have the following Python libraries installed before running the notebook:

pip install pandas numpy scikit-learn matplotlib seaborn scipy

## Dataset

The dataset dataset_insilico.csv is loaded into a Pandas DataFrame.

Only relevant columns (excluding the first three) are selected for further processing.

Percentage values are converted to decimal format.

## Preprocessing Steps

Missing Value Handling with KNN Imputation

A pipeline is created with a KNN imputer and a standard scaler.

GridSearchCV is used to find the optimal hyperparameters for imputation.

Best parameters are applied to transform the dataset.

##Statistical Analysis After Imputation

Missing Value Check: Ensures no missing values remain.

Distribution Comparison: Uses histograms to compare original and imputed distributions.

Consistency Check: Compares mean and variance before and after imputation.

Kolmogorov-Smirnov (KS) Test: Checks if the original and imputed data come from the same distribution.

Correlation Analysis: Heatmaps visualize correlations before and after imputation.

## Results

The best hyperparameters for KNN imputation are printed.

The final imputed and scaled dataset is saved as filled_with_knn.csv.

## Usage

To run the notebook:

import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp

Then execute the cells sequentially to perform data imputation and analysis.

## Output Files

filled_with_knn.csv: The final dataset after imputation and scaling.

Visualization

Histograms comparing original vs. imputed data.

Correlation heatmaps for original and imputed datasets.

Author

Created by Ahmet Taşdemir

