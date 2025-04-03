# K-Nearest Neighbors (KNN) Algorithm Implementation

This repository contains my implementation of the K-Nearest Neighbors classification algorithm on tje Irs dataset in R. The implementation is applied to the classic Iris dataset to demonstrate the algorithm's functionality.

## Project Overview

The K-Nearest Neighbors (KNN) algorithm is a simple yet powerful classification method that classifies data points based on the majority class of their k nearest neighbors in the feature space. This implementation follows a step-by-step approach to build the algorithm from scratch, understand its components, and evaluate its performance.

## Implementation Steps

### 1. Data Exploration and Preparation
- Load the built-in Iris dataset in R
- Create visualizations to understand data relationships
- Split data into training (80%) and test (20%) sets
- Separate feature variables from the target variable (Species)

### 2. Distance Function Implementation
- Create a custom `dist(p, q)` function that calculates the Euclidean distance between two data points

### 3. Nearest Neighbor Calculation
- For each test point, calculate distances to all training points
- Identify the k nearest neighbors using the `topn()` function from the kit package

### 4. Classification
- Determine the majority class among the k nearest neighbors
- Predict the class of each test observation

### 5. Evaluation
- Apply the KNN function across all test data
- Generate confusion matrix to evaluate performance

## Usage

```r
# Load required libraries
library(kit)  # For topn function

# Load the data
df <- iris

# Visualize the data
# [Your visualization code here]

# Split data into training and test sets
set.seed(10L)  # For reproducibility
test_idx <- sample(1:150, 30)
train_idx <- setdiff(1:150, test_idx)
X_test <- df[test_idx, -5]
y_test <- df[test_idx, 5]
X_train <- df[train_idx, -5]
y_train <- df[train_idx, 5]

# Euclidean distance function
dist <- function(p, q) {
  sqrt(sum((p - q)^2))
}

# KNN classification function
knn <- function(X, X_train, y_train, k) {
  # Calculate distances
  v_dist <- apply(X_train, 1, dist, q = X)
  
  # Find k nearest neighbors
  nn <- topn(v_dist, n = k, decreasing = FALSE)
  
  # Get their classes
  nn_spec <- y_train[nn]
  
  # Return majority class
  sort_tab <- table(nn_spec) %>% sort(decreasing = TRUE)
  names(sort_tab[1])
}

# Apply KNN to all test data
predictions <- apply(X_test, 1, knn, X_train = X_train, y_train = y_train, k = 5)

# Generate confusion matrix
confusion_matrix <- table(predictions, y_test)
print(confusion_matrix)
```

## Requirements
- R (version 4.0.0 or higher recommended)
- Required packages:
  - `kit` (for the `topn()` function)
  - `dplyr` (for data manipulation)
  - `ggplot2` (for visualization, optional)

## Results
The implementation demonstrates how KNN classifies Iris species based on sepal and petal measurements. The confusion matrix shows the performance of the algorithm on the test dataset.

## License
[MIT licence]

## Author
[Michael Kiboi]
