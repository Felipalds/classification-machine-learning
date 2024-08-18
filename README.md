# Classification Machine Learning
> Repository made to compare classification models in machine learning.

## Table of contents
* [Introduction](#introduction)
* [Implementation](#implementation)
* [Read the Article](#article)

## 1. Introduction
- The problem is to compare different classification models in machine learning in terms of accuracy.
- Besides, we will compare multiple strategy classifiers.

## 2. Implementation
### 2.1 Descritive Analysis of the Data
- Principal caracteristics of the data;
- Size;
- Dimensionality;
- Origin;
- Types of attributes;
- Average, median, mode, standard deviation, etc;
- Min and max values;
- Correlation between attributes (Pearson or heatmap);
- Explain the data and the problem in question.

### 2.2 Data division
- Train and test data;
- 50% of the data is for train;
- 25% of the data is for validation;
- 25% of the data is for test.

- *Important: the selection of the instances must be 100% random*;
- *Important: you need to shuffle the data before the divisions*;

### 2.3 Model and Calibration of the Hyperparameters
- KNN;
- Decision Tree;
- Naive Bayes;
- SVM;
- MLP;

### 2.4 Model Evaluation
- We will store the accuracy for each model in 20 iterations;
- In the end, we will have 100 accuracy values (5 models * 20 iterations);

| **Classifier** | **Parameters** |
| --- | --- |
| KNN | `distance`, `n_neighbors` |
| Decision Tree | `criterion`, `max_depth`, `min_samples_split`, `min_samples_leaf`|
| Naive Bayes |  |
| SVM | `C`, `kernel` |
| MLP | `hidden_layer_sizes`, `activation`, `max_iter`, `learning_rate`|

### 2.5 Comparative Analysis
- The last step is to compare the models in terms of accuracy;
- Kruskal-Wallis test (5% significance level);
- If we have one model with different comportment, we will apply the Mann-Whitney test (5% significance level).

| **Step** | **KNN** | **Decision Tree** | **Naive Bayes** | **SVM** | **MLP**|
| --- | --- | --- | --- | --- | --- |
| 1 | Acc 1 | Acc 1 | Acc 1 | Acc 1 | Acc 1 |
| 2 | Acc 2 | Acc 2 | Acc 2 | Acc 2 | Acc 2 |
| ... | ... | ... | ... | ... | ... |
| 20 | Acc 20 | Acc 20 | Acc 20 | Acc 20 | Acc 20 |
| **Mean** | Mean Acc KNN | Mean Acc Decision Tree | Mean Acc Naive Bayes | Mean Acc SVM | Mean Acc MLP |

### 2.6 Multiple Strategy Classifiers
- Sum rule;
- Majority vote;
- Bordas count;
> This approach will combine the opnion of the 5 models.

| **Step** | **Sum Rule** | **Majority Vote** | **Bordas Count** |
| --- | --- | --- | --- |
| 1 | Acc 1 | Acc 1 | Acc 1 |
| 2 | Acc 2 | Acc 2 | Acc 2 |
| ... | ... | ... | ... |
| 20 | Acc 20 | Acc 20 | Acc 20 |
| **Mean** | Mean Acc Sum Rule | Mean Acc Majority Vote | Mean Acc Bordas Count |

Here we will have to apply the Kruskal-Wallis test (5% significance level) and the Mann-Whitney test (5% significance level) if necessary.
**Important: one the multiple strategy classifiers use the trustworthiness of the models, save these values.**

### 2.7 Comparing mono and multiple strategy classifiers
- Compare the accuracy of the mono and multiple strategy classifiers;
- Mann-Whitney test (5% significance level).

### 2.8 How to do
- Java or Python
- Sci-kit Learn or Weka

## 3. Team
@Felipalds
@Pfzoz

## 4. Article
