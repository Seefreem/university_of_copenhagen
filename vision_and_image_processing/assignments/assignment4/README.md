# Source code files
The code files respect to each question set are named as "assignment4_{question description}.ipynb".

assignment4_TF_IDF.ipynb
assignment4_explore_different_k.ipynb
assignment4_common_words_and_bhattacharyya_distance.ipynb


# Data set
https://data.caltech.edu/records/mzrjq-6wc02

# Software Environment:
Ubuntu 20.04 LTS
Python 3.11.4
opencv-python 4.8.1.78
numpy 1.23.5
scipy 1.10.1
matplotlib 3.7.1
scikit-learn 1.3.0
scikit-learn-intelex 20230426.111612

# Time consuming on full data
It takes about 39 min to run one experiment.

# Evaluation on full data set
## TF-IDF
parameters:
k = 1000
categories: 102
training set size: 4544 
test set size: 4601
Evaluation of test data: mean reciprocal rank: 0.329, percentage of the correct category in top 3: 35.253%
