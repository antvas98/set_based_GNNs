# Set Based Higher Order Graph Neural Networks

This repository contains a collection of scripts for performing graph classification tasks using the Higher order (multi)set based Graph Neural architectures described in my MSc Thesis project. The scripts are designed to work with datasets from [TUDataset](https://chrsmrrs.github.io/datasets/) and implement various steps of the pipeline, including dataset transformation, GNN architecture implementation, hyperparameter selection, and cross-validation.

## Files

1. `input_dataset.py`: This file provides functionality to load one of the benchmark datasets as input for the transformation and classification tasks.

2. `transform_datasets.py`: This file contains a function that transforms datasets to their set-multiset based variants. It takes the original dataset as input and produces the transformed dataset (nodes are sets and multisets and neighbor relations are as described in thesis project). 

3. `GNN_architectures.py`: This file implements GNN architectures for graph classification tasks. It takes the transformed datasets as input and predicts the class of each graph. There are different architectures in this file. The GINconv_one_aggregator_Net which can be applied for local (only local neighbors) and non-local (no discrimination between local and non-local neighbors) set and multisets models, the GINconv_two_aggregators_Net which should be applied for delta variants of sets and multisets architectures (discrimination between local and non local neighbors) and the gin_tuple2_Net and gin_tuple3_Net which should be applied for tuple based architectures described [here](https://arxiv.org/abs/1904.01543) (Similarly for the gcn architectures)

4. `hyperparameter_selection.py`: This file splits a dataset into a training and validation set and performs hyperparameter selection based on the validation set. It explores different combinations of hyperparameters to find the optimal settings for a GNN model and a specific dataset selection.

5. `cross_validation_metric.py`: This file implements cross-validation for a given number of folds (default is 10). It performs the hyperparameter selection process from step 4 within each fold to evaluate the model's performance in 10 different test sets.

6. `example_run.py`: A short example that automatically splits a given (transformed) dataset, train and test a $\delta$-2-WL{{ }}-GNN  for a given number of hidden units, epochs.

## Getting Started

To use these scripts, follow these steps:

1. Install the required dependencies mentioned in the `requirements.txt` file.
