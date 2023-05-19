from sklearn.model_selection import StratifiedKFold
from hyperparameter_selection import hyperparameter_selection_within_fold
import numpy as np

def cross_validation(transformed_data,
                     combinations={'hidden_units':[16,32,64],'lr':[0.001], 'weight_decay':[0.00007]},
                     aggregators='gcn 1',
                     n_splits=10):
    performance = []
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    splits = list(skf.split(X=transformed_data, y=[int(d.y) for d in transformed_data]))
    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        print(f"Test fold: {n_splits-fold_idx}")

        # Create the data loaders for train and validation sets
        train_folds = [transformed_data[i] for i in train_idx]
        test_fold = [transformed_data[i] for i in test_idx]
        performance.append(hyperparameter_selection_within_fold(train_folds, test_fold, batch_size=64, combinations=combinations, aggregators=aggregators))
    return np.mean(np.array(performance)), np.std(np.array(performance)) 

