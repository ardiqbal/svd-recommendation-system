from surprise import SVD
from surprise import Dataset
from surprise.model_selection import GridSearchCV
from collections import defaultdict

import numpy as np

def main():
    data = Dataset.load_builtin('ml-100k', prompt=False)
    dataset = data.build_full_trainset()

    # define set of parameters on this dictionary
    param_grid = {
        'n_epochs': np.arange(2,11,2), 
        'lr_all': np.random.uniform(0.002,0.005,5),
        'reg_all': np.arange(0.1,0.6,0.1)
    }

    # Find the best model using gridsearch with 5 kfold
    print('Run grid search, find best model...')
    gs = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=3)
    gs.fit(data)
    print('Best model found!')

    # Train the best model
    print('Train the best model...')
    model = gs.best_estimator['rmse']
    model.fit(dataset)
    print('Done!')

    print('Predict the ratings...')
    test = dataset.build_testset()
    predictions = model.test(test)
    print('Done!')

    recommendations = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        recommendations[uid].append((iid, est))

    for uid, user_ratings in recommendations.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        recommendations[uid] = user_ratings

if __name__ == '__main__':
    main()