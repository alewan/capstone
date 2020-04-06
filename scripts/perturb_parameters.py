# Written by Aleksei Wan on 07.01.2020

from numpy import absolute, round, random as rand


def pull_random_float(mean: float = 0.0, stddev: float = 1) -> float:
    return absolute(rand.normal(mean, stddev, 1)[0])

def pull_random_float_with_ceiling(ceiling: float, mean: float = 0.0, stddev: float = 1) -> float:
    x = pull_random_float(mean, stddev)
    return x if x < ceiling else ceiling

def pull_random_float_with_floor(floor: float, mean: float = 0.0, stddev: float = 1) -> float:
    x = pull_random_float(mean, stddev)
    return x if x > floor else floor

def pull_random_int(mean: int = 0, stddev: float = 1.0) -> int:
    return int(round(pull_random_float(mean, stddev)))

def pull_random_int_with_floor(floor: int, mean: int = 0, stddev: float = 1.0) -> int:
   x = pull_random_int(mean, stddev) 
   return x if x > floor else floor


def generate_bdst_param_pack() -> dict:
    """
    param = {'objective': 'multiclass',
             'num_class': 8,
             'metric': 'multi_logloss',
             'early_stopping_rounds': 100,
             'max_depth': 8,
             'max_bin': 255,
             'feature_fraction': 1,
             'bagging_fraction': 0.65,
             'bagging_freq': 5,
             'learning_rate': 0.020,
             'num_rounds': 1000,
             'num_leaves': 32,
             'min_data_in_leaf': 10,
             'lambda_l1': 0.0001}
    """

    depth = pull_random_int_with_floor(0, 14, 10.0)
    bagging_frac = pull_random_float_with_ceiling(1, 0.75, 0.25)
    bagging_freq = pull_random_int_with_floor(0, 7, 3.0)
    lr = pull_random_float_with_floor(0.0001, 0.015, 0.015)
    l1 = pull_random_float_with_floor(0.00001, 0.005, 0.001)
    leaves = pull_random_int(32, 8.0)
    rounds = pull_random_int(4050, 550.0)
    es_rounds = round(rounds / 20)
    # feature_frac = pull_random_float_with_ceiling(0.8, 0.2, 1)
    min_data_leaf = pull_random_int(20, 5)

    return {'verbose': -1,  # suppress output
            'objective': 'multiclass',
            'num_class': 8,
            'metric': 'multi_logloss',
            'early_stopping_rounds': es_rounds,
            'max_depth': depth,
            'max_bin': 255,
            'feature_fraction': 1,
            'bagging_fraction': bagging_frac,
            'bagging_freq': bagging_freq,
            'learning_rate': lr,
            'num_rounds': rounds,
            'num_leaves': leaves,
            'min_data_in_leaf': min_data_leaf,
            'lambda_l1': l1}
