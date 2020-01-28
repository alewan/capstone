# Written by Aleksei Wan on 07.01.2020

from numpy import absolute, round, random as rand


def pull_random_float(mean: float = 0.0, stddev: float = 1) -> float:
    return absolute(rand.normal(mean, stddev, 1)[0])


def pull_random_int(mean: int = 0, stddev: float = 1.0) -> int:
    return int(round(pull_random_float(mean, stddev)))


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

    depth = pull_random_int(8, 1.0)
    bagging_frac = pull_random_float(0.65, 0.15)
    bagging_freq = pull_random_int(5, 2.0)
    lr = pull_random_float(0.01, 0.025)
    l1 = pull_random_float(0.001, 0.005)
    leaves = pull_random_int(32, 2.0)
    rounds = pull_random_int(1000, 150.0)
    es_rounds = round(rounds / 10)

    return {'objective': 'multiclass',
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
            'min_data_in_leaf': 20,
            'lambda_l1': l1}
