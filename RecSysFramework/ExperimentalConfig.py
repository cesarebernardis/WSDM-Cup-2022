import os

from RecSysFramework.DataManager.Splitter import LeaveKOut

from RecSysFramework.Recommender.NonPersonalized import TopPop, GlobalEffects
from RecSysFramework.Recommender.KNN import ItemKNNCF, UserKNNCF, EASE_R, HOEASE_R
from RecSysFramework.Recommender.GraphBased import RP3beta, P3alpha
from RecSysFramework.Recommender.MatrixFactorization import IALS, PureSVD
from RecSysFramework.Recommender.MatrixFactorization import WARP_LightFM as WARP
from RecSysFramework.Recommender.MatrixFactorization import KOSWARP_LightFM as KOSWARP
from RecSysFramework.Recommender.SLIM.ElasticNet import MTSLIM as SLIM
from RecSysFramework.Recommender.DeepLearning import MultVAE, RecVAE


EXPERIMENTAL_CONFIG = {
    'n_folds': 0,
    'splitter': LeaveKOut(k_value=2, with_validation=False, test_rating_threshold=3., allow_cold_users=True),
    'dataset_folder': "datasets" + os.sep,
    'datasets': [
        't1-t2-iu', 't1', 't2',
        's1-t2-iu', 's1-t1-iu', 's2-t1-iu', 's2-t2-iu', 's3-t1-iu', 's3-t2-iu',
        's2-s3-t1-iu', 's2-s3-t2-iu', 's1-s3-t1-iu', 's1-s3-t2-iu', 's1-s2-t1-iu', 's1-s2-t2-iu',
        's1-t1-t2-iu', 's2-t1-t2-iu', 's3-t1-t2-iu',
        's2-s3-t1-t2-iu', 's1-s3-t1-t2-iu', 's1-s2-t1-t2-iu',
        's1-s2-s3-t1-t2-iu', 's1-s2-s3-t1-iu', 's1-s2-s3-t2-iu'
    ],
    'test-datasets': ['t1', 't2'],
    'baselines': [TopPop, ItemKNNCF, UserKNNCF, P3alpha, RP3beta, PureSVD, EASE_R, IALS, SLIM, HOEASE_R, MultVAE, RecVAE, WARP, KOSWARP],
    'best-baselines': [ItemKNNCF, RP3beta, SLIM, EASE_R, HOEASE_R, RecVAE],
    'recap_metrics': ["HR", "NDCG"],
    'cutoffs': [5, 10, 25],
    'cold_user_threshold': 5,
    'quite_cold_user_threshold': 8,
    'quite_warm_user_threshold': 12,
}


