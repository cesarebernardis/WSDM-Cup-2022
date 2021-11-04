import numpy as np

from RecSysFramework.ExperimentalConfig import EXPERIMENTAL_CONFIG


for splitter in EXPERIMENTAL_CONFIG['splits']:

    for dataset_config in EXPERIMENTAL_CONFIG['datasets']:

        datareader = dataset_config['datareader']()
        postprocessings = dataset_config['postprocessings']

        dataset = datareader.load_data(postprocessings=postprocessings)
        dataset.save_data()

        train, test, validation = splitter.split(dataset, random_seed=42)
        splitter.save_split([train, test, validation])
