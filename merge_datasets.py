import os
import gzip
import numpy as np
import pandas as pd
import scipy.sparse as sps

from RecSysFramework.ExperimentalConfig import EXPERIMENTAL_CONFIG

all_datasets_to_merge = [
    ["s2", "s3", "t1"],
    ["s2", "s3", "t2"],
]


for datasets_to_merge in all_datasets_to_merge:
    for data in ["train.tsv.gz", "train_5core.tsv.gz", "valid_qrel.tsv.gz", "valid_run.tsv.gz"]:

        dfs = []
        colnames = ["userid", "itemid", "score"]
        header = 0

        if data == "valid_run.tsv.gz":
            header = None
            colnames = colnames[:2]

        for folder in datasets_to_merge:
            basepath = EXPERIMENTAL_CONFIG['dataset_folder'] + folder + os.sep
            dfs.append(pd.read_csv(basepath + data, sep="\t", index_col=False, header=header, names=colnames))

        df = pd.concat(dfs)
        del dfs
        #df["userid"] = df["userid"].apply(lambda x: x[2:])
        if data == "valid_run.tsv.gz":
            df.drop_duplicates(subset=["userid"], keep="last")
        else:
            df.drop_duplicates(subset=["userid", "itemid"], keep="last")
        #df["userid"] = df["userid"].apply(lambda x: "xx" + x)

        new_dir = EXPERIMENTAL_CONFIG['dataset_folder'] + "-".join(sorted(datasets_to_merge)) + "-iu" + os.sep
        os.makedirs(new_dir, exist_ok=True)
        df.to_csv(new_dir + data, sep="\t", header=data != "valid_run.tsv.gz", index=False)
