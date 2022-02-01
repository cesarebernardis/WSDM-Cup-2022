import os
import gzip
import numpy as np
import pandas as pd
import scipy.sparse as sps

from RecSysFramework.ExperimentalConfig import EXPERIMENTAL_CONFIG


def gzip_file(filename):
    with open(filename, 'rb') as f_in, gzip.open(filename + '.gz', 'wb') as f_out:
        f_out.writelines(f_in)


if __name__ == "__main__":

    source_markets = [
        ["s1"], ["s2"], ["s3"], ["s1", "s2", "s3"],
        ["s1", "s2"], ["s2", "s3"], ["s1", "s3"],
    ]

    target_markets = [
        ["t1"], ["t2"], ["t1", "t2"]
    ]

    for source_market in source_markets:
        for target_market in target_markets:

            datasets_to_merge = sorted(source_market + target_market)

            for data in ["train.tsv.gz", "train_5core.tsv.gz", "valid_qrel.tsv.gz", "valid_run.tsv.gz"]:

                dfs = []
                colnames = ["userid", "itemid", "score"]
                header = 0

                if data == "valid_run.tsv.gz":
                    header = None
                    colnames = colnames[:2]

                for folder in datasets_to_merge:
                    basepath = EXPERIMENTAL_CONFIG['dataset_folder'] + folder + os.sep
                    filename = basepath + data
                    if not os.path.isfile(filename):
                        if os.path.isfile(filename[:-3]):
                            gzip_file(filename[:-3])
                        else:
                            raise Exception("File {} is missing!".format(filename[:-3]))
                    dfs.append(pd.read_csv(filename, sep="\t", index_col=False, header=header, names=colnames))

                df = pd.concat(dfs)
                del dfs

                if data == "valid_run.tsv.gz":
                    df.drop_duplicates(subset=["userid"], keep="last")
                else:
                    df.drop_duplicates(subset=["userid", "itemid"], keep="last")

                new_dir = EXPERIMENTAL_CONFIG['dataset_folder'] + "-".join(sorted(datasets_to_merge)) + "-iu" + os.sep
                os.makedirs(new_dir, exist_ok=True)
                df.to_csv(new_dir + data, sep="\t", header=data != "valid_run.tsv.gz", index=False)
