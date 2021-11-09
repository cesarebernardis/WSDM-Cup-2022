import os
import gzip

walk_dir = "datasets" + os.sep

for root, subdirs, files in os.walk(walk_dir):
    root += os.sep
    for filename in files:
        if filename.endswith(".tsv"):
            if not os.path.isfile(root + filename + '.gz'):
                with open(root + filename, 'rb') as f_in, gzip.open(root + filename + '.gz', 'wb') as f_out:
                    f_out.writelines(f_in)
            os.remove(root + filename)
