import os
import shutil
import zipfile

sub_filename = "submission" + os.sep + "submission.zip"

os.path.makedirs("submission" + os.sep + "t1", exist_ok=True)
os.path.makedirs("submission" + os.sep + "t2", exist_ok=True)

if os.path.exists(sub_filename):
    os.remove(sub_filename)

for folder in ["t1", "t2"]:
    for target in ["valid", "test"]:
        shutil.copyfile(os.sep.join(["datasets", folder, target + "_scores.tsv"]),
                        os.sep.join(["submission", folder, target + "_pred.tsv"]))

with zipfile.ZipFile(sub_filename, mode='a') as zipfile:
    for folder in ["t1", "t2"]:
        for target in ["valid", "test"]:
            zipfile.write(os.sep.join(["submission", folder, target + "_pred.tsv"]))
