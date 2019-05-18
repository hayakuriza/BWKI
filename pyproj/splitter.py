import numpy as np
import os
import shutil
import time

rename_folders=True


print("fetching labels")
labels_numeric = np.genfromtxt('../labels.csv', delimiter=',', dtype=int)

os.mkdir("../categorys")

for i in range(1, 103):
    os.mkdir("../categorys/" + str(i))

print("copying images to folders")
for i in range(1, len(labels_numeric) + 1):
    shutil.copy2(('../images/image_' + str(i).zfill(5) + '.jpg'), '../categorys/' + str(labels_numeric[i - 1]))



if rename_folders:
    print("renaming folders")
    with open('../Oxford-102_Flower_dataset_labels.txt') as f:
        label_names = [label[2:-1] for label in f.read().split("\n")]

    label_names = [w.replace(' ', '_') for w in label_names]
    label_names = [w.replace('?', '') for w in label_names]

    for i in range(1, 103):
        for retry in range(100):
            try:
                os.rename(('../categorys/' + str(i)), ('../categorys/' + label_names[i-1]))
                break
            except:
                print("rename failed, retrying in 1s...")
                time.sleep(1)


