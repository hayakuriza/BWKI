import numpy as np
import os
import shutil

print("fetching labels")
labels_numeric = np.genfromtxt('../labels.csv', delimiter=',', dtype=int)

os.mkdir("../categorys")

for i in range(1, 103):
    os.mkdir("../categorys/" + str(i))

for i in range(1, len(labels_numeric) + 1):
    shutil.copy2(('../images/image_' + str(i).zfill(5) + '.jpg'), '../categorys/' + str(labels_numeric[i - 1]))