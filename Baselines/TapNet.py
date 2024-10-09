# TapNet is from the AEON library: 
!pip install -U aeon
!pip install keras-self-attention

import numpy as np
import pandas as pd
from aeon.classification.deep_learning import TapNetClassifier
from aeon.datasets import load_unit_test
from sklearn.metrics import accuracy_score

# results are saved to a pandas dataframe and stored in a google drive as a CSV.
# please change the directory/drive locations and filenames to ones that work for you.

# method for saving a file in a dataset directory for individual laps results.
# OPTIONAL
def save_to_file_directory(data_dir, dataset_name, shot_dir, normalize_data, acc):
  path = data_dir + dataset_name + shot_dir
  with open(path+'results_tapnet.txt', 'w') as f:
    f.write(dataset_name + '\n')
    f.write(shot_dir + '\n')
    f.write(str(normalize_data) + '\n')
    f.write(str(acc) + '\n')
    f.write(str(acc.mean()))

