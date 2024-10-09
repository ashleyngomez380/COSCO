# TapNet is from the AEON library: 
!pip install -U aeon
!pip install keras-self-attention

import numpy as np
import pandas as pd
from aeon.classification.deep_learning import TapNetClassifier
from aeon.datasets import load_unit_test
from sklearn.metrics import accuracy_score

