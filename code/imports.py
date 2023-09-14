import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

import multiprocessing

import scipy.io
from scipy import signal
from scipy.stats import ttest_1samp

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    GridSearchCV,
    ShuffleSplit,
    cross_val_score,
    StratifiedKFold,
)
from sklearn.preprocessing import LabelEncoder

import mne
from mne import Epochs, create_info, pick_types, events_from_annotations
from mne.io import concatenate_raws, read_raw_edf
from mne.channels import make_standard_montage
from mne.decoding import (
    GeneralizingEstimator,
    SlidingEstimator,
    Scaler,
    cross_val_multiscore,
    Vectorizer,
    CSP,
)
from mne.time_frequency import AverageTFR, tfr_morlet

# from autoreject import AutoReject
# from autoreject import get_rejection_threshold
# from autoreject import Ransac

# import pymc3 as pm
