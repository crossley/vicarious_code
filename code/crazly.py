import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.io
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import mne
from mne import Epochs, pick_types, events_from_annotations
from mne.channels import make_standard_montage
from mne.io import concatenate_raws, read_raw_edf
from mne.datasets import eegbci
from mne.decoding import CSP
from mne.decoding import (SlidingEstimator, GeneralizingEstimator, Scaler,
                          cross_val_multiscore, LinearModel, get_coef,
                          Vectorizer, CSP)

mpl.use('TkAgg')

raw_data_dir = '/Volumes/LaCie/Sophie/EEG_Analysis_new/raw_data/'

sub_dirs = np.sort([x[0] for x in os.walk(raw_data_dir)])
sub_dirs = sub_dirs[2:]

scores_rec = []
for s in sub_dirs:

    print(s[-3:])

    mat_1 = scipy.io.loadmat(s + '/' + s[-3:] +
                             '_Session01_Touch_Processing_EEG.mat')
    mat_2 = scipy.io.loadmat(s + '/' + s[-3:] +
                             '_Session02_Touch_Processing_EEG.mat')
    mat_3 = scipy.io.loadmat(s + '/' + s[-3:] +
                             '_Session03_Touch_Processing_EEG.mat')

    raw_1 = mne.io.read_raw_bdf(s + '/' + s[-3:] + '_S01.bdf', preload=True)
    raw_2 = mne.io.read_raw_bdf(s + '/' + s[-3:] + '_S02.bdf', preload=True)
    raw_3 = mne.io.read_raw_bdf(s + '/' + s[-3:] + '_S03.bdf', preload=True)

    trial_mat_1 = mat_1['trial_mat']
    trial_mat_2 = mat_2['trial_mat']
    trial_mat_3 = mat_3['trial_mat']
    trial_mat_1 = np.transpose(trial_mat_1, (0, 2, 1))
    trial_mat_2 = np.transpose(trial_mat_2, (0, 2, 1))
    trial_mat_3 = np.transpose(trial_mat_3, (0, 2, 1))
    trial_mat_1 = np.reshape(trial_mat_1, (-1, 10), order='F')
    trial_mat_2 = np.reshape(trial_mat_2, (-1, 10), order='F')
    trial_mat_3 = np.reshape(trial_mat_3, (-1, 10), order='F')
    trial_mat_1 = trial_mat_1[(trial_mat_1 != 0).any(axis=1), :]
    trial_mat_2 = trial_mat_2[(trial_mat_2 != 0).any(axis=1), :]
    trial_mat_3 = trial_mat_3[(trial_mat_3 != 0).any(axis=1), :]
    trial_mat_1 = trial_mat_1[~np.isnan(trial_mat_1).any(axis=1), :]
    trial_mat_2 = trial_mat_2[~np.isnan(trial_mat_2).any(axis=1), :]
    trial_mat_3 = trial_mat_3[~np.isnan(trial_mat_3).any(axis=1), :]
    trial_mat = np.concatenate((trial_mat_1, trial_mat_2, trial_mat_3))

    # NOTE: Low pass filter in prep for decimation. This comes into play in the
    # 'decim' parameter when creating epochs.
    current_sfreq = raw_1.info['sfreq']
    desired_sfreq = 90
    decim = np.round(current_sfreq / desired_sfreq).astype(int)
    obtained_sfreq = current_sfreq / decim
    lowpass_freq = obtained_sfreq / 3.

    raw_1 = raw_1.copy().filter(l_freq=None, h_freq=lowpass_freq)
    raw_2 = raw_2.copy().filter(l_freq=None, h_freq=lowpass_freq)
    raw_3 = raw_3.copy().filter(l_freq=None, h_freq=lowpass_freq)

    # NOTE: See find_events doc for an explanation of the &= operation below
    # NOTE: Although also note that it might not be doing anything at all. Probably
    # worth a bit more investigation.
    events_1 = mne.find_events(raw_1, consecutive=False)
    events_1[:, 2] &= (2**16 - 1)
    events_2 = mne.find_events(raw_2, consecutive=False)
    events_2[:, 2] &= (2**16 - 1)
    events_3 = mne.find_events(raw_3, consecutive=False)
    events_3[:, 2] &= (2**16 - 1)

    # NOTE: concatenate raw and events.
    raw, events = mne.concatenate_raws(
        raws=[raw_1, raw_2, raw_3],
        preload=True,
        events_list=[events_1, events_2, events_3])

    # NOTE: Interacting with this plot is probably the best way to remove bad
    # channels and perhaps do other quality control.
    # raw.plot(events=events,
    #          event_id=events_dict,
    #          order=chan_idxs,
    #          duration=100,
    #          clipping=None,
    #          decim=2,
    #          scalings={'eeg': 5e-4})

    # NOTE: Define channels to drop
    chs_discard = [
        'EXG1', 'EXG2', 'EXG3', 'EXG4', 'EXG5', 'EXG6', 'EXG7', 'EXG8', 'GSR1',
        'GSR2', 'Erg1', 'Erg2', 'Resp', 'Plet', 'Temp', 'Status'
    ]
    raw.drop_channels(chs_discard)

    # NOTE: Define channel names so that we can rename them with proper biosemi64
    # names and load the corresponding layout and montage.
    chs = [
        'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A11',
        'A12', 'A13', 'A14', 'A15', 'A16', 'A17', 'A18', 'A19', 'A20', 'A21',
        'A22', 'A23', 'A24', 'A25', 'A26', 'A27', 'A28', 'A29', 'A30', 'A31',
        'A32', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9', 'B10',
        'B11', 'B12', 'B13', 'B14', 'B15', 'B16', 'B17', 'B18', 'B19', 'B20',
        'B21', 'B22', 'B23', 'B24', 'B25', 'B26', 'B27', 'B28', 'B29', 'B30',
        'B31', 'B32'
    ]
    chan_idxs = [raw.ch_names.index(ch) for ch in chs]

    # NOTE: Set biosemi64 layout and montage
    biosemi_layout = mne.channels.read_layout('biosemi')
    biosemi_names = biosemi_layout.names
    name_map = dict(zip(chs, biosemi_names))
    raw.rename_channels(name_map)
    montage = make_standard_montage('biosemi64')
    raw.set_montage(montage, on_missing='warn')

    # NOTE: re-code various touch ids
    events[events[:, 2] == 1, 2] = 1  # touch start
    events[events[:, 2] == 2, 2] = 1  # touch start
    events[events[:, 2] == 8, 2] = 1  # touch start
    events[events[:, 2] == 4, 2] = 4  # trial start; unchanged.
    events_dict = {'touch': 1, 'trial': 4}

    # NOTE: mode
    events_mode = events[events[:, 2] == 1, :]
    events_mode[:, 2] = trial_mat[:, 0]
    events_mode_dict = {'visual': 1, 'tactile': 2}

    # NOTE: movement
    events_movement = events[events[:, 2] == 1, :]
    events_movement[:, 2] = trial_mat[:, 2]
    visual_ind = trial_mat[:, 0] == 1
    events_movement = events_movement[visual_ind, :]
    events_movement_dict = {'left': 1, 'right': 2}

    # NOTE: finger | mode == visual
    events_finger = events[events[:, 2] == 1, :]
    events_finger[:, 2] = trial_mat[:, 2]
    visual_ind = trial_mat[:, 0] == 1
    tactile_ind = trial_mat[:, 0] == 2
    events_finger[tactile_ind, 2] += 2
    events_finger_dict = {
        'visual_thumb': 1,
        'visual_pinky': 2,
        'tactile_thumb': 3,
        'tactile_pinky': 4
    }
    # events_finger = events_finger[visual_ind, :]
    # events_finger_dict = {'thumb': 1, 'pinky': 2}

    # NOTE: Plot epochs raster and erps (define and select events below with
    # comment / uncomment)
    plt_chans = chan_idxs
    # plt_chans = ['C5', 'C6']
    # plt_chans = ['C5']
    # plt_chans = ['C6']
    # plt_chans = ['Oz']
    # ev = events_mode
    # ev_dict = events_mode_dict
    # ev = events_movement
    # ev_dict = events_movement_dict
    ev = events_finger
    ev_dict = events_finger_dict
    epochs = mne.Epochs(
        raw,
        ev,
        event_id=ev_dict,
        tmin=-1.1,
        tmax=1.8,
        decim=decim,
        # baseline=(-0.5, 0.0),
        detrend=None,
        # reject=None,
        reject=dict(eeg=1e-3),
        # picks=None,
        picks=plt_chans,
        preload=True)

    # fig, ax = plt.subplots(2, len(ev_dict.keys()))
    # for i, k in enumerate(ev_dict.keys()):
    #     epochs[k].plot_image(picks=plt_chans,
    #                          combine='mean',
    #                          axes=ax[:, i],
    #                          show=False,
    #                          colorbar=False)
    #     ax[1, i].set_title(k)
    # plt.tight_layout()
    # plt.show()

    # fig, ax = plt.subplots(5, len(ev_dict.keys()))
    # for i, k in enumerate(ev_dict.keys()):
    #     epochs[k].plot_psd_topomap(axes=ax[:, i], show=False)
    # plt.tight_layout()
    # plt.show()

    # evoked = epochs.average()
    # times = np.arange(-1.09, 1.8, 0.2)
    # evoked.plot_topomap(times, nrows=3)

    ###########################
    # NOTE: classifier analyses
    ###########################
    events_mode = events[events[:, 2] == 1, :]
    events_mode[:, 2] = trial_mat[:, 0]
    events_mode_dict = {'visual': 1, 'tactile': 2}

    # NOTE: movement
    events_movement = events[events[:, 2] == 1, :]
    events_movement[:, 2] = trial_mat[:, 2]
    visual_ind = trial_mat[:, 0] == 1
    events_movement = events_movement[visual_ind, :]
    events_movement_dict = {'left': 1, 'right': 2}

    # NOTE: finger | mode == visual
    events_finger = events[events[:, 2] == 1, :]
    events_finger[:, 2] = trial_mat[:, 2]
    visual_ind = trial_mat[:, 0] == 1
    tactile_ind = trial_mat[:, 0] == 2
    events_finger = events_finger[visual_ind, :]
    # events_finger = events_finger[tactile_ind, :]
    events_finger_dict = {'thumb': 1, 'pinky': 2}

    # TODO: Having ground truth really helped anchor Andrea's analyses. Can
    # decoding thumb vs pinky on tactile mode trials serve that purpose here?
    # NOTE: Ground truth: touch side on visual trials
    # NOTE: Ground truth: pinky vs thumb on tactile trials
    # NOTE: cross generalise

    # ev = events_mode
    # ev_dict = events_mode_dict
    # ev = events_movement
    # ev_dict = events_movement_dict
    ev = events_finger
    ev_dict = events_finger_dict

    epochs = mne.Epochs(
        raw,
        ev,
        event_id=ev_dict,
        tmin=-1.1,
        tmax=1.8,
        decim=decim,
        baseline=(-1.1, -1.0),
        # baseline=None,
        # detrend=None,
        detrend=1,
        # reject=None,
        reject=dict(eeg=1e-3),
        # picks=None,
        preload=True)

    # # NOTE: CSP classification
    # # Define a monte-carlo cross-validation generator (reduce variance):
    # scores = []
    # epochs_data = epochs.get_data()
    # labels = epochs.events[:, -1]
    # cv = ShuffleSplit(10, test_size=0.2, random_state=42)
    # cv_split = cv.split(epochs_data)

    # # signal decomposition using the Common Spatial Patterns (CSP)
    # csp = CSP(n_components=2, reg=None, log=True, norm_trace=False)

    # # define the classifier: Linear discriminant analysis
    # lda = LinearDiscriminantAnalysis()

    # # Use scikit-learn Pipeline with cross_val_score function
    # clf = Pipeline([('CSP', csp), ('LDA', lda)])
    # scores = cross_val_score(clf, epochs_data, labels, cv=cv, n_jobs=1)

    # # Print the results
    # class_balance = np.mean(labels == labels[0])
    # class_balance = max(class_balance, 1. - class_balance)
    # print("Classification accuracy: %f / Chance level: %f" %
    #       (np.mean(scores), class_balance))

    # NOTE: temporal decoding
    X = epochs.get_data()
    y = epochs.events[:, -1]

    # clf = Pipeline([('CSP', csp), ('SVC', svc)])

    # clf = make_pipeline(StandardScaler(), LogisticRegression(solver='lbfgs', max_iter=500))
    clf = make_pipeline(StandardScaler(), SVC(C=1, kernel='linear'))
    time_decod = SlidingEstimator(clf,
                                  n_jobs=1,
                                  scoring='roc_auc',
                                  verbose=True)
    scores = cross_val_multiscore(time_decod, X, y, cv=5, n_jobs=1)

    # Mean scores across cross-validation splits
    scores = np.mean(scores, axis=0)
    scores_rec.append(scores)
    np.savetxt('scores' + s[-3:] + '.txt', scores)

scores = np.vstack(scores_rec)
scores_mean = np.mean(scores, 0)

np.savetxt('scores_mean.txt', scores_mean)

# Plot
fig, ax = plt.subplots()
ax.plot(epochs.times, scores_mean, label='score')
ax.axhline(.5, color='k', linestyle='--', label='chance')
ax.set_xlabel('Times')
ax.set_ylabel('AUC')  # Area Under the Curve
ax.legend()
ax.axvline(.0, color='k', linestyle='-')
ax.set_title('Sensor space decoding')
plt.show()

# clf = make_pipeline(StandardScaler(),
#                     LinearModel(LogisticRegression(solver='lbfgs', max_iter=500)))
# time_decod = SlidingEstimator(clf, n_jobs=1, scoring='roc_auc', verbose=True)
# time_decod.fit(X, y)

# coef = get_coef(time_decod, 'patterns_', inverse_transform=True)
# evoked_time_gen = mne.EvokedArray(coef, epochs.info, tmin=epochs.times[0])
# joint_kwargs = dict(ts_args=dict(time_unit='s'),
#                     topomap_args=dict(time_unit='s'))
# evoked_time_gen.plot_joint(times=np.arange(0., .500, .100), title='patterns',
#                            **joint_kwargs)
