from imports import *


def load_trial_mat(data_dir_behaviour, subject):
    mat_1 = scipy.io.loadmat(
        data_dir_behaviour
        + "/P"
        + subject
        + "_behavioural/P"
        + subject
        + "_Session01_Touch_Processing_EEG.mat"
    )
    mat_2 = scipy.io.loadmat(
        data_dir_behaviour
        + "/P"
        + subject
        + "_behavioural/P"
        + subject
        + "_Session02_Touch_Processing_EEG.mat"
    )
    mat_3 = scipy.io.loadmat(
        data_dir_behaviour
        + "/P"
        + subject
        + "_behavioural/P"
        + subject
        + "_Session03_Touch_Processing_EEG.mat"
    )

    trial_mat_1 = mat_1["trial_mat"]
    trial_mat_2 = mat_2["trial_mat"]
    trial_mat_3 = mat_3["trial_mat"]

    trial_mat_1 = np.transpose(trial_mat_1, (0, 2, 1))
    trial_mat_2 = np.transpose(trial_mat_2, (0, 2, 1))
    trial_mat_3 = np.transpose(trial_mat_3, (0, 2, 1))

    trial_mat_1 = np.reshape(trial_mat_1, (-1, 10), order="F")
    trial_mat_2 = np.reshape(trial_mat_2, (-1, 10), order="F")
    trial_mat_3 = np.reshape(trial_mat_3, (-1, 10), order="F")

    trial_mat_1 = trial_mat_1[(trial_mat_1 != 0).any(axis=1), :]
    trial_mat_2 = trial_mat_2[(trial_mat_2 != 0).any(axis=1), :]
    trial_mat_3 = trial_mat_3[(trial_mat_3 != 0).any(axis=1), :]

    trial_mat_1 = trial_mat_1[~np.isnan(trial_mat_1).any(axis=1), :]
    trial_mat_2 = trial_mat_2[~np.isnan(trial_mat_2).any(axis=1), :]
    trial_mat_3 = trial_mat_3[~np.isnan(trial_mat_3).any(axis=1), :]

    trial_mat = np.concatenate((trial_mat_1, trial_mat_2, trial_mat_3))

    return trial_mat


def load_raw(data_dir_eeg, subject):
    raw_1 = mne.io.read_raw_bdf(
        data_dir_eeg + "/P" + subject + "_eeg/P" + subject + "_S01.bdf", preload=True
    )
    raw_2 = mne.io.read_raw_bdf(
        data_dir_eeg + "/P" + subject + "_eeg/P" + subject + "_S02.bdf", preload=True
    )
    raw_3 = mne.io.read_raw_bdf(
        data_dir_eeg + "/P" + subject + "_eeg/P" + subject + "_S03.bdf", preload=True
    )

    # TODO: sort out other preprocessing steps

    # Low pass filter
    current_sfreq = raw_1.info["sfreq"]
    desired_sfreq = 90
    decim = np.round(current_sfreq / desired_sfreq).astype(int)
    obtained_sfreq = current_sfreq / decim
    lowpass_freq = obtained_sfreq / 3.0

    raw_1 = raw_1.copy().filter(l_freq=None, h_freq=lowpass_freq)
    raw_2 = raw_2.copy().filter(l_freq=None, h_freq=lowpass_freq)
    raw_3 = raw_3.copy().filter(l_freq=None, h_freq=lowpass_freq)

    # TODO: See find_events doc for an explanation of the &= operation below.
    # Although also note that it might not be doing anything at all. Probably
    # worth a bit more investigation.
    events_1 = mne.find_events(raw_1, consecutive=False)
    events_1[:, 2] &= 2**16 - 1
    events_2 = mne.find_events(raw_2, consecutive=False)
    events_2[:, 2] &= 2**16 - 1
    events_3 = mne.find_events(raw_3, consecutive=False)
    events_3[:, 2] &= 2**16 - 1

    raw, events = mne.concatenate_raws(
        raws=[raw_1, raw_2, raw_3],
        preload=True,
        events_list=[events_1, events_2, events_3],
    )

    # NOTE: Define channels to drop
    chs_discard = [
        "EXG1",
        "EXG2",
        "EXG3",
        "EXG4",
        "EXG5",
        "EXG6",
        "EXG7",
        "EXG8",
        "GSR1",
        "GSR2",
        "Erg1",
        "Erg2",
        "Resp",
        "Plet",
        "Temp",
        "Status",
    ]
    raw.drop_channels(chs_discard)

    # NOTE: Define channel names so that we can rename them with proper biosemi64
    # names and load the corresponding layout and montage.
    chs = [
        "A1",
        "A2",
        "A3",
        "A4",
        "A5",
        "A6",
        "A7",
        "A8",
        "A9",
        "A10",
        "A11",
        "A12",
        "A13",
        "A14",
        "A15",
        "A16",
        "A17",
        "A18",
        "A19",
        "A20",
        "A21",
        "A22",
        "A23",
        "A24",
        "A25",
        "A26",
        "A27",
        "A28",
        "A29",
        "A30",
        "A31",
        "A32",
        "B1",
        "B2",
        "B3",
        "B4",
        "B5",
        "B6",
        "B7",
        "B8",
        "B9",
        "B10",
        "B11",
        "B12",
        "B13",
        "B14",
        "B15",
        "B16",
        "B17",
        "B18",
        "B19",
        "B20",
        "B21",
        "B22",
        "B23",
        "B24",
        "B25",
        "B26",
        "B27",
        "B28",
        "B29",
        "B30",
        "B31",
        "B32",
    ]
    chan_idxs = [raw.ch_names.index(ch) for ch in chs]

    # NOTE: Set biosemi64 layout and montage
    biosemi_layout = mne.channels.read_layout("biosemi")
    biosemi_names = biosemi_layout.names
    name_map = dict(zip(chs, biosemi_names))
    raw.rename_channels(name_map)
    montage = make_standard_montage("biosemi64")
    raw.set_montage(montage, on_missing="warn")

    return raw, events


def compute_epochs(raw, events, trial_mat):
    events[events[:, 2] == 1, 2] = 1  # touch start
    events[events[:, 2] == 2, 2] = 1  # touch start
    events[events[:, 2] == 8, 2] = 1  # touch start
    events[events[:, 2] == 4, 2] = 4  # trial start; unchanged.
    events_dict = {"touch": 1, "trial": 4}

    target = trial_mat[:, 5]
    trial_mat = trial_mat[target == 1, :]
    events_touch = events[(events[:, 2] == events_dict["touch"]), :]
    events_touch = events_touch[target == 1]
    events_touch[:, 2] = trial_mat[:, 2]
    visual_ind = trial_mat[:, 0] == 1
    tactile_ind = trial_mat[:, 0] == 2
    events_touch[tactile_ind, 2] += 2
    events_touch_dict = {
        "visual_thumb": 1,
        "visual_pinky": 2,
        "tactile_thumb": 3,
        "tactile_pinky": 4,
    }

    epochs = mne.Epochs(
        raw,
        events_touch,
        event_id=events_touch_dict,
        tmin=-1.1,
        tmax=1.8,
        # decim=decim,
        # baseline=(-0.5, 0.0),
        detrend=None,
        # reject=None,
        # reject=dict(eeg=1e-3),
        # picks=None,
        # picks=plt_chans,
        preload=True,
    )

    return epochs


def load_epochs(data_dir_epochs, subject, trial_mat):
    epochs = mne.io.read_epochs_eeglab(
        data_dir_epochs + "sub-" + subject[-2:] + "_task-touchdecoding_continuous.set"
    )

    # TODO: Clarify mode == visual is and what else is possible / interesting
    # finger | mode == visual
    target = trial_mat[:, 5]
    trial_mat = trial_mat[target == 1, :]
    events_finger = epochs.events[(epochs.events[:, 2] == 1) & (target == 1), :]
    events_finger[:, 2] = trial_mat[:, 2]
    visual_ind = trial_mat[:, 0] == 1
    tactile_ind = trial_mat[:, 0] == 2
    events_finger[tactile_ind, 2] += 2
    events_finger_dict = {
        "visual_thumb": 1,
        "visual_pinky": 2,
        "tactile_thumb": 3,
        "tactile_pinky": 4,
    }
    epochs.events = events_finger

    # resample to reduce compute time
    epochs = epochs.resample(sfreq=90)

    return epochs


def run_time_gens(epochs):
    X = epochs.get_data()
    y = epochs.events[:, -1]

   # TODO: down sample here for test purposes

    n_splits = 12
    metric = "accuracy"

    # within mode: everything on touch
    # touchtouch
    XX = X[(y == 3) | (y == 4)]
    yy = y[(y == 3) | (y == 4)]
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True)
    cv = list(skf.split(XX, yy))
    clf = LinearDiscriminantAnalysis(solver="svd")
    pipe = Pipeline([("scl", Scaler(epochs.info)), ("vec", Vectorizer()), ("clf", clf)])
    time_gen = GeneralizingEstimator(pipe, n_jobs=-1, scoring=metric, verbose=True)
    scores = cross_val_multiscore(time_gen, X, y, cv=cv, n_jobs=-1)
    scores = np.mean(scores, 0)
    scores_touchtouch = scores

    # within mode: everything on vision
    # visvis
    XX = X[(y == 1) | (y == 2)]
    yy = y[(y == 1) | (y == 2)]
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True)
    cv = list(skf.split(XX, yy))
    clf = LinearDiscriminantAnalysis(solver="svd")
    pipe = Pipeline([("scl", Scaler(epochs.info)), ("vec", Vectorizer()), ("clf", clf)])
    time_gen = GeneralizingEstimator(pipe, n_jobs=-1, scoring=metric, verbose=True)
    scores = cross_val_multiscore(time_gen, X, y, cv=cv, n_jobs=-1)
    scores = np.mean(scores, 0)
    scores_visvis = scores

    # cross mode: train on touch / test on vision
    # touchvis
    XX = X
    yy = y
    train_ind = np.where((yy == 3) | (yy == 4))[0]
    test_ind = np.where((yy == 1) | (yy == 2))[0]
    np.random.shuffle(train_ind)
    np.random.shuffle(test_ind)
    train_ind = np.array_split(train_ind, n_splits)
    test_ind = np.array_split(test_ind, n_splits)
    cv = list(zip(train_ind, test_ind))
    pipe = Pipeline([("scl", Scaler(epochs.info)), ("vec", Vectorizer()), ("clf", clf)])
    time_gen = GeneralizingEstimator(pipe, n_jobs=-1, scoring=metric, verbose=True)
    scores = cross_val_multiscore(time_gen, X, y, cv=cv, n_jobs=-1)
    scores = np.mean(scores, 0)
    scores_touchvis = scores

    # cross mode: train on vision / test on touch
    # vistouch
    XX = X
    yy = y
    train_ind = np.where((yy == 1) | (yy == 2))[0]
    test_ind = np.where((yy == 3) | (yy == 4))[0]
    np.random.shuffle(train_ind)
    np.random.shuffle(test_ind)
    train_ind = np.array_split(train_ind, n_splits)
    test_ind = np.array_split(test_ind, n_splits)
    cv = list(zip(train_ind, test_ind))
    clf = LinearDiscriminantAnalysis(solver="svd")
    pipe = Pipeline([("scl", Scaler(epochs.info)), ("vec", Vectorizer()), ("clf", clf)])
    time_gen = GeneralizingEstimator(pipe, n_jobs=-1, scoring=metric, verbose=True)
    scores = cross_val_multiscore(time_gen, X, y, cv=cv, n_jobs=-1)
    scores = np.mean(scores, 0)
    scores_vistouch = scores

    return (scores_touchtouch, scores_visvis, scores_touchvis, scores_vistouch)


def tune_hyper_params(X, y):
    # within mode: everything on touch
    tag = "touchtouch"
    XX = X[(y == 3) | (y == 4)]
    yy = y[(y == 3) | (y == 4)]

    # pick a single time point based on previous time gen results
    # TODO: deal with hard code hack
    XX = XX[:, :, 106]

    # Split the dataset in two equal parts. One part to be used to generate
    # train / test splits for each grid, point, and the other part to be used
    # for overall validation.
    X_train, X_test, y_train, y_test = train_test_split(
        XX, yy, test_size=0.5, random_state=0
    )

    train_ind_grid = np.where((y_train == 3) | (y_train == 4))[0]
    test_ind_grid = np.where((y_train == 3) | (y_train == 4))[0]

    train_ind_grid = np.array_split(train_ind_grid, n_splits)
    test_ind_grid = np.array_split(test_ind_grid, n_splits)

    np.random.shuffle(train_ind_grid)
    np.random.shuffle(test_ind_grid)

    cv_grid = list(zip(train_ind_grid, test_ind_grid))

    pipe = Pipeline(
        [("scl", Scaler(epochs.info)), ("vec", Vectorizer()), ("svc", SVC())]
    )

    param_grid = [
        {"svc__C": [0.1, 1, 10, 100, 1000], "svc__kernel": ["linear"]},
        {
            "svc__C": [0.1, 1, 10, 100, 1000],
            "svc__gamma": [0.001, 0.0001],
            "svc__kernel": ["rbf"],
        },
    ]

    clf = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring=metric,
        cv=cv_grid,
        verbose=1,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    # TODO: sort out how to use X_test to select best params

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_["mean_test_score"]
    stds = clf.cv_results_["std_test_score"]
    for mean, std, params in zip(means, stds, clf.cv_results_["params"]):
        print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))
    print()

    # results_df = pd.DataFrame(clf.cv_results_)
    # results_df = results_df.sort_values(by=['rank_test_score'])
    # results_df = (results_df.set_index(
    #     results_df["params"].apply(lambda x: "_".join(
    #         str(val) for val in x.values()))).rename_axis('kernel'))
    # print(results_df[[
    #     'params', 'rank_test_score', 'mean_test_score', 'std_test_score'
    # ]])
