from imports import *


def create_sub_nums_sophie():
    dir_data_behaviour = "/Users/mq20185996/projects/crazly/raw_data"
    dir_data_eeg = "/Users/mq20185996/projects/crazly/raw_data/"
    dir_figs = "../figures/sophie/"
    dir_results = "../results/sophie/"

    sub_dirs_eeg = np.sort(os.listdir(dir_data_eeg))
    sub_dirs_eeg = sub_dirs_eeg[sub_dirs_eeg != ".DS_Store"]
    sub_nums_eeg = [x[1:3] for x in sub_dirs_eeg]

    sub_dirs_behaviour = np.sort(os.listdir(dir_data_behaviour))
    sub_dirs_behaviour = sub_dirs_behaviour[sub_dirs_behaviour != ".DS_Store"]
    sub_nums_behaviour = [x[1:3] for x in sub_dirs_behaviour]

    sub_nums = np.intersect1d(sub_nums_eeg, sub_nums_behaviour)

    exclude_subs = ["01", "10", "13", "16", "18", "32", "33"]
    sub_nums = [x for x in sub_nums if x not in exclude_subs]

    subs_non_mts = [1, 2, 3, 4, 5, 6, 7, 10, 15, 17, 19, 22, 25, 26, 27, 34, 37]
    subs_non_mts = ["{0:0=2d}".format(x) for x in subs_non_mts]

    subs_mts = [
        8,
        9,
        12,
        13,
        14,
        16,
        18,
        20,
        21,
        23,
        24,
        28,
        29,
        30,
        31,
        32,
        33,
        35,
        36,
        38,
        39,
        40,
    ]
    subs_mts = ["{0:0=2d}".format(x) for x in subs_mts]

    sub_nums = subs_mts

    return sub_nums, dir_data_behaviour, dir_data_eeg, dir_figs, dir_results


def create_sub_nums_elen():
    dir_data_behaviour = "/Users/mq20185996/projects/crazly/elen/data_behaviour"
    dir_data_eeg = "/Users/mq20185996/projects/crazly/elen/data_eeg"
    dir_figs = "../figures/elen/"
    dir_results = "../results/elen/"

    sub_dirs_eeg = np.sort(os.listdir(dir_data_eeg))
    sub_dirs_eeg = sub_dirs_eeg[sub_dirs_eeg != ".DS_Store"]
    sub_nums_eeg = [x[1:3] for x in sub_dirs_eeg]

    sub_dirs_behaviour = np.sort(os.listdir(dir_data_behaviour))
    sub_dirs_behaviour = sub_dirs_behaviour[sub_dirs_behaviour != ".DS_Store"]
    sub_nums_behaviour = [x[1:3] for x in sub_dirs_behaviour]

    sub_nums = np.intersect1d(sub_nums_eeg, sub_nums_behaviour)
    # sub_nums = np.delete(sub_nums, [1, 17])  # TODO fix weird subject

    return sub_nums, dir_data_behaviour, dir_data_eeg, dir_figs, dir_results


def load_trial_mat(data_dir_behaviour, subject):
    mat_1 = scipy.io.loadmat(
        data_dir_behaviour
        + "/P"
        + subject
        + "/P"
        + subject
        + "_Session01_Touch_Processing_EEG.mat"
    )
    mat_2 = scipy.io.loadmat(
        data_dir_behaviour
        + "/P"
        + subject
        + "/P"
        + subject
        + "_Session02_Touch_Processing_EEG.mat"
    )
    mat_3 = scipy.io.loadmat(
        data_dir_behaviour
        + "/P"
        + subject
        + "/P"
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


def manually_inspect_raws(res, dir_data_eeg_cleaned, s):
    sub_nums = res[0]
    dir_data_behaviour = res[1]
    dir_data_eeg = res[2]
    dir_figs = res[3]
    dir_results = res[4]

    trial_mat = load_trial_mat(dir_data_behaviour, s)
    raw, events = load_raw(dir_data_eeg, s)
    for i in range(len(raw)):
        raw[i].plot(events=events[i], show=True, block=True)
        raw[i].save(
            dir_data_eeg_cleaned + str(s) + "_" + str(i) + "-raw.fif",
            overwrite=True,
        )


def load_raw(data_dir_eeg, subject, dir_data_eeg_cleaned):
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

    if (subject != "02") and (subject != "18"):
        raw_1 = mne.io.read_raw_bdf(
            data_dir_eeg + "/P" + subject + "/P" + subject + "_S01.bdf", preload=True
        )
        raw_2 = mne.io.read_raw_bdf(
            data_dir_eeg + "/P" + subject + "/P" + subject + "_S02.bdf", preload=True
        )
        raw_3 = mne.io.read_raw_bdf(
            data_dir_eeg + "/P" + subject + "/P" + subject + "_S03.bdf", preload=True
        )
        raw_container = (raw_1, raw_2, raw_3)

    else:
        raw_1 = mne.io.read_raw_bdf(
            data_dir_eeg + "/P" + subject + "/P" + subject + "_S01.bdf", preload=True
        )
        raw_2 = mne.io.read_raw_bdf(
            data_dir_eeg + "/P" + subject + "/P" + subject + "_S02-S03.bdf",
            preload=True,
        )
        raw_container = (raw_1, raw_2)

    raw_list = []
    events_list = []
    for i, raw in enumerate(raw_container):
        # TODO: Look into &= the operation below.
        events = mne.find_events(raw, consecutive=False)
        events[:, 2] &= 2**16 - 1

        chan_idxs = [raw.ch_names.index(ch) for ch in chs]

        biosemi_layout = mne.channels.read_layout("biosemi")
        biosemi_names = biosemi_layout.names
        name_map = dict(zip(chs, biosemi_names))
        montage = make_standard_montage("biosemi64")

        raw.drop_channels(chs_discard)
        raw.rename_channels(name_map)
        raw.set_montage(montage, on_missing="warn")

        raw_list.append(raw)
        events_list.append(events)

        mne.write_events(
            dir_data_eeg_cleaned + subject + "_events_" + str(i) + ".fif",
            events,
            overwrite=True,
        )

    return raw_list, events_list


def interpolate_bad_raws(res, dir_data_eeg_cleaned, s):
    sub_nums = res[0]
    dir_data_behaviour = res[1]
    dir_data_eeg = res[2]
    dir_figs = res[3]
    dir_results = res[4]

    # TODO: refacor this to be work well with arbitrary number of sessions. Perhaps
    # start by altering subject directory stuff.
    if (s != "02") and (s != "18"):
        raw_0 = mne.io.read_raw(dir_data_eeg_cleaned + s + "_0-raw.fif", preload=True)
        raw_1 = mne.io.read_raw(dir_data_eeg_cleaned + s + "_1-raw.fif", preload=True)
        raw_2 = mne.io.read_raw(dir_data_eeg_cleaned + s + "_2-raw.fif", preload=True)

        raw_bads = raw_0.info["bads"] + raw_1.info["bads"] + raw_2.info["bads"]
        raw_0.info["bads"] = raw_bads
        raw_1.info["bads"] = raw_bads
        raw_2.info["bads"] = raw_bads

        raw_0_interp = raw_0.copy().interpolate_bads(reset_bads=True)
        raw_1_interp = raw_1.copy().interpolate_bads(reset_bads=True)
        raw_2_interp = raw_2.copy().interpolate_bads(reset_bads=True)

        raw = [raw_0_interp, raw_1_interp, raw_2_interp]

        events_0 = mne.read_events(
            dir_data_eeg_cleaned + s + "_events_" + str(0) + ".fif"
        )
        events_1 = mne.read_events(
            dir_data_eeg_cleaned + s + "_events_" + str(1) + ".fif"
        )
        events_2 = mne.read_events(
            dir_data_eeg_cleaned + s + "_events_" + str(2) + ".fif"
        )

        events = [events_0, events_1, events_2]

    else:
        raw_0 = mne.io.read_raw(dir_data_eeg_cleaned + s + "_0-raw.fif", preload=True)
        raw_1 = mne.io.read_raw(dir_data_eeg_cleaned + s + "_1-raw.fif", preload=True)

        raw_bads = raw_0.info["bads"] + raw_1.info["bads"]
        raw_0.info["bads"] = raw_bads
        raw_1.info["bads"] = raw_bads

        raw_0_interp = raw_0.copy().interpolate_bads(reset_bads=True)
        raw_1_interp = raw_1.copy().interpolate_bads(reset_bads=True)

        raw = [raw_0_interp, raw_1_interp]

        events_0 = mne.read_events(
            dir_data_eeg_cleaned + s + "_events_" + str(0) + ".fif"
        )
        events_1 = mne.read_events(
            dir_data_eeg_cleaned + s + "_events_" + str(1) + ".fif"
        )

        events = [events_0, events_1]

    return raw, events


def filter_raw(raw, dir_data_eeg_cleaned, s):
    rf_list = []
    for i, r in enumerate(raw):
        current_sfreq = r.info["sfreq"]
        desired_sfreq = 90
        decim = np.round(current_sfreq / desired_sfreq).astype(int)
        obtained_sfreq = current_sfreq / decim
        lowpass_freq = obtained_sfreq / 3.0

        rf = r.filter(l_freq=None, h_freq=lowpass_freq)
        rf_list.append(rf)

        r.save(
            dir_data_eeg_cleaned + str(s) + "_" + str(i) + "-raw_filt.fif",
            overwrite=True,
        )

    raw = rf_list

    return raw, decim


def load_filt(dir_data_eeg_cleaned, s):
    if (s != "02") and (s != "18"):
        raw_filt_0 = mne.io.read_raw(
            dir_data_eeg_cleaned + s + "_" + str(0) + "-raw_filt.fif", preload=True
        )

        raw_filt_1 = mne.io.read_raw(
            dir_data_eeg_cleaned + s + "_" + str(1) + "-raw_filt.fif", preload=True
        )

        raw_filt_2 = mne.io.read_raw(
            dir_data_eeg_cleaned + s + "_" + str(2) + "-raw_filt.fif", preload=True
        )

        raw_filt = [raw_filt_0, raw_filt_1, raw_filt_2]

    else:
        raw_filt_0 = mne.io.read_raw(
            dir_data_eeg_cleaned + s + "_" + str(0) + "-raw_filt.fif", preload=True
        )

        raw_filt_1 = mne.io.read_raw(
            dir_data_eeg_cleaned + s + "_" + str(1) + "-raw_filt.fif", preload=True
        )

        raw_filt = [raw_filt_0, raw_filt_1]

    return raw_filt


def compute_epochs(raw, events, trial_mat, decim, dir_data_eeg_cleaned, s):
    raw, events = mne.concatenate_raws(
        raws=raw,
        events_list=events,
        preload=True,
    )

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
        decim=decim,
        # baseline=(-1.1, -1.0),
        # detrend=1,
        # reject=dict(eeg=1e-4),
        # picks=None,
        preload=True,
    )

    epochs = epochs.interpolate_bads(reset_bads=False)

    # save epochs to disk
    epochs.save(dir_data_eeg_cleaned + str(s) + "_epochs.fif", overwrite=True)

    return epochs


def load_epochs(dir_data_eeg_cleaned, s):
    epochs = mne.read_epochs(dir_data_eeg_cleaned + str(s) + "_epochs.fif")
    return epochs


def run_spatio_temporal(epochs):
    X = epochs.get_data()
    y = epochs.events[:, -1]

    n_splits = 3

    # within mode: everything on touch
    # touchtouch
    XX = X[(y == 3) | (y == 4)]
    yy = y[(y == 3) | (y == 4)]
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True)
    cv = list(skf.split(XX, yy))
    clf = make_pipeline(
        Scaler(epochs.info), Vectorizer(), LogisticRegression(solver="liblinear")
    )
    scores_touchtouch = cross_val_multiscore(clf, XX, yy, cv=cv, n_jobs=1)

    # within mode: everything on vision
    # visvis
    XX = X[(y == 1) | (y == 2)]
    yy = y[(y == 1) | (y == 2)]
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True)
    cv = list(skf.split(XX, yy))
    clf = make_pipeline(
        Scaler(epochs.info), Vectorizer(), LogisticRegression(solver="liblinear")
    )
    scores_visvis = cross_val_multiscore(clf, XX, yy, cv=cv, n_jobs=1)

    # cross mode: train on touch / test on vision
    # touchvis
    XX = X.copy()
    yy = y.copy()
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True)
    train_ind = np.where((yy == 3) | (yy == 4))[0]
    test_ind = np.where((yy == 1) | (yy == 2))[0]
    yy[yy == 3] = 1
    yy[yy == 4] = 2
    np.random.shuffle(train_ind)
    np.random.shuffle(test_ind)
    train_ind = np.array_split(train_ind, n_splits)
    test_ind = np.array_split(test_ind, n_splits)
    cv = list(zip(train_ind, test_ind))
    clf = make_pipeline(
        Scaler(epochs.info), Vectorizer(), LogisticRegression(solver="liblinear")
    )
    scores_touchvis = cross_val_multiscore(clf, XX, yy, cv=cv, n_jobs=1)

    # cross mode: train on vision / test on touch
    # vistouch
    XX = X.copy()
    yy = y.copy()
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True)
    train_ind = np.where((yy == 1) | (yy == 2))[0]
    test_ind = np.where((yy == 3) | (yy == 4))[0]
    yy[yy == 3] = 1
    yy[yy == 4] = 2
    np.random.shuffle(train_ind)
    np.random.shuffle(test_ind)
    train_ind = np.array_split(train_ind, n_splits)
    test_ind = np.array_split(test_ind, n_splits)
    cv = list(zip(train_ind, test_ind))
    clf = make_pipeline(
        Scaler(epochs.info), Vectorizer(), LogisticRegression(solver="liblinear")
    )
    scores_vistouch = cross_val_multiscore(clf, XX, yy, cv=cv, n_jobs=1)

    return (scores_touchtouch, scores_visvis, scores_touchvis, scores_vistouch)


def run_csp(epochs):
    X = epochs.get_data()
    y = epochs.events[:, -1]

    n_splits = 3

    # within mode: everything on touch
    # touchtouch
    XX = X[(y == 3) | (y == 4)]
    yy = y[(y == 3) | (y == 4)]
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True)
    cv = list(skf.split(XX, yy))
    csp = CSP(n_components=2, reg=None, log=True, norm_trace=False)
    lda = LinearDiscriminantAnalysis()
    clf = Pipeline([("CSP", csp), ("LDA", lda)])
    scores_touchtouch = cross_val_multiscore(clf, XX, yy, cv=cv, n_jobs=1)

    # within mode: everything on vision
    # visvis
    XX = X[(y == 1) | (y == 2)]
    yy = y[(y == 1) | (y == 2)]
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True)
    cv = list(skf.split(XX, yy))
    csp = CSP(n_components=2, reg=None, log=True, norm_trace=False)
    lda = LinearDiscriminantAnalysis()
    clf = Pipeline([("CSP", csp), ("LDA", lda)])
    scores_visvis = cross_val_multiscore(clf, XX, yy, cv=cv, n_jobs=1)

    # cross mode: train on touch / test on vision
    # touchvis
    XX = X.copy()
    yy = y.copy()
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True)
    train_ind = np.where((yy == 3) | (yy == 4))[0]
    test_ind = np.where((yy == 1) | (yy == 2))[0]
    yy[yy == 3] = 1
    yy[yy == 4] = 2
    np.random.shuffle(train_ind)
    np.random.shuffle(test_ind)
    train_ind = np.array_split(train_ind, n_splits)
    test_ind = np.array_split(test_ind, n_splits)
    cv = list(zip(train_ind, test_ind))
    csp = CSP(n_components=2, reg=None, log=True, norm_trace=False)
    lda = LinearDiscriminantAnalysis()
    clf = Pipeline([("CSP", csp), ("LDA", lda)])
    scores_touchvis = cross_val_multiscore(clf, XX, yy, cv=cv, n_jobs=1)

    # cross mode: train on vision / test on touch
    # vistouch
    XX = X.copy()
    yy = y.copy()
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True)
    train_ind = np.where((yy == 1) | (yy == 2))[0]
    test_ind = np.where((yy == 3) | (yy == 4))[0]
    yy[yy == 3] = 1
    yy[yy == 4] = 2
    np.random.shuffle(train_ind)
    np.random.shuffle(test_ind)
    train_ind = np.array_split(train_ind, n_splits)
    test_ind = np.array_split(test_ind, n_splits)
    cv = list(zip(train_ind, test_ind))
    csp = CSP(n_components=2, reg=None, log=True, norm_trace=False)
    lda = LinearDiscriminantAnalysis()
    clf = Pipeline([("CSP", csp), ("LDA", lda)])
    scores_vistouch = cross_val_multiscore(clf, XX, yy, cv=cv, n_jobs=1)

    return (scores_touchtouch, scores_visvis, scores_touchvis, scores_vistouch)


def run_time_decoding(epochs):
    X = epochs.get_data()
    y = epochs.events[:, -1]

    n_splits = 3
    metric = "accuracy"

    # within mode: everything on touch
    # touchtouch
    XX = X[(y == 3) | (y == 4)]
    yy = y[(y == 3) | (y == 4)]
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True)
    cv = list(skf.split(XX, yy))
    clf = LinearDiscriminantAnalysis(solver="svd")
    pipe = Pipeline([("scl", Scaler(epochs.info)), ("vec", Vectorizer()), ("clf", clf)])
    time_gen = SlidingEstimator(pipe, n_jobs=-1, scoring=metric, verbose=True)
    scores = cross_val_multiscore(time_gen, XX, yy, cv=cv, n_jobs=-1)
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
    time_gen = SlidingEstimator(pipe, n_jobs=-1, scoring=metric, verbose=True)
    scores = cross_val_multiscore(time_gen, XX, yy, cv=cv, n_jobs=-1)
    scores = np.mean(scores, 0)
    scores_visvis = scores

    # cross mode: train on touch / test on vision
    # touchvis
    XX = X.copy()
    yy = y.copy()
    train_ind = np.where((yy == 3) | (yy == 4))[0]
    test_ind = np.where((yy == 1) | (yy == 2))[0]
    yy[yy == 3] = 1
    yy[yy == 4] = 2
    np.random.shuffle(train_ind)
    np.random.shuffle(test_ind)
    train_ind = np.array_split(train_ind, n_splits)
    test_ind = np.array_split(test_ind, n_splits)
    cv = list(zip(train_ind, test_ind))
    pipe = Pipeline([("scl", Scaler(epochs.info)), ("vec", Vectorizer()), ("clf", clf)])
    time_gen = SlidingEstimator(pipe, n_jobs=-1, scoring=metric, verbose=True)
    scores = cross_val_multiscore(time_gen, XX, yy, cv=cv, n_jobs=-1)
    scores = np.mean(scores, 0)
    scores_touchvis = scores

    # cross mode: train on vision / test on touch
    # vistouch
    XX = X.copy()
    yy = y.copy()
    train_ind = np.where((yy == 1) | (yy == 2))[0]
    test_ind = np.where((yy == 3) | (yy == 4))[0]
    yy[yy == 3] = 1
    yy[yy == 4] = 2
    np.random.shuffle(train_ind)
    np.random.shuffle(test_ind)
    train_ind = np.array_split(train_ind, n_splits)
    test_ind = np.array_split(test_ind, n_splits)
    cv = list(zip(train_ind, test_ind))
    clf = LinearDiscriminantAnalysis(solver="svd")
    pipe = Pipeline([("scl", Scaler(epochs.info)), ("vec", Vectorizer()), ("clf", clf)])
    time_gen = SlidingEstimator(pipe, n_jobs=-1, scoring=metric, verbose=True)
    scores = cross_val_multiscore(time_gen, XX, yy, cv=cv, n_jobs=-1)
    scores = np.mean(scores, 0)
    scores_vistouch = scores

    return (scores_touchtouch, scores_visvis, scores_touchvis, scores_vistouch)


def run_time_gens(epochs):
    X = epochs.get_data()
    y = epochs.events[:, -1]

    n_splits = 3
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
    scores = cross_val_multiscore(time_gen, XX, yy, cv=cv, n_jobs=-1)
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
    scores = cross_val_multiscore(time_gen, XX, yy, cv=cv, n_jobs=-1)
    scores = np.mean(scores, 0)
    scores_visvis = scores

    # cross mode: train on touch / test on vision
    # touchvis
    XX = X.copy()
    yy = y.copy()
    train_ind = np.where((yy == 3) | (yy == 4))[0]
    test_ind = np.where((yy == 1) | (yy == 2))[0]
    yy[yy == 3] = 1
    yy[yy == 4] = 2
    np.random.shuffle(train_ind)
    np.random.shuffle(test_ind)
    train_ind = np.array_split(train_ind, n_splits)
    test_ind = np.array_split(test_ind, n_splits)
    cv = list(zip(train_ind, test_ind))
    pipe = Pipeline([("scl", Scaler(epochs.info)), ("vec", Vectorizer()), ("clf", clf)])
    time_gen = GeneralizingEstimator(pipe, n_jobs=-1, scoring=metric, verbose=True)
    scores = cross_val_multiscore(time_gen, XX, yy, cv=cv, n_jobs=-1)
    scores = np.mean(scores, 0)
    scores_touchvis = scores

    # cross mode: train on vision / test on touch
    # vistouch
    XX = X.copy()
    yy = y.copy()
    train_ind = np.where((yy == 1) | (yy == 2))[0]
    test_ind = np.where((yy == 3) | (yy == 4))[0]
    yy[yy == 3] = 1
    yy[yy == 4] = 2
    np.random.shuffle(train_ind)
    np.random.shuffle(test_ind)
    train_ind = np.array_split(train_ind, n_splits)
    test_ind = np.array_split(test_ind, n_splits)
    cv = list(zip(train_ind, test_ind))
    clf = LinearDiscriminantAnalysis(solver="svd")
    pipe = Pipeline([("scl", Scaler(epochs.info)), ("vec", Vectorizer()), ("clf", clf)])
    time_gen = GeneralizingEstimator(pipe, n_jobs=-1, scoring=metric, verbose=True)
    scores = cross_val_multiscore(time_gen, XX, yy, cv=cv, n_jobs=-1)
    scores = np.mean(scores, 0)
    scores_vistouch = scores

    return (scores_touchtouch, scores_visvis, scores_touchvis, scores_vistouch)


def filter_data(r):
    rf = r.copy().filter(fmin, fmax, fir_design="firwin", skip_by_annotation="edge")
    return rf


def run_timefreq_parallel(freq_range):
    fmin, fmax = freq_range
    w_size = n_cycles / ((fmax + fmin) / 2.0)
    raw_filt = []

    print("made it")
    print(raw_filt)

    # Apply band-pass filter to isolate the specified frequencies
    # raw_filt = []
    # pool = multiprocessing.Pool()
    # raw_filt = pool.map(filter_data, raw)
    # pool.close()
    # pool.join()

    print("made it")
    print(raw_filt)

#    decim = 1
#    epochs = compute_epochs(raw_filt, events, trial_mat, decim, dir_data_eeg_cleaned, s)
#
#    n_splits = 3
#
#    X = epochs.get_data()
#    y = epochs.events[:, -1]
#
#    # touchtouch
#    XX = X[(y == 3) | (y == 4)]
#    yy = y[(y == 3) | (y == 4)]
#    skf = StratifiedKFold(n_splits=n_splits, shuffle=True)
#    cv = list(skf.split(XX, yy))
#    clf = make_pipeline(
#        Scaler(epochs.info), Vectorizer(), LogisticRegression(solver="liblinear")
#    )
#    scores_touchtouch = cross_val_multiscore(
#        clf, XX, yy, cv=cv, scoring="accuracy", n_jobs=-1
#    )
#    freq_scores = np.mean(scores_touchtouch, axis=0)
#
#    return freq_scores


def run_timefreq(raw, events, trial_mat, dir_data_eeg_cleaned, s):
    # Classification & time-frequency parameters
    tmin, tmax = -0.200, 2.000
    n_cycles = 10.0  # how many complete cycles: used to define window size
    min_freq = 8.0
    max_freq = 20.0
    n_freqs = 6  # how many frequency bins to use

    # Assemble list of frequency range tuples
    freqs = np.linspace(min_freq, max_freq, n_freqs)  # assemble frequencies
    freq_ranges = list(zip(freqs[:-1], freqs[1:]))  # make freqs list of tuples

    # Infer window spacing from the max freq and number of cycles to avoid gaps
    window_spacing = n_cycles / np.max(freqs) / 2.0
    centered_w_times = np.arange(tmin, tmax, window_spacing)[1:]
    n_windows = len(centered_w_times)

    # init scores
    # freq_scores = np.zeros((n_freqs - 1,))

    # Loop through each frequency range of interest in parallel
    pool = multiprocessing.Pool()
    freq_scores = pool.map(run_timefreq_parallel, freq_ranges)
    pool.close()
    pool.join()

    # Loop through each frequency range of interest
    # for freq, (fmin, fmax) in enumerate(freq_ranges):
    #     # Infer window size based on the frequency being used
    #     w_size = n_cycles / ((fmax + fmin) / 2.0)  # in seconds

    #     # Apply band-pass filter to isolate the specified frequencies
    #     raw_filt = []
    #     pool = multiprocessing.Pool()
    #     raw_filt = pool.map(filter_data, raw)
    #     pool.close()
    #     pool.join()

    #     # Apply band-pass filter to isolate the specified frequencies
    #     # for r in raw:
    #     #     rf = r.copy().filter(
    #     #         fmin, fmax, fir_design="firwin", skip_by_annotation="edge"
    #     #     )
    #     #     raw_filt.append(rf)

    #     decim = 1
    #     epochs = compute_epochs(
    #         raw_filt, events, trial_mat, decim, dir_data_eeg_cleaned, s
    #     )

    #     n_splits = 3

    #     X = epochs.get_data()
    #     y = epochs.events[:, -1]

    #     # touchtouch
    #     XX = X[(y == 3) | (y == 4)]
    #     yy = y[(y == 3) | (y == 4)]
    #     skf = StratifiedKFold(n_splits=n_splits, shuffle=True)
    #     cv = list(skf.split(XX, yy))
    #     clf = make_pipeline(
    #         Scaler(epochs.info), Vectorizer(), LogisticRegression(solver="liblinear")
    #     )
    #     scores_touchtouch = cross_val_multiscore(
    #         clf, XX, yy, cv=cv, scoring="accuracy", n_jobs=-1
    #     )
    #     freq_scores[freq] = np.mean(scores_touchtouch, axis=0)

    np.save("../results/elen/" + s + "_timefreq.npy", freq_scores)

    plt.bar(
        freqs[:-1],
        freq_scores,
        width=np.diff(freqs)[0],
        align="edge",
        edgecolor="black",
    )
    plt.xticks(freqs)
    plt.ylim([0, 1])
    plt.axhline(
        0.5,
        color="k",
        linestyle="--",
        label="chance level",
    )
    plt.legend()
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Decoding Scores")
    plt.title("Frequency Decoding Scores")
    plt.savefig("../figures/elen/" + s + "_csp_timefreq.png")
    plt.close()


def inspect_spatio_temporal():
    st = np.loadtxt("st.txt")

    st_touchtouch = st[0::4, :].mean(1)
    st_visvis = st[1::4, :].mean(1)
    st_touchvis = st[2::4, :].mean(1)
    st_vistouch = st[3::4, :].mean(1)

    res = ttest_1samp(st_touchtouch, 0.5, alternative="greater")
    print("\n")
    print("touchtouch")
    print(res)

    res = ttest_1samp(st_visvis, 0.5, alternative="greater")
    print("\n")
    print("visvis")
    print(res)

    res = ttest_1samp(st_touchvis, 0.5, alternative="greater")
    print("\n")
    print("touchvis")
    print(res)

    res = ttest_1samp(st_vistouch, 0.5, alternative="greater")
    print("\n")
    print("vistouch")
    print(res)


def inspect_csp():
    csp = np.loadtxt("csp.txt")

    csp_touchtouch = csp[0::4, :].mean(1)
    csp_visvis = csp[1::4, :].mean(1)
    csp_touchvis = csp[2::4, :].mean(1)
    csp_vistouch = csp[3::4, :].mean(1)

    res = ttest_1samp(csp_touchtouch, 0.5, alternative="greater")
    print("\n")
    print("touchtouch")
    print(res)

    res = ttest_1samp(csp_visvis, 0.5, alternative="greater")
    print("\n")
    print("visvis")
    print(res)

    res = ttest_1samp(csp_touchvis, 0.5, alternative="greater")
    print("\n")
    print("touchvis")
    print(res)

    res = ttest_1samp(csp_vistouch, 0.5, alternative="greater")
    print("\n")
    print("vistouch")
    print(res)


def inspect_timegen(dir_results, dir_figs, s_list, f_name):
    timegen_list = []
    for f in os.listdir(dir_results):
        if f != ".DS_Store":
            # print(f)
            # print(f[-9:-7])
            s = f[-9:-7]
            if s in s_list:
                timegen_list.append(np.load(dir_results + f))

    # decimate for ease of plotting
    timegen_list = [x[:, ::2, ::2] for x in timegen_list]

    timegen_touchtouch = [x[0] for x in timegen_list]
    timegen_touchtouch = np.dstack(timegen_touchtouch)

    timegen_visvis = [x[1] for x in timegen_list]
    timegen_visvis = np.dstack(timegen_visvis)

    timegen_vistouch = [x[2] for x in timegen_list]
    timegen_vistouch = np.dstack(timegen_vistouch)

    timegen_touchvis = [x[3] for x in timegen_list]
    timegen_touchvis = np.dstack(timegen_touchvis)

    # times = epochs_list[0].times
    # times = np.linspace(-1.1, 1.8, timegen_touchtouch.shape[1])

    touch_diag = timegen_touchtouch.diagonal()
    touch_diag_df = pd.DataFrame(touch_diag)
    touch_diag_df = touch_diag_df.melt(var_name="time", value_name="accuracy")

    vis_diag = timegen_visvis.diagonal()
    vis_diag_df = pd.DataFrame(vis_diag)
    vis_diag_df = vis_diag_df.melt(var_name="time", value_name="accuracy")

    fig, ax = plt.subplots(3, 2, squeeze=False, figsize=(10, 10))

    sns.lineplot(touch_diag_df, x="time", y="accuracy", ax=ax[0, 0])
    sns.lineplot(vis_diag_df, x="time", y="accuracy", ax=ax[0, 1])

    ax[0, 0].axhline(0.5, linestyle="--", color="k")
    ax[0, 1].axhline(0.5, linestyle="--", color="k")

    sns.heatmap(timegen_touchtouch.mean(axis=2).T, ax=ax[1, 0])
    sns.heatmap(timegen_visvis.mean(axis=2).T, ax=ax[1, 1])
    sns.heatmap(timegen_vistouch.mean(axis=2).T, ax=ax[2, 0])
    sns.heatmap(timegen_touchvis.mean(axis=2).T, ax=ax[2, 1])

    [x.invert_yaxis() for x in ax[1:, :].flatten()]
    ax[1, 0].set_title("touchtouch")
    ax[1, 1].set_title("visvis")
    ax[2, 0].set_title("vistouch")
    ax[2, 1].set_title("touchvis")

    for a in ax.flatten():
        ticks = np.linspace(0, timegen_touchtouch.shape[0], 8)
        labels = np.round(np.linspace(-1.1, 1.8, ticks.shape[0]), 1)
        a.set_xticks(ticks)
        a.set_xticklabels(labels)

    for a in ax[1:, :].flatten():
        ticks = np.linspace(0, timegen_touchtouch.shape[0], 8)
        labels = np.round(np.linspace(-1.1, 1.8, ticks.shape[0]), 1)
        a.set_yticks(ticks)
        a.set_yticklabels(labels)

    plt.tight_layout()
    plt.savefig(dir_figs + "/" + f_name)
    plt.close()


def inspect_timefreq(dir_results, dir_figs, s_list, f_name):
    timefreq_list = []
    for f in os.listdir(dir_results):
        if f != ".DS_Store":
            print(f)
            s = f[:2]
            if s in s_list:
                timefreq_list.append(np.load(dir_results + f))

    print(timefreq_list)


def plot_raw(raw, ax):
    # n_chan x n_samp
    X = raw.get_data()
    means = np.mean(X, axis=1, keepdims=True)
    X = X - means
    X = X[:, ::5000]
    x = np.arange(0, X.shape[1], 1)
    for i in range(X.shape[0]):
        ax.plot(x, X[i, :], linewidth=1.0)


def plot_raw_psd(raw, f_fig_name):
    fig, ax = plt.subplots(2, 3, squeeze=False, figsize=(8, 5))
    for i in range(len(raw)):
        plot_raw(raw[i], ax=ax[0, i])
        raw[i].compute_psd(fmax=250).plot(average=False, axes=ax[1, i])
    plt.tight_layout()
    plt.savefig(f_fig_name)
    plt.close()


def plot_epochs(epochs, f_fig_name):
    fig, ax = plt.subplots(2, 2, squeeze=False, figsize=(8, 5))
    for i, k in enumerate(epochs.event_id.keys()):
        epochs[k].average().plot(spatial_colors=True, axes=ax.flatten()[i], show=False)
        ax.flatten()[i].set_title(k)
    plt.tight_layout()
    plt.savefig(f_fig_name)
    plt.close()


def plot_epochs_freq(epochs, f_fig_name):
    fig, ax = plt.subplots(2, 2, squeeze=False, figsize=(8, 5))
    for i, k in enumerate(epochs.event_id.keys()):
        epochs[k].compute_psd(fmax=40.0).plot(average=False, axes=ax.flatten()[i])
        ax.flatten()[i].set_title(k)
    plt.tight_layout()
    plt.savefig(f_fig_name)
    plt.close()


def plot_epochs_freq_topo(epochs, f_fig_name):
    fig, ax = plt.subplots(4, 5, squeeze=False, figsize=(12, 7))
    for i, k in enumerate(epochs.event_id.keys()):
        epochs[k].compute_psd(fmax=40.0).plot_topomap(axes=ax[i, :])
        ax[i, 0].set_ylabel(k)
    plt.tight_layout()
    plt.savefig(f_fig_name)
    plt.close()


def plot_epochs_ave(epochs_list, f_fig_name):
    epochs = mne.concatenate_epochs(epochs_list)
    fig, ax = plt.subplots(2, 2, squeeze=False, figsize=(8, 5))
    for i, k in enumerate(epochs.event_id.keys()):
        epochs[k].average().plot(spatial_colors=True, axes=ax.flatten()[i], show=False)
        ax.flatten()[i].set_title(k)
    plt.tight_layout()
    plt.savefig(f_fig_name)
