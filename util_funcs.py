from imports import *


def load_trial_mat(subject):

    mat_1 = scipy.io.loadmat(subject + '/' + subject[-3:] +
                             '_Session01_Touch_Processing_EEG.mat')
    mat_2 = scipy.io.loadmat(subject + '/' + subject[-3:] +
                             '_Session02_Touch_Processing_EEG.mat')
    mat_3 = scipy.io.loadmat(subject + '/' + subject[-3:] +
                             '_Session03_Touch_Processing_EEG.mat')

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

    return trial_mat


def load_epochs(subject, trial_mat, detrend_epoch_dir):

    epochs = mne.io.read_epochs_eeglab(detrend_epoch_dir + 'sub-' +
                                       subject[-2:] +
                                       '_task-touchdecoding_continuous.set')

    # TODO: Clarify mode == visual is and what else is possible / interesting
    # finger | mode == visual
    target = trial_mat[:, 5]
    trial_mat = trial_mat[target == 1, :]
    events_finger = epochs.events[(epochs.events[:, 2] == 1) &
                                  (target == 1), :]
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
    epochs.events = events_finger

    # resample to reduce compute time
    # epochs = epochs.resample(sfreq=90)

    X = epochs.get_data()
    X = X[target == 1, :]
    y = epochs.events[:, -1]

    return epochs, X, y


def time_gen(X, y, cv, clf, metric, tag):

    pipe = Pipeline([('scl', Scaler(epochs.info)), ('vec', Vectorizer()),
                     ('clf', clf)])

    time_gen = GeneralizingEstimator(pipe,
                                     n_jobs=-1,
                                     scoring=metric,
                                     verbose=True)

    scores = cross_val_multiscore(time_gen, X, y, cv=cv, n_jobs=-1)
    scores = np.mean(scores, 0)
    np.savetxt(
        '/Users/mq20185996/Dropbox/crazly/scores_' + tag + '_' + s[-3:] +
        '.txt', scores)


def run_time_gens():

    # within mode: everything on touch
    tag = 'touchtouch'
    XX = X[(y == 3) | (y == 4)]
    yy = y[(y == 3) | (y == 4)]
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True)
    cv = list(skf.split(XX, yy))
    clf = LinearDiscriminantAnalysis(solver='svd')
    tag = tag + '_lda'
    time_gen(XX, yy, cv, clf, metric, tag)

    # within mode: everything on vision
    tag = 'visvis'
    XX = X[(y == 1) | (y == 2)]
    yy = y[(y == 1) | (y == 2)]
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True)
    cv = list(skf.split(XX, yy))
    clf = LinearDiscriminantAnalysis(solver='svd')
    tag = tag + '_lda'
    time_gen(XX, yy, cv, clf, metric, tag)

    # cross mode: train on touch / test on vision
    tag = 'touchvis'
    XX = X
    yy = y
    train_ind = np.where((yy == 3) | (yy == 4))[0]
    test_ind = np.where((yy == 1) | (yy == 2))[0]
    np.random.shuffle(train_ind)
    np.random.shuffle(test_ind)
    train_ind = np.array_split(train_ind, n_splits)
    test_ind = np.array_split(test_ind, n_splits)
    cv = list(zip(train_ind, test_ind))
    tag = tag + '_lda'
    time_gen(XX, yy, cv, clf, metric, tag)

    # cross mode: train on vision / test on touch
    tag = 'vistouch'
    XX = X
    yy = y
    train_ind = np.where((yy == 1) | (yy == 2))[0]
    test_ind = np.where((yy == 3) | (yy == 4))[0]
    np.random.shuffle(train_ind)
    np.random.shuffle(test_ind)
    train_ind = np.array_split(train_ind, n_splits)
    test_ind = np.array_split(test_ind, n_splits)
    cv = list(zip(train_ind, test_ind))
    clf = LinearDiscriminantAnalysis(solver='svd')
    tag = tag + '_lda'
    time_gen(XX, yy, cv, clf, metric, tag)


def tune_hyper_params(X, y):

    # within mode: everything on touch
    tag = 'touchtouch'
    XX = X[(y == 3) | (y == 4)]
    yy = y[(y == 3) | (y == 4)]

    # pick a single time point based on previous time gen results
    # TODO: deal with hard code hack
    XX = XX[:, :, 106]

    # Split the dataset in two equal parts. One part to be used to generate
    # train / test splits for each grid, point, and the other part to be used
    # for overall validation.
    X_train, X_test, y_train, y_test = train_test_split(XX,
                                                        yy,
                                                        test_size=0.5,
                                                        random_state=0)

    train_ind_grid = np.where((y_train == 3) | (y_train == 4))[0]
    test_ind_grid = np.where((y_train == 3) | (y_train == 4))[0]

    train_ind_grid = np.array_split(train_ind_grid, n_splits)
    test_ind_grid = np.array_split(test_ind_grid, n_splits)

    np.random.shuffle(train_ind_grid)
    np.random.shuffle(test_ind_grid)

    cv_grid = list(zip(train_ind_grid, test_ind_grid))

    pipe = Pipeline([('scl', Scaler(epochs.info)), ('vec', Vectorizer()),
                     ('svc', SVC())])

    param_grid = [
        {
            'svc__C': [0.1, 1, 10, 100, 1000],
            'svc__kernel': ['linear']
        },
        {
            'svc__C': [0.1, 1, 10, 100, 1000],
            'svc__gamma': [0.001, 0.0001],
            'svc__kernel': ['rbf']
        },
    ]

    clf = GridSearchCV(estimator=pipe,
                       param_grid=param_grid,
                       scoring=metric,
                       cv=cv_grid,
                       verbose=1,
                       n_jobs=-1)
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
