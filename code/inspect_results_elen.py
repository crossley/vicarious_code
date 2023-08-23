from imports import *
from util_funcs_elen import *

mpl.use("TkAgg")

res = create_sub_nums_sophie_trmdt()
# res = create_sub_nums_sophie()
# res = create_sub_nums_elen()

sub_nums = res[0]
dir_data_behaviour = res[1]
dir_data_eeg = res[2]
dir_figs = res[3]
dir_results = res[4]

print(sub_nums)
# sub_nums = [sub_nums[0]]

epochs_list = []
csp_list = []
st_list = []
timegen_list = []
for s in sub_nums:
    print(s)

    trial_mat = load_trial_mat(dir_data_behaviour, s)
    raw, events = load_raw(dir_data_eeg, s)
    epochs = compute_epochs(raw, events, trial_mat)

    epochs_list.append(epochs)

    # TODO: save raw figure really zoomed out along with psd etc for quality
    # diagnostic

    # NOTE: Combine across trials but keep sensors separate
    fig, ax = plt.subplots(2, 2, squeeze=False, figsize=(8, 5))
    for i, k in enumerate(epochs.event_id.keys()):
        epochs[k].average().plot(spatial_colors=True, axes=ax.flatten()[i], show=False)
        ax.flatten()[i].set_title(k)
    plt.tight_layout()
    plt.savefig(dir_figs + "fig_epochs_sub_" + s + ".pdf")

    # TODO: Do both in the time domain as well in the frequency domain
    # TODO: what is the mutual information between visual and tactile trials?

    # TODO: Is information present at all (spatio-tepmoral classifier)? -- V1
    # scores_st = run_spatio_temporal(epochs)
    # st_list.append(scores_st)

    # TODO: Is information present at all (CSP classifier)? -- V2
    # scores_csp = run_csp(epochs)
    # csp_list.append(scores_csp)

    # TODO: When is information present (time gen)?
    epochs.resample(sfreq=150)
    scores_timegen = run_time_gens(epochs)
    timegen_list.append(scores_timegen)

    # fig, ax = plt.subplots(2, 2, squeeze=False)
    # ax[0, 0].imshow(scores_timegen[0])
    # ax[0, 1].imshow(scores_timegen[1])
    # ax[1, 0].imshow(scores_timegen[2])
    # ax[1, 1].imshow(scores_timegen[3])
    # plt.savefig(dir_figs + 'timegen.pdf')
    # plt.show()

    del raw, epochs

# TODO: Move all this to dedicated util functions
epochs = mne.concatenate_epochs(epochs_list)
fig, ax = plt.subplots(2, 2, squeeze=False, figsize=(8, 5))
for i, k in enumerate(epochs.event_id.keys()):
    epochs[k].average().plot(spatial_colors=True, axes=ax.flatten()[i], show=False)
    ax.flatten()[i].set_title(k)
plt.tight_layout()
plt.savefig(dir_figs + "fig_epochs_sub_" + "ave" + ".pdf")

# st = np.concatenate(st_list)
# np.savetxt('st.txt', st)

# st_touchtouch = st[0::4, :].mean(1)
# st_visvis = st[1::4, :].mean(1)
# st_touchvis = st[2::4, :].mean(1)
# st_vistouch = st[3::4, :].mean(1)

# from scipy.stats import ttest_1samp
# res = ttest_1samp(st_touchtouch, 0.5, alternative='greater')
# print(res.pvalue)
# res = ttest_1samp(st_visvis, 0.5, alternative='greater')
# print(res.pvalue)
# res = ttest_1samp(st_touchvis, 0.5, alternative='greater')
# print(res.pvalue)
# res = ttest_1samp(st_vistouch, 0.5, alternative='greater')
# print(res.pvalue)

# csp = np.concatenate(csp_list)
# np.savetxt('csp.txt', csp)

# csp_touchtouch = csp[0::4, :].mean(1)
# csp_visvis = csp[1::4, :].mean(1)
# csp_touchvis = csp[2::4, :].mean(1)
# csp_vistouch = csp[3::4, :].mean(1)

# from scipy.stats import ttest_1samp
# res = ttest_1samp(csp_touchtouch, 0.5, alternative='greater')
# print(res.pvalue)
# res = ttest_1samp(csp_visvis, 0.5, alternative='greater')
# print(res.pvalue)
# res = ttest_1samp(csp_touchvis, 0.5, alternative='greater')
# print(res.pvalue)
# res = ttest_1samp(csp_vistouch, 0.5, alternative='greater')
# print(res.pvalue)

timegen_touchtouch = [x[0] for x in timegen_list]
timegen_touchtouch = np.dstack(timegen_touchtouch)
np.save(dir_results + 'timegen_touchtouch.npy', timegen_touchtouch)

timegen_visvis = [x[1] for x in timegen_list]
timegen_visvis = np.dstack(timegen_visvis)
np.save(dir_results + 'timegen_visvis.npy', timegen_visvis)

timegen_vistouch = [x[2] for x in timegen_list]
timegen_vistouch = np.dstack(timegen_vistouch)
np.save(dir_results + 'timegen_vistouch.npy', timegen_vistouch)

timegen_touchvis = [x[3] for x in timegen_list]
timegen_touchvis = np.dstack(timegen_touchvis)
np.save(dir_results + 'timegen_touchvis.npy', timegen_touchvis)

fig, ax = plt.subplots(2, 2, squeeze=False)
sns.heatmap(timegen_touchtouch.mean(axis=2).T, ax=ax[0, 0])
sns.heatmap(timegen_visvis.mean(axis=2).T, ax=ax[0, 1])
sns.heatmap(timegen_vistouch.mean(axis=2).T, ax=ax[1, 0])
sns.heatmap(timegen_touchvis.mean(axis=2).T, ax=ax[1, 1])
[x.invert_yaxis() for x in ax.flatten()]
plt.tight_layout()
plt.savefig(dir_figs + '/timegen.pdf')
