from imports import *
from util_funcs_elen import *

mpl.use("TkAgg")

if __name__ == "__main__":
    # TODO: ICA
    # TODO: freq decoding
    # TODO: put everything in BIDS format
    # TODO: pull events from cleaned raws instead of original raws
    # TODO: write a function to load filtered raws from file if they exist
    # TODO: write a function to load epochs from file if they exist
    # TODO: vtp = [1, 3, 4, 5, 8, 10, 11, 13, 14, 15, 16, 19]
    # TODO: ERP differences between Elen and soph (predict vs no predict)
    # TODO: time-freq and just freq (rename existing)
    # TODO: mutual information between visual and tactile trials at each time pair
    # TODO: more intelligent folder structure
    # TODO: parallelize all for loops

    # get subject number file and directory information
    # res = create_sub_nums_sophie()
    res = create_sub_nums_elen()

    # extract relevant variables from res
    sub_nums = res[0]
    dir_data_behaviour = res[1]
    dir_data_eeg = res[2]
    dir_figs = res[3]
    dir_results = res[4]

    # sub_nums = [sub_nums[0]]
    # sub_nums = sub_nums[[1, 17]]
    # print(sub_nums)

    res = sub_nums, dir_data_behaviour, dir_data_eeg, dir_figs, dir_results

    dir_data_eeg_cleaned = "/Users/mq20185996/projects/crazly/elen/data_eeg_cleaned/"

    # empty lists to use as containers for results
    st_list = []
    csp_list = []

    for s in sub_nums:
        trial_mat = load_trial_mat(dir_data_behaviour, s)

        # manually inspect raws to annotate bad channels
        # manually_inspect_raws(res, dir_data_eeg_cleaned, s)

        raw, events = load_raw(dir_data_eeg, s, dir_data_eeg_cleaned)
        # raw, events = interpolate_bad_raws(res, dir_data_eeg_cleaned, s)

        # raw_filt, decim = filter_raw(raw, dir_data_eeg_cleaned, s)
        # load_filt(dir_data_eeg_cleaned, s)

        # epochs = compute_epochs(raw_filt, events, trial_mat, decim, dir_data_eeg_cleaned, s)
        # epochs = load_epochs(dir_data_eeg_cleaned, s)

        # Is information present at all (spatio-tepmoral classifier)? -- V1
        # scores_st = run_spatio_temporal(epochs)
        # st_list.append(scores_st)
        # st = np.concatenate(st_list)
        # np.savetxt(dir_results + "st.txt", st)

        # Is information present at all (CSP classifier)? -- V2
        # scores_csp = run_csp(epochs)
        # csp_list.append(scores_csp)
        # csp = np.concatenate(csp_list)
        # np.savetxt('csp.txt', csp)

        # When is information present (within modality diag)?
        # scores_time_decode = run_time_decoding(epochs)
        # np.save(dir_results + "time_decode_." + s + "npy", scores_time_decode)

        # When is information present (time gen)?
        # scores_timegen = run_time_gens(epochs)
        # np.save(dir_results + "timegen_." + s + "npy", scores_timegen)

        # run_timefreq(raw, events, trial_mat, dir_data_eeg_cleaned, s)

        # del raw, raw_filt, epochs

    # f_fig_name = dir_figs + "fig_epochs_sub_" + "ave" + ".pdf"
    # plot_epochs_ave(epochs_list, f_fig_name)

    # inspect_spatio_temporal()

    # inspect_csp()

    # s_vt_no = [1, 3, 4, 5, 8, 10, 11, 13, 14, 15, 16, 19]
    # s_vt_no = [str(x) for x in s_vt_no]
    # inspect_timegen(dir_results, dir_figs, s_vt_no, "timegen_vt_no.png")
    #
    # s_vt_yes = [2, 6, 7, 9, 12, 17, 18, 20 , 21]
    # s_vt_yes = [str(x) for x in s_vt_yes]
    # inspect_timegen(dir_results, dir_figs, s_vt_yes, "timegen_vt_yes.png")
    #
    # s_vt_all = s_vt_no + s_vt_yes
    # s_vt_all = [str(x) for x in s_vt_all]
    ### inspect_timegen(dir_results, dir_figs, s_vt_all, "timegen_vt_all.png")

    # s_list = sub_nums
    # s_list = [str(x) for x in s_list]
    # inspect_timefreq(dir_results + "timefreq/", dir_figs, s_list, "timefreq.png")

    # NOTE: plot raws, psds, epochs, etc.
    # f_fig_name = dir_figs + "fig_sub_" + s + "_raw.pdf"
    # plot_raw_psd(raw, f_fig_name)

    # f_fig_name = dir_figs + "fig_sub_" + s + "_raw_filt.pdf"
    # plot_raw_psd(raw_filt, f_fig_name)

    # f_fig_name = dir_figs + "fig_sub_" + s + "_epochs.pdf"
    # plot_epochs(epochs, f_fig_name)

    # f_fig_name = dir_figs + "fig_sub_" + s + "_epochs_freq.pdf"
    # plot_epochs_freq(epochs, f_fig_name)

    # f_fig_name = dir_figs + "fig_sub_" + s + "_epochs_freq_topo.pdf"
    # plot_epochs_freq_topo(epochs, f_fig_name)
