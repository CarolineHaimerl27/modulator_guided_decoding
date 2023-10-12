import numpy as np
import matplotlib.pyplot as plt

from PLDS import PLDS
from PLDS_evaluation import model_train, fit_to_all_trials, model_test_lno, PLDS_rotations, compute_latent

import pickle
import os
import os.path
import sklearn.decomposition as decomposition
import seaborn as sb
import scipy.stats as stats
from scipy.stats import linregress, spearmanr
from scipy.optimize import minimize
import scipy.io as sio
import pandas as pd
import math
import multiprocessing as mp
import time

################## classes to fit SR & PLDS model #####################

class DATA:

    def par(self, coFR, remov, path_get, path_save,
            data_trial=None, mfr=None, D=None, Ttrials=None, X=None, counts0=None, day=None, att=None,
            mask_neuron=None, data=None, attention=None, orientation=None, contrast=None, trialind=None,
            bins0=None, name=None, block=None,mitte=None,
            ST_onoff_stim=None, ST_N_timebins=4, ST_aft_stim_windows=1, ST_inc_direction=False,
            betaGT=None, wGT=None, mGT=None, AGT=None,
            ISI_percmax =None, ISI_minms=None, isi_crit=False, simulate=False):
        self.data_trial = data_trial
        self.mfr = mfr
        self.D = D
        self.Ttrials = Ttrials
        self.X = X
        self.counts0 = counts0
        self.day = day
        self.att = att
        self.mask_neuron = mask_neuron
        self.data = data
        self.attention = attention
        self.orientation = orientation
        self.contrast = contrast
        self.trialind = trialind
        self.coFR = coFR
        self.remov = remov
        self.coFR = coFR
        self.path_get = path_get
        self.path_save = path_save
        self.bins0 = bins0
        self.name=name
        self.block = block
        self.mitte = mitte
        self.ST_N_timebins = ST_N_timebins
        self.ST_aft_stim_windows = ST_aft_stim_windows
        self.ST_inc_direction = ST_inc_direction
        self.ST_onoff_stim = ST_onoff_stim
        self.ISI_percmax = ISI_percmax
        self.ISI_minms = ISI_minms
        self.isi_crit = isi_crit
        self.simulate = simulate
        self.betaGT = betaGT
        self.mGT = mGT
        self.wGT = wGT

    def preprocess(self, count, DATAFILE, behavior_kickout=True, change_to=True, check_adapt=True,
                   code_kickout=None):

        self.day = DATAFILE[count,0]
        self.att =  DATAFILE[count,1]# which attentional condition
        self.block = DATAFILE[count,2]
        print('day '+np.str(self.day)+' attentional condition '+ np.str(self.att))

        ################################################################
        ################## Preprocessing ###############################
        ################################################################
        self.load_data(kickout=behavior_kickout, code_kickout=code_kickout)

        # reorder spikes and stimulus into trial structure
        self.load_spikes_stim(change_to=change_to)
        self.mfr = np.mean(np.nanmean(self.data_trial, axis=0), axis=1)

        # apply ISI criteria, FR criteria and FF criteria
        if self.isi_crit:
            ISI_kickout = self.ISI_test()
        if self.remov:
            print(' ')
            print('   total of %.0f units' %self.D)
            self.mask_neuron = np.ones(self.D, dtype='bool')
            if self.isi_crit:
                print('   remove neurons with ISI violations')
                self.mask_neuron[ISI_kickout] = False
                print('   left with %.0f neurons' % (np.sum(self.mask_neuron)))
            print('   remove neurons with firing rate lower than %.0f' %self.coFR)
            self.mask_neuron[self.mfr<self.coFR] = False
            print('   left with %.0f neurons' %(np.sum(self.mask_neuron)))

            mvar = np.nanmean(np.nanvar(self.data_trial, axis=0), axis=1)
            cutweird = (np.mean(mvar / self.mfr) + 5 * np.sqrt(np.var(mvar / self.mfr)))
            print('   remove neurons with a ff larger than %.2f (5 sd above mean)' %cutweird)
            self.mask_neuron[(mvar / self.mfr) > cutweird] = False
            print('   left with %.0f neurons' % (np.sum(self.mask_neuron)))

            self.data_trial = self.data_trial[:,self.mask_neuron,:]
            if change_to:
                self.data_trial_change = self.data_trial_change[:, self.mask_neuron, :]
            self.D = np.sum(self.mask_neuron)
            print('removed %.0f percent of neurons' %(np.mean(self.mask_neuron==0)*100))
            self.mfr = np.mean(np.nanmean(self.data_trial, axis=0), axis=1)
        # test for adaptation:
        if check_adapt:
            print(' ')
            self.stationary_check()

        # save itself
        print('class DATA saving itself under ', self.path_save +self.name + '.pk')
        pickle.dump(self, open(self.path_save +self.name + '.pk', 'wb'))

    def load_data(self, kickout=True, code_kickout=None):
        # load data: data are rpeprocessed in matlab and contain all trials except for empty ones and except for those
        # that had more than one stimulus change happening
        # if kickout is true there are automatic criteria for kicking out (failed) trials
        # if it is false then a specific array with code is given to know which behavioral conditions to kick out
        attention = sio.loadmat(self.path_get + str(self.day) + '_attention')['A'][0]
        orientation = sio.loadmat(self.path_get + str(self.day) + '_orientation')['O']
        contrast = sio.loadmat(self.path_get + str(self.day) + '_contrast')['C']
        presind = sio.loadmat(self.path_get + str(self.day) + '_presind')['presind'][0]
        trialind = sio.loadmat(self.path_get + str(self.day) + '_trialind')['trialind'][0]
        spikes = sio.loadmat(self.path_get + str(self.day) + '_spiketimes.mat')['spiketimes'][0, :]
        stimtimes = sio.loadmat(self.path_get + str(self.day) + '_stimtimes.mat')['stimtimes'][0, :]
        RT = sio.loadmat(self.path_get + str(self.day) + '_RT.mat')['RT'][0, :]
        sacc = sio.loadmat(self.path_get + str(self.day) + '_sacc_detected.mat')['sac_detected'][0, :]
        iscatch = sio.loadmat(self.path_get + str(self.day) + '_iscatch.mat')['iscatch'][0, :]
        isdistr = sio.loadmat(self.path_get + str(self.day) + '_isdistr.mat')['isdistr'][0, :]
        out = sio.loadmat(self.path_get + str(self.day) + '_outcome.mat')['out'][0, :]
        Ttrials_tot = len(out)
        trial_name = np.unique(trialind)

        if kickout:
            val_trials = np.zeros([Ttrials_tot, 6], dtype='bool')
            for tt in range(Ttrials_tot):
                val_trials[tt, 0] = (out[tt] == 1) # keep if trial was correct
                val_trials[tt, 1] = (isdistr[tt] == 0) # keep if trial was not a distractor
                val_trials[tt, 2] = (iscatch[tt] == 0) # keep if trial was not a catch
                val_trials[tt, 4] = (len(stimtimes[tt]) > 1) # kick out if only one stimulus presentation
                # chose task condition
                val_trials[tt, 5] = (attention[tt] == self.att)

        else:
            val_trials = np.ones([Ttrials_tot, len(code_kickout)+1], dtype='bool')
            for tt in range(Ttrials_tot):
                for cc in range(len(code_kickout)):
                    val_trials[tt, cc] = 1-(out[tt] ==code_kickout[cc])  # kick out if trial was a random saccade somewhere
                val_trials[tt, -1] = (attention[tt] == self.att)
        valid = np.where(np.sum(val_trials, axis=1) == val_trials.shape[1])[0]
        if self.block is not None:
            trial_name_valid = trial_name[valid]
            argmaxtrialdiff = np.argmax(np.diff(trial_name_valid))
            self.mitte = np.mean(trial_name_valid[(argmaxtrialdiff):(argmaxtrialdiff + 2)])
            if self.block==1:
                valid = valid[trial_name_valid<self.mitte]
            if self.block==2:
                valid = valid[trial_name_valid>self.mitte]
        self.trial_name = trial_name[valid]

        mask = np.zeros(len(trialind), dtype='bool')
        for tt in valid:
            mask[trialind == trial_name[tt]] = True

        self.attention = attention[valid]
        self.orientation = orientation[:, mask]
        self.contrast = contrast[:, mask]
        self.presind = presind[mask]
        self.trialind = trialind[mask]
        self.spikes = spikes[valid]
        self.stimtimes = stimtimes[valid]
        self.RT = RT[valid]
        self.sacc = sacc[valid]
        self.iscatch = iscatch[valid]
        self.isdistr = isdistr[valid]
        self.out = out[valid]
        self.Ttrials = len(valid)
        # saccade made
        self.saccyn = np.array(np.zeros(self.Ttrials), dtype='bool')
        self.saccyn[self.out == 1] = True
        self.saccyn[self.out > 4] = True

    def load_spikes_stim(self, max_stim_pres=20, exclude_first=True, change_to=True):

        # fixed parameters of recordings
        winsize = .2 / self.ST_N_timebins
        stim_timedelay = .03
        tot_windows = self.ST_N_timebins + self.ST_aft_stim_windows # in these windows we collect spike dyanmics after a stimulus
        self.microsaccyn = np.array(np.zeros(self.Ttrials), dtype='bool')
        neurons = np.unique(
            self.spikes[0][:, 0])  # assuming that the units don't shift between channels and are stable within a sess
        neurons = neurons[neurons != 97]  # ch 97 sort 1 is always the MT unit

        if change_to:
            SPIKES_change = np.zeros([self.ST_N_timebins, len(neurons), self.Ttrials]) * np.nan
            MT_SPIKES_change = np.zeros([self.ST_N_timebins, 1, self.Ttrials]) * np.nan


        if self.att<3:
            X_change = np.zeros([1 + self.ST_inc_direction, self.Ttrials]) * np.nan
            # session specific contrast and orientation condition, neuron indices
            ucontr = np.sort(np.unique(self.contrast[self.att - 1, :]))
            uorien = np.sort(np.unique(self.orientation[self.att - 1, :]))

            # make empty files to collect spikes and stimulus up to a change in stimulus or a saccade
            SPIKES = np.zeros([max_stim_pres * (tot_windows + 2 * self.ST_N_timebins), len(neurons), self.Ttrials]) * np.nan
            MT_SPIKES = np.zeros([max_stim_pres * (tot_windows + 2 * self.ST_N_timebins), 1, self.Ttrials]) * np.nan
            X = np.zeros([max_stim_pres * (tot_windows + 2 * self.ST_N_timebins),
                          1 + 2*self.ST_N_timebins + self.ST_aft_stim_windows + self.ST_onoff_stim + self.ST_inc_direction,
                          self.Ttrials]) * np.nan  # one is for block and is ignored for now

            valid = np.array(np.ones(self.Ttrials), dtype='bool')
            for tt in range(self.Ttrials):  # trial
                if np.any(np.isnan(self.stimtimes[tt][:, 0])):
                    valid[tt] = False
                    print('    warning: there are nans in the stimulus times, trial=%.0f' % tt)
                    continue
                # stimulus times and properties of this trial (go until the second to last stimulus)
                if exclude_first: # first stimulus presentation will not be anaylzed since there can be adaptation
                    want = np.arange(1, len(self.stimtimes[tt][:, 0]))
                else:
                    want = np.arange(len(self.stimtimes[tt][:, 0]))
                if (len(want) <= 2):
                    valid[tt] = False
                    print('    warning: number of valid stimulus presentations<2, trial=%.0f' % tt)
                    continue
                # stimulus times, stimulus properties (contrast, direction)
                stt = self.stimtimes[tt][:, 0][want]
                stt_del = stt + stim_timedelay
                contrast = self.contrast[self.att - 1, self.trial_name[tt] == self.trialind][want]
                direction = self.orientation[self.att - 1, self.trial_name[tt] == self.trialind][want]
                nochange = (np.sum(np.abs(np.diff(direction)) > 0) == 0)
                if nochange:  # no change occured --> take end of trial (after last stimulus presentation)
                    print('no change occured on trial ' + np.str(tt), ' where outcome: ', self.out[tt])
                    valid[tt] = False
                    continue
                else:
                    prechange_stt = np.where(np.diff(direction) > 0)[0][0] # last stimulus before change occured!!!!!
                    stim_change = stt[prechange_stt+1]  # stimulus change

                # spike times of this trial
                sptt = self.spikes[tt]

                times = np.zeros(200)*np.nan
                countii = 0
                for ii in range(len(stt) - 1):
                    # are we still before the end point (defined by change happening)
                    if stt_del[ii]<=stim_change:
                        # what ends this window, new stimulus or end point
                        endii = np.min([stt_del[ii+1], stim_change])
                        # break up this time window into x ms time windows
                        tmp = np.arange(stt_del[ii], endii - .0001, winsize)
                        times[countii: (countii+len(tmp))] = tmp
                        countii += len(tmp)
                times = times[np.isnan(times)==False]
                if len(times)==0:
                    valid[tt] = False
                    print('    warning: this trial has no valid time points, something is off, trial=%.0f' % tt)
                    continue

                ###############################################################################################
                ##################################### target response #########################################
                ###############################################################################################
                # RESPONSE either before saccade or before change point (response to target - changed - stimulus)
                if change_to & (nochange==False):

                    # compute when the animal reacted or stimulus was over
                    if len(self.RT[tt]) == 0:
                        # the monkey did not give a response
                        RT = stt[prechange_stt + 1] + winsize * self.ST_N_timebins + .0001
                    else:
                        # get spike until response
                        RT = stt[prechange_stt + 1] + self.RT[tt][0]

                    times_change =  np.arange(stt_del[prechange_stt+1],RT-.01, winsize)

                    # just in case there are more than expected time windows produced:
                    times_change = times_change[:self.ST_N_timebins]

                    # getstimulus and spikes for changed stimulus
                    # stimlus matrix will have: one dimension for contrast (- or 1) and one optional for direction
                    xchangetmp = np.zeros([len(times_change) - 1, 1 + self.ST_inc_direction]) * np.nan
                    spikchangtmp = np.zeros([len(times_change) - 1, len(neurons)]) * np.nan
                    mt_spikchangtmp = np.zeros([len(times_change) - 1, 1]) * np.nan
                    for ii in range(len(times_change) - 1):
                        # contrast condition
                        xchangetmp[ii, 0] = (contrast[prechange_stt + 1] == ucontr[-1])
                        # direction
                        if self.ST_inc_direction:
                            # since the orientation change is fixed, the orientation at the stimulus before the change determines the stimulus at change orientation
                            xchangetmp[ii, 1] = (direction[prechange_stt] == uorien[0])
                        #### SPIKES ###
                        for nn in range(len(neurons)):
                            tmp = sptt[(sptt[:, 0] == neurons[nn]) & (sptt[:, 1] == 1), :]
                            spikchangtmp[ii, nn] = np.round(
                                (np.sum((tmp[:, 2] >= times_change[ii]) & (tmp[:, 2] <= times_change[ii + 1])) / \
                                 (times_change[ii + 1] - times_change[ii])) * winsize)
                        tmp = sptt[(sptt[:, 0] == 97) & (sptt[:, 1] == 1), :]
                        mt_spikchangtmp[ii,0] = np.round(
                                (np.sum((tmp[:, 2] >= times_change[ii]) & (tmp[:, 2] <= times_change[ii + 1])) / \
                                 (times_change[ii + 1] - times_change[ii])) * winsize)

                    X_change[:(1+self.ST_inc_direction), tt] = xchangetmp[0,:(1+self.ST_inc_direction)]
                    SPIKES_change[:spikchangtmp.shape[0], :, tt] = spikchangtmp
                    MT_SPIKES_change[:mt_spikchangtmp.shape[0], :, tt] = mt_spikchangtmp

                ###############################################################################################
                ############################## before target response #########################################
                ###############################################################################################
                # RESPONSE before change or saccade or end of trial
                for ii in range(len(times)):
                    ### WINDOW ###
                    startwin = times[ii]
                    endwin = startwin + winsize

                    ### STIMULUS ###
                    # 1) was there a stimulus-onset in this specific time window?
                    timdiff = np.abs(stt_del - startwin) # difference between stimulus onsets and our window starting point
                    if any(timdiff < (winsize)): # this means there was a stimulus in our window
                        # is that stimulus in the middle? (or just the stimulus that marked the beginning of the window?)
                        if (stt_del[timdiff < winsize] - startwin) > 0:
                            endwin = stt_del[timdiff < winsize]
                        if endwin <= startwin:
                            print('failure, end smaller than start?')
                            print(tt)
                            break

                    # 2) which stimulus was last flashed
                    if len(np.where(endwin > (stt_del))[0]) == 0:
                        valid[tt] = False
                        print('    warning: time window defined before first stim on, trial=%.0f' %tt)  # no stimulus present yet (this is a time bin before stimulus start)
                    else:
                        whichstimpres = np.where(endwin >= stt_del)[0][-1]
                        X[ii, :, tt] = np.zeros(X.shape[1])
                        # 3) is that stimulus still on?
                        if (endwin <= (stt_del[whichstimpres] + .0001 + winsize * tot_windows)) & (
                                endwin > stt_del[whichstimpres]): # stimulus is still on
                            # how many time steps away did it occur?
                            temp = np.array(np.ceil((endwin - stt_del[whichstimpres] - .0001) / (winsize)), dtype='int') - 1
                            # STIMULUS ENCODING
                            if temp<self.ST_N_timebins: # is the stimulus actually on
                                # contrast condition specific time window
                                if contrast[whichstimpres] == ucontr[0]: # low contrast condition
                                    X[ii,temp,tt] = 1
                                else: # high contrast condition
                                    X[ii, self.ST_N_timebins + temp, tt] = 1
                            else: # or are we in the after stimulus time bin
                                X[ii, self.ST_N_timebins + temp, tt] = 1
                            if self.ST_inc_direction:
                                X[ii, 2*self.ST_N_timebins + self.ST_aft_stim_windows, tt] = direction[whichstimpres] == uorien[0] # preferred direction
                            # time window
                            if self.ST_onoff_stim:
                                X[ii, -2, tt] = 1
                    #### SPIKES ###
                    # take current channel, only first unit so as not to have multiple units from one channel
                    tempsiz = endwin - startwin
                    for nn in range(len(neurons)):
                        tmp = sptt[(sptt[:, 0] == neurons[nn]) & (sptt[:, 1] == 1), :]
                        SPIKES[ii, nn, tt] = np.round((np.sum((tmp[:, 2] >= startwin) & (tmp[:, 2] <= endwin)) / tempsiz)*winsize)
                    tmp = sptt[(sptt[:, 0] == 97) & (sptt[:, 1] == 1), :]
                    MT_SPIKES[ii, 0, tt] = np.round((np.sum((tmp[:, 2] >= startwin) & (tmp[:, 2] <= endwin)) / tempsiz)*winsize)


                X[:len(times), -1, tt] = np.ones(
                    len(times))  # last variable is constant offset (baseline FR) --> always one
        elif self.att==3:
            X_change = np.zeros([2, self.Ttrials]) * np.nan
            # session specific contrast and orientation condition, neuron indices
            ucontr = np.sort(np.unique(self.contrast[0, :]))
            uorien = np.sort(np.unique(self.orientation[0, :]))

            # make empty files to collect spikes and stimulus up to a change in stimulus or a saccade
            SPIKES = np.zeros(
                [max_stim_pres * (tot_windows + 2 * self.ST_N_timebins), len(neurons), self.Ttrials]) * np.nan
            MT_SPIKES = np.zeros([max_stim_pres * (tot_windows + 2 * self.ST_N_timebins), 1, self.Ttrials]) * np.nan

            X = np.zeros([max_stim_pres * (tot_windows + 2 * self.ST_N_timebins),
                          1 + 4 * self.ST_N_timebins + self.ST_aft_stim_windows,
                          self.Ttrials]) * np.nan
            valid = np.array(np.ones(self.Ttrials), dtype='bool')
            for tt in range(self.Ttrials):  # trial
                if np.any(np.isnan(self.stimtimes[tt][:, 0])):
                    valid[tt] = False
                    print('    warning: there are nans in the stimulus times, trial=%.0f' % tt)
                    continue
                # stimulus times and properties of this trial (go until the one before to last stimulus)
                if exclude_first:  # first stimulus presentation will not be anaylzed since there can be adaptation
                    want = np.arange(1, len(self.stimtimes[tt][:, 0]))
                else:
                    want = np.arange(len(self.stimtimes[tt][:, 0]))
                if (len(want) <= 2):
                    valid[tt] = False
                    print('    warning: number of valid stimulus presentations<2, trial=%.0f' % tt)
                    continue
                # stimulus times, stimulus properties (contrast, direction)
                stt = self.stimtimes[tt][:, 0][want]
                stt_del = stt + stim_timedelay
                contrast = self.contrast[:2, self.trial_name[tt] == self.trialind].T[want].T
                direction = self.orientation[2, self.trial_name[tt] == self.trialind][want]
                prechange_stt = np.where(np.diff(direction) > 0)[0][0]  # last stimulus before change occured!!!!!
                stim_change = stt[prechange_stt + 1]  # stimulus change
                # spike times of this trial
                sptt = self.spikes[tt]
                # set time windows from beginning of stimulus times (either first or second presentation) to
                # when a saccade happened OR when the stimulus change happened OR at end of trial
                times = np.zeros(200) * np.nan
                countii = 0
                for ii in range(len(stt) - 1):
                    # are we still before the end point (defined by change happening)
                    if stt_del[ii] <= stim_change:
                        # what ends this window, new stimulus or end point
                        endii = np.min([stt_del[ii + 1], stim_change])
                        # break up this time window into x ms time windows
                        tmp = np.arange(stt_del[ii], endii - .0001, winsize)
                        times[countii: (countii + len(tmp))] = tmp
                        countii += len(tmp)
                times = times[np.isnan(times) == False]
                if len(times) == 0:
                    valid[tt] = False
                    print('    warning: this trial has no valid time points, something is off, trial=%.0f' % tt)
                    continue

                if change_to:
                    # compute when the animal reacted or stimulus was over
                    if len(self.RT[tt]) == 0:
                        # the monkey did not give a response (miss)
                        RT = stt[prechange_stt + 1] + winsize * self.ST_N_timebins + .0001
                    else:
                        # get spike until response
                        RT = stt[prechange_stt + 1] + self.RT[tt][0]
                    times_change =  np.arange(stt_del[prechange_stt+1],RT-.01, winsize)

                    # just in case there are more than expected time windows produced:
                    times_change = times_change[:self.ST_N_timebins]

                    # get stimulus and spikes for changed stimulus
                    # stimlus matrix will have: one dimension for contrast (- or 1) and one optional for direction
                    xchangetmp = np.zeros([len(times_change) - 1, 2]) * np.nan
                    spikchangtmp = np.zeros([len(times_change) - 1, len(neurons)]) * np.nan
                    mt_spikchangtmp = np.zeros([len(times_change) - 1, 1]) * np.nan
                    for ii in range(len(times_change) - 1):
                        # contrast condition
                        xchangetmp[ii, 0] = (contrast[0, prechange_stt + 1] == ucontr[-1])
                        xchangetmp[ii, 1] = (contrast[1, prechange_stt + 1] == ucontr[-1])
                        #### SPIKES ###
                        for nn in range(len(neurons)):
                            tmp = sptt[(sptt[:, 0] == neurons[nn]) & (sptt[:, 1] == 1), :]
                            spikchangtmp[ii, nn] = np.round(
                                (np.sum((tmp[:, 2] >= times_change[ii]) & (tmp[:, 2] <= times_change[ii + 1])) / \
                                 (times_change[ii + 1] - times_change[ii])) * winsize)
                        tmp = sptt[(sptt[:, 0] == 97) & (sptt[:, 1] == 1), :]
                        mt_spikchangtmp[ii,0] = np.round(
                                (np.sum((tmp[:, 2] >= times_change[ii]) & (tmp[:, 2] <= times_change[ii + 1])) / \
                                 (times_change[ii + 1] - times_change[ii])) * winsize)
                    X_change[:(1+self.ST_inc_direction), tt] = xchangetmp[0,:(1+self.ST_inc_direction)]
                    SPIKES_change[:spikchangtmp.shape[0], :, tt] = spikchangtmp
                    MT_SPIKES_change[:mt_spikchangtmp.shape[0], :, tt] = mt_spikchangtmp


                ###############################################################################################
                ############################## before target response #########################################
                ###############################################################################################
                # RESPONSE before change or saccade or end of trial
                for ii in range(len(times)):
                    ### WINDOW ###
                    startwin = times[ii]
                    endwin = startwin + winsize

                    ### STIMULUS ###
                    # 1) was there a stimulus-onset in this specific time window?
                    timdiff = np.abs(
                        stt_del - startwin)  # difference between stimulus onsets and our window starting point
                    if any(timdiff < (winsize)):  # this means there was a stimulus in our window
                        # is that stimulus in the middle? (or just the stimulus that marked the beginning of the window?)
                        if (stt_del[timdiff < winsize] - startwin) > 0:
                            endwin = stt_del[timdiff < winsize]
                        if endwin <= startwin:
                            print('failure, end smaller than start?')
                            print(tt)
                            break

                    # 2) which stimulus was last flashed
                    if len(np.where(endwin > (stt_del))[0]) == 0:
                        valid[tt] = False
                        print(
                            '    warning: time window defined before first stim on, trial=%.0f' % tt)  # no stimulus present yet (this is a time bin before stimulus start)
                    else:
                        whichstimpres = np.where(endwin >= stt_del)[0][-1]
                        X[ii, :, tt] = np.zeros(X.shape[1])
                        # 3) is that stimulus still on?
                        if (endwin <= (stt_del[whichstimpres] + .0001 + winsize * tot_windows)) & (
                                endwin > stt_del[whichstimpres]):  # stimulus is still on
                            # how many time steps away did it occur?
                            temp = np.array(np.ceil((endwin - stt_del[whichstimpres] - .0001) / (winsize)),
                                            dtype='int') - 1
                            # STIMULUS ENCODING
                            if temp < self.ST_N_timebins:  # is the stimulus actually on
                                # contrast condition specific time window
                                if contrast[0, whichstimpres] == ucontr[1]:  # stim 1: low contrast condition
                                    X[ii, temp, tt] = 1
                                elif contrast[0, whichstimpres] == ucontr[2]:  # stim 1: high contrast condition
                                    X[ii, self.ST_N_timebins + temp, tt] = 1
                                if contrast[1, whichstimpres] == ucontr[1]:  # stim 2: low contrast condition
                                    X[ii, 2 * self.ST_N_timebins + temp, tt] = 1
                                elif contrast[1, whichstimpres] == ucontr[2]:  # stim 2: high contrast condition
                                    X[ii, 3 * self.ST_N_timebins + temp, tt] = 1

                            else:  # or are we in the after stimulus time bin
                                X[ii, 3 * self.ST_N_timebins + temp, tt] = 1
                    #### SPIKES ###
                    # take current channel, only first unit! so as not to have multiple units from one channel
                    tempsiz = endwin - startwin
                    for nn in range(len(neurons)):
                        tmp = sptt[(sptt[:, 0] == neurons[nn]) & (sptt[:, 1] == 1), :]
                        SPIKES[ii, nn, tt] = np.round(
                            (np.sum((tmp[:, 2] >= startwin) & (tmp[:, 2] <= endwin)) / tempsiz) * winsize)
                    tmp = sptt[(sptt[:, 0] == 97) & (sptt[:, 1] == 1), :]
                    MT_SPIKES[ii, 0, tt] = np.round(
                        (np.sum((tmp[:, 2] >= startwin) & (tmp[:, 2] <= endwin)) / tempsiz) * winsize)

                X[:len(times), -1, tt] = np.ones(
                    len(times))  # last variable is constant offset (baseline FR) --> always one
            tmp = np.mean(np.isnan(X[:, 0, :]), axis=1) < 1

        tmp = np.mean(np.isnan(X[:, 0, :]), axis=1) < 1

        self.data_trial = SPIKES[tmp, :, :]
        self.MT_data_trial = MT_SPIKES[tmp,:,:]
        if change_to:
            self.data_trial_change = SPIKES_change[np.sum(np.isnan(SPIKES_change[:, 0, :]) == False, axis=1) > 0, :, :]
            self.MT_data_trial_change = MT_SPIKES_change[np.sum(np.isnan(MT_SPIKES_change[:, 0, :]) == False, axis=1) > 0, :, :]
        self.counts0 = np.sum(np.isnan(self.data_trial[:, 0, :]) == False, axis=0)
        self.D = self.data_trial.shape[1]
        print('total time points (over all trials): ', np.sum(self.counts0))
        print('mean per trial: ', np.round(np.mean(self.counts0), 2))

        self.X = X[tmp, :, :]
        # create variable dictionary
        if self.att<3:
            Xvar = np.array(['contrast 1 stim_win1'])
            for ii in range(1, self.ST_N_timebins):
                Xvar = np.concatenate((Xvar, np.array(['contrast 1 stim_win' + np.str(ii + 1)])))
            #for ii in range(self.ST_aft_stim_windows):
            #    Xvar = np.concatenate((Xvar, np.array(['contrast 1 aftstim_win' + np.str(ii + 1)])))
            for ii in range(self.ST_N_timebins):
                Xvar = np.concatenate((Xvar, np.array(['contrast 2 stim_win' + np.str(ii + 1)])))
            for ii in range(self.ST_aft_stim_windows):
                Xvar = np.concatenate((Xvar, np.array(['aftstim_win' + np.str(ii + 1)])))
            if self.ST_inc_direction:
                Xvar = np.concatenate((Xvar, np.array(['direction'])))
            if self.ST_onoff_stim:
                Xvar = np.concatenate((Xvar, np.array(['stim_onoff'])))
            self.Xvar = list(np.concatenate((Xvar, np.array(['offset']))))
        else:
            Xvar = np.array(['stim 1 contrast 1 stim_win1'])
            for ii in range(1, self.ST_N_timebins):
                Xvar = np.concatenate((Xvar, np.array(['stim 1 contrast 1 stim_win' + np.str(ii + 1)])))
            for ii in range(self.ST_N_timebins):
                Xvar = np.concatenate((Xvar, np.array(['stim 1 contrast 2 stim_win' + np.str(ii + 1)])))
            for ii in range(1, self.ST_N_timebins):
                Xvar = np.concatenate((Xvar, np.array(['stim 2 contrast 1 stim_win' + np.str(ii + 1)])))
            for ii in range(self.ST_N_timebins):
                Xvar = np.concatenate((Xvar, np.array(['stim 2 contrast 2 stim_win' + np.str(ii + 1)])))

            for ii in range(self.ST_aft_stim_windows):
                Xvar = np.concatenate((Xvar, np.array(['aftstim_win' + np.str(ii + 1)])))


            self.Xvar = list(np.concatenate((Xvar, np.array(['offset']))))


        # when stimulus changes
        if change_to:
            self.X_change = X_change
        self.valid_trials = valid

        valid_num = self.trial_name[valid]

        # only take valid trials:
        self.attention = self.attention[valid]
        self.RT = self.RT[valid]
        # self.sacc = self.sacc[valid]
        self.iscatch = self.iscatch[valid]
        self.isdistr = self.isdistr[valid]
        self.out = self.out[valid]
        self.Ttrials = np.sum(valid)
        self.trial_name = self.trial_name[valid]
        self.data_trial = self.data_trial[:, :, valid]
        self.MT_data_trial = self.MT_data_trial[:,:,valid]
        self.counts0 = self.counts0[valid]
        self.X = self.X[:, :, valid]
        self.microsaccyn = self.microsaccyn[valid]
        self.saccyn = self.saccyn[valid]
        mask = np.zeros(len(self.trialind), dtype='bool')
        for tt in valid_num:
            mask[self.trialind == tt] = True
        self.orientation = self.orientation[:, mask]
        self.contrast = self.contrast[:, mask]
        self.presind = self.presind[mask]
        self.trialind = self.trialind[mask]

        if change_to:
            self.X_change = self.X_change[:,valid]
            self.data_trial_change = self.data_trial_change[:,:,valid]
            self.MT_data_trial_change = self.MT_data_trial_change[:, :, valid]

        #### compute BASELINE #########

        self.baseline = np.zeros([np.sum(valid), 4, len(neurons)]) * np.nan
        for nnii in range(len(neurons)):
            nn = neurons[nnii]

            #plt.figure(figsize=(15, 10))
            for tt in range(self.Ttrials):
                if valid[tt]==False:
                    continue
                spikestmp = self.spikes[tt][self.spikes[tt][:, 1] == 1, :]
                ind = spikestmp[:, 0] == nn
                stimtim = np.round(self.stimtimes[tt][0], 3)

                wind = np.arange(stimtim - .2, stimtim + .04, .05)
                self.baseline[tt, :, nnii] = np.histogram(spikestmp[ind, 2], wind)[0]


    def ISI_test(self, percpop = 8, cut=3):
        neurons = np.unique(
            self.spikes[0][:, 0])  # assuming that the units don't shift between channels and are stable within a sess
        neurons = neurons[neurons != 97]  # ch 97 sort 1 is always the MT unit

        SPIKES = np.zeros([100000, len(neurons)]) * np.nan

        for nncount in range(len(neurons)):
            nn = neurons[nncount]
            countspik = 0
            for tt in range(self.Ttrials):
                spik = self.spikes[tt][(self.spikes[tt][:, 0] == nn) & (self.spikes[tt][:, 1] == 1), 2]
                SPIKES[countspik:(countspik + len(spik)), nncount] = spik
                countspik += len(spik)

        bins = np.array([-100, 0, .001, cut / 1000, .01, .1, 1, 1000])
        ISIperc = np.zeros(len(neurons)) * np.nan
        for nncount in range(len(neurons)):
            hist = np.histogram(np.diff(SPIKES[:-(np.sum(np.isnan(SPIKES[:, nncount]))), nncount]), bins=bins)
            h1 = np.array(hist[0], dtype='float')
            h1[bins[:-1] < 0] = np.nan
            h1 = hist[0] / np.nanmax(hist[0]) * 100
            ISIperc[nncount] = h1[2] + h1[1]
        return np.where(ISIperc > percpop)[0]

    def simulate_data(self, D_fake=None, scalcoefGT=None, constGT=None, mod_sigGT=None, mod_dimGT = None,
                      plotit=False, times=None, AGT=0, seed=None, temporal_response=True,
                      multiunits=0, multi_tuning='rand', multi_mod='rand'):
        # simulate spiking with ground truth stimulus coefficients matched for this specific dataset
        # fake data set:
        self.AGT = AGT
        if times is not None:
            self.Ttrials = np.array(times*self.Ttrials, dtype='int')
            counts0tmp = np.zeros(len(self.counts0)*times, dtype='int')
            Xtmp = np.zeros([self.X.shape[0], self.X.shape[1], self.X.shape[2]*times])*np.nan
            for ii in range(times):
                for tt in range(self.X.shape[2]):
                    Xtmp[:,:,tt+ii*self.X.shape[2]] = self.X[:,:,tt]
                    counts0tmp[tt+ii*self.X.shape[2]] = self.counts0[tt]
            self.X = np.copy(Xtmp)
            self.counts0 = np.copy(counts0tmp)
        self.simulate = True
        if (D_fake is None)|(multiunits>0):
            D_single = np.copy(self.D)
            if (multiunits>0):
                D_fake = np.array(np.floor(self.D/multiunits), dtype='int')
                self.pairs = np.random.choice(self.D, D_fake*multiunits,
                                              replace=False).reshape(D_fake,multiunits)
            else:
                D_fake = np.copy(self.D)
        INF = np.array(D_single / 2, dtype='int')
        if self.betaGT is None:
            self.betaGT = scalcoefGT * np.random.randn(D_single * self.X.shape[1]).reshape(self.X.shape[1], D_single)
            if temporal_response==False:
                self.betaGT[:4,:] = self.betaGT[0,:]
                self.betaGT[4:8, :] = self.betaGT[4, :]

            self.betaGT[:, INF:] = 0 # make untuned neurons
            self.betaGT[-1, :] += constGT  # constant offset
            # to make up for the expected increase in firing due to stimulus coefficient in informative neurons
            self.betaGT[-1,INF:] +=scalcoefGT**2/2
            if multi_tuning=='same':
                self.betaGT[:,self.pairs[:,1]] = np.copy(self.betaGT[:,self.pairs[:,0]])
        if self.mGT is None:
            sim_mod = True
            self.mGT = np.zeros([np.max(self.counts0), mod_dimGT, self.Ttrials])
            if len(self.AGT)<mod_dimGT:
                self.AGT = np.eye(mod_dimGT) * self.AGT
            print('get random modulator')
        else: sim_mod=False
        if self.wGT is None:
            self.wGT = np.random.rand(mod_dimGT * D_single).reshape([mod_dimGT, D_single])#*2-1
            if multi_mod=='same':
                self.wGT[0,self.pairs[:,1]] = np.copy(self.wGT[0,self.pairs[:,0]])

        SPIKES_fake = np.zeros([self.data_trial.shape[0], D_fake, self.X.shape[2]]) * np.nan
        if multiunits > 0:
            self.SPIKES_fake_single  = np.zeros([self.data_trial.shape[0], D_single, self.X.shape[2]]) * np.nan
        if seed is not None: np.random.seed(seed)
        for tt in range(self.Ttrials):
            if sim_mod:
                self.mGT[0, :, tt] = np.random.randn(mod_dimGT) * mod_sigGT/10
                for ss in range(1, self.counts0[tt]):
                    self.mGT[ss, :, tt] = self.mGT[ss - 1, :, tt].dot(self.AGT) + np.random.randn(mod_dimGT) * mod_sigGT
                self.mGT[self.counts0[tt]:, :, tt] = np.nan
            tmprate = np.exp(self.X[:self.counts0[tt], :, tt].dot(self.betaGT) +
                            self.mGT[:self.counts0[tt], :, tt].dot(self.wGT))
            if any(tmprate.ravel()>1000):
                wherehigh = np.nanargmax(tmprate.ravel())
                print('firing rate too high for Poisson process (max=', np.max(tmprate), ")")
                tmpmod = self.mGT[:self.counts0[tt], :, tt]
                print(tmpmod[wherehigh])
            if multiunits>0:
                self.SPIKES_fake_single[:self.counts0[tt], :, tt] = \
                    np.random.poisson(tmprate)
                for ss in range(self.counts0[tt]):
                    SPIKES_fake[ss, :, tt] = \
                        np.sum(self.SPIKES_fake_single[ss, self.pairs, tt],axis=1)
            else:
                SPIKES_fake[:self.counts0[tt], :, tt] = \
                    np.random.poisson(tmprate)
        self.data_trial = SPIKES_fake
        self.D = D_fake
        self.mod_sigGT = mod_sigGT
        self.mod_dimGT = mod_dimGT
        self.mfr = np.mean(np.nanmean(self.data_trial, axis=0), axis=1)
        if plotit:

            if multiunits==0:
                fig2, ax2 = plt.subplots(1, 2, figsize=(17, 4))
                _, YTMP = concat_trials(SPIKES_fake[:, :INF, :], self.counts0, self.X)
                ax2[0].boxplot(YTMP)
                _, YTMP = concat_trials(SPIKES_fake[:, INF:, :], self.counts0, self.X)
                ax2[1].boxplot(YTMP)
                ax2[0].set_title('FR distribution of stim-tuned')
                ax2[1].set_title('FR distribution of untuned')
                ax2[0].set_xlabel('simulated neurons')
                ax2[0].set_ylabel('FR')
            else:
                fig2, ax2 = plt.subplots(1, 1, figsize=(17, 3))
                _, YTMP = concat_trials(SPIKES_fake, self.counts0, self.X)
                ax2.boxplot(YTMP)
                ax2.set_title('FR distribution of multiunits')
                ax2.set_xlabel('simulated neurons')
                ax2.set_ylabel('FR')


    def comp_informative(self):

        # only use the correct outcome trials:
        data_trial = self.data_trial[:,:,self.out==1]
        X = self.X[:,:,self.out==1]
        data_trial_change = self.data_trial_change[:,:,self.out==1]
        X_change = self.X_change[:,self.out==1]
        counts0 = self.counts0[self.out==1]

        if self.att < 3:
            highcontr = np.where(X_change[0, :] == 1)[0] # high contr trials
            lowcontr = np.where(X_change[0, :] == 0)[0] # low contr trials
        else:
            highcontr = np.arange(X_change.shape[1])
            lowcontr = np.arange(X_change.shape[1])
        if len(highcontr)>0:
            #######################################################################
            ############################## FLD ####################################
            #######################################################################
            print("compute FLD informativeness")
            print('FLD deactivated!!!')
            self.FLD = None
            self.THRESH = None
            self.DEC = None
            xtmp, ytmp = concat_trials(data_trial, counts0, X)

            #######################################################################
            ############################## d' #####################################
            #######################################################################
            print("compute d' informativeness")
            # get time points where there was a high contrast stimulus:
            # for first time window
            if self.att<3:
                indtt = np.array([ self.Xvar.index('contrast 2 stim_win' + np.str(1))])  # only take high contrast trials
                # for all other time windows
                for tt in range(1, self.ST_N_timebins):
                    indtt = np.concatenate((indtt, np.array([self.Xvar.index('contrast 2 stim_win' + np.str(tt + 1))])))
                xind = np.nansum(xtmp[:, indtt], axis=1) == 1
            else:
                xind = np.nansum(xtmp[:,:16], axis=1) >= 1
            # take the mean of those time points
            tmp = ytmp[xind, :]
            mu_resp = np.mean(tmp, axis=0)
            var_resp = np.var(tmp, axis=0)

            _, change = concat_trials(data_trial_change[:, :, highcontr],
                                      np.sum(np.isnan(data_trial_change[:, 0, highcontr]) == False, axis=0),
                                      data_trial_change[:, :, highcontr])
            mu_change = np.mean(change, axis=0)
            var_change = np.var(change, axis=0)
            self.dprim = (mu_resp - mu_change) / np.sqrt(.5 * (var_resp + var_change))
            # test for significance
            p = np.zeros([self.D]) * np.nan
            for nn in range(self.D):
                _, p[nn] = stats.ttest_ind(tmp[:,nn], change[:,nn])
            p[var_change==0] = 1
            self.dprim_p = p

            # compute also for low-contrast case
            if (len(lowcontr)>0) &(self.att<3):
                indtt_low = np.array([self.Xvar.index('contrast 1 stim_win' + np.str(1))])  # only take high contrast trials
                # for all other time windows
                for tt in range(1, self.ST_N_timebins):
                    indtt_low  = np.concatenate((indtt_low , np.array([self.Xvar.index('contrast 1 stim_win' + np.str(tt + 1))])))
                xind_low = np.nansum(xtmp[:, indtt_low ], axis=1) == 1
                # take the mean of those time points
                tmp_low = ytmp[xind_low, :]
                mu_resp_low = np.mean(tmp_low, axis=0)
                var_resp_low = np.var(tmp_low, axis=0)

                _, change_low = concat_trials(data_trial_change[:, :, lowcontr],
                                          np.sum(np.isnan(data_trial_change[:, 0, lowcontr]) == False, axis=0),
                                          data_trial_change[:, :, lowcontr])
                mu_change_low = np.mean(change_low, axis=0)
                var_change_low = np.var(change_low, axis=0)
                self.dprim_low = (mu_resp_low - mu_change_low) / np.sqrt(.5 * (var_resp_low + var_change_low))
                # test for significance
                p_low = np.zeros([self.D]) * np.nan
                for nn in range(self.D):
                    _, p_low[nn] = stats.ttest_ind(tmp_low[:,nn], change_low[:,nn])
                p_low[var_change==0] = 1
                self.dprim_p_low = p_low
            else:
                self.dprim_low = None
                self.dprim_p_low = None


            ############################## choice probs ###########################
            if np.sum(self.out == 4)>0:
                print("compute choice probabilities")
                resp_nosacc = self.data_trial_change[:, :, self.out == 4]
                resp_sacc = self.data_trial_change[:, :, self.out == 1]
                meandiff = np.nanmean(np.nanmean(resp_nosacc, axis=0), axis=1)-np.nanmean(np.nanmean(resp_sacc, axis=0), axis=1)
                varmean = np.sqrt(.5*(np.nanvar(np.nanmean(resp_nosacc, axis=0), axis=1)+np.nanvar(np.nanmean(resp_sacc, axis=0), axis=1)))
                self.choice_probs = meandiff/varmean
            else:
                self.choice_probs = None

    def stationary_check(self):
        stimpres = np.zeros([self.Ttrials]) * np.nan
        for tt in range(self.Ttrials):
            stimpres[tt] = len(self.stimtimes[tt]) - 1

        tot_win = self.ST_N_timebins + self.ST_aft_stim_windows
        MA = np.zeros([1000, 2, self.D]) * np.nan

        count = 0
        for tt in range(self.Ttrials):
            # get the stim presentation indizes
            ind = self.X[:, self.Xvar.index('stim_onoff'), tt] == 1
            datt = self.data_trial[ind, :, tt]
            ind = np.repeat(np.arange(stimpres[tt]), tot_win)
            for ss in range(np.array(stimpres[tt], dtype='int')):
                MA[count, 0, :] = np.mean(datt[ind == ss, :], axis=0)
                MA[count, 1, :] = self.stimtimes[tt][ss]
                count += 1
        MA = MA[:count, :, :]
        linreg = np.zeros([self.D, 2]) * np.nan
        spear = np.zeros([self.D, 2]) * np.nan
        for nn in range(self.D):
            lr = linregress(MA[:, 0, nn], MA[:, 1, nn])
            linreg[nn, :] = [lr.slope, lr.pvalue]
            spear[nn, :] = spearmanr(MA[:, 0, nn], MA[:, 1, nn])

        adapt = np.zeros([self.D, 2])
        adapt[:, 0] = (linreg[:, 0] < 0) & (linreg[:, 1] < .05)
        adapt[:, 1] = (spear[:, 0] < 0) & (spear[:, 1] < .05)
        self.adaptation_data = MA
        self.adaptation_linreg = linreg
        self.adaptation_spearman = spear
        print('% neurons with a significant negative (adaptation) slope : ', np.mean(adapt[:, 0]))
        print('% neurons with a significant negative (adaptation) corr : ', np.mean(adapt[:, 1]))

    def visualize_adaptation(self):
        fig, ax = plt.subplots(1, 3, figsize=(18, 4))
        neurons = np.arange(self.D)
        ax[0].plot(neurons[self.adaptation_linreg[:, 0] > 0], self.adaptation_linreg[self.adaptation_linreg[:, 0] > 0, 1], 'or', label='pos slope')
        ax[0].plot(neurons[self.adaptation_linreg[:, 0] < 0], self.adaptation_linreg[self.adaptation_linreg[:, 0] < 0, 1], 'ob', label='neg slope')
        ax[0].plot([0, self.D], [.05, .05], '-k')
        ax[0].legend()
        ax[0].set_xlabel('neurons')
        ax[0].set_ylabel('slope p')

        ax[1].plot(neurons[self.adaptation_spearman[:, 0] > 0], self.adaptation_spearman[self.adaptation_spearman[:, 0] > 0, 1], 'or', label='pos slope')
        ax[1].plot(neurons[self.adaptation_spearman[:, 0] < 0], self.adaptation_spearman[self.adaptation_spearman[:, 0] < 0, 1], 'ob', label='neg slope')
        ax[1].plot([0, self.D], [.05, .05], '-k')
        ax[1].legend()
        ax[1].set_xlabel('neurons')
        ax[1].set_ylabel('separman p')

        ax[2].plot(self.adaptation_linreg[self.adaptation_linreg[:, 0] > 0, 1], self.adaptation_spearman[self.adaptation_linreg[:, 0] > 0, 1], 'or', label='pos lr slope')
        ax[2].plot(self.adaptation_linreg[self.adaptation_linreg[:, 0] < 0, 1], self.adaptation_spearman[self.adaptation_linreg[:, 0] < 0, 1], 'ob', label='neg lr slope')
        ax[2].legend()
        ax[2].set_xlabel('lr p')
        ax[2].set_ylabel('spearman p')

        # EXAMPLES
        bins = np.arange(0, np.max(self.adaptation_data[:, 1, :]) + .6, .5)

        poslope = np.where(self.adaptation_linreg[:, 0] > 0)[0]
        nn1 = poslope[np.argmin(self.adaptation_linreg[poslope, 1])]
        print('strong positive slope example (in pop '+np.str(np.round(np.mean((self.adaptation_linreg[:,0]>0)&
                                                                               (self.adaptation_linreg[:,1]<.05))*100,1))+
              ' percent), neuron ' + np.str(nn1) + (
                    ', p(slope) %.2f' % (self.adaptation_linreg[nn1, 1])) + (
                          ', p(spearman=) %.2f' % self.adaptation_spearman[nn1, 1]))
        print('  lin slope: %.2f' % (self.adaptation_linreg[nn1, 0]))
        print('  spearman r: %.2f' % (self.adaptation_spearman[nn1, 0]))
        neslope = np.where(self.adaptation_linreg[:, 0] < 0)[0]
        nn2 = neslope[np.argmin(self.adaptation_linreg[neslope, 1])]
        print('strong negative slope example (in pop '+np.str(np.round(np.mean((self.adaptation_linreg[:,0]<0)&
                                                                               (self.adaptation_linreg[:,1]<.05))*100,1))+
              ' percent), neuron ' + np.str(nn2) + (
                    ', p(slope) %.2f' % self.adaptation_linreg[nn2, 1]) + (
                          ', p(spearman=) %.2f' % self.adaptation_spearman[nn2, 1]))
        print('  lin slope: %.2f' % (self.adaptation_linreg[nn2, 0]))
        print('  spearman r: %.2f' % (self.adaptation_spearman[nn2, 0]))
        nn3 = np.argmax(self.adaptation_linreg[:, 1])
        print('weak slope example, neuron ' + np.str(nn3) + (', p(slope=) %.2f' % self.adaptation_linreg[nn3, 1]) + (
                    ', p(spearman=) %.2f' % self.adaptation_spearman[nn3, 1]))
        print('  lin slope: %.2f' % (self.adaptation_linreg[nn3, 0]))
        print('  spearman r: %.2f' % (self.adaptation_spearman[nn3, 0]))
        NN = np.array([nn1, nn2, nn3])
        fig2, ax2 = plt.subplots(1, 3, figsize=(18, 2))
        fig, ax = plt.subplots(1, 3, figsize=(18, 5))
        for ii in range(len(NN)):
            nn = NN[ii]
            _ = ax2[ii].hist(self.adaptation_data[:, 1, nn], bins=bins)
            ax2[ii].set_xlabel('stimulus times')
            ax2[ii].set_ylabel('frequency')
            means = np.zeros([len(bins) - 1, self.D]) * np.nan
            means = np.zeros([len(bins) - 1, self.D]) * np.nan
            FR_tim = pd.DataFrame(self.adaptation_data[:, :, nn], columns=['FR', 'stim_tim'])
            FR_tim.stim_tim = pd.cut(self.adaptation_data[:, 1, nn], bins=bins, labels=bins[:-1])
            sb.lineplot(x='stim_tim', y='FR', ci='sd', data=FR_tim, ax=ax[ii])
            ax[ii].plot(self.adaptation_data[:, 1, nn], self.adaptation_data[:, 0, nn], 'ob')
            ax2[ii].set_title('neuron ' + np.str(nn))

class helperarg_PLDSfitting():
    def par(self, data, DtM, printit=True, fig=None, norm=False):
        self.data=data
        self.DtM = DtM
        self.printit = printit
        self.fig=fig
        self.norm = norm

class DATAtoMODEL:


    def par(self, data, MINDIM, MAXDIM, Ncrossval, path, saveresults=False, seed=0,
            scalQ=.01, scalQ0=.001, maxiter=10, maxtim=1000, difflikthresh=.01,
            upx0=True, upQ0=True, upQ=True, upA=True, upC=True, upB=True, Adiag=False, regA=True,
            backtracking=True, backtrack_diff=100, seedtest=0, residuals=True,
            estA=None, estQ=None, estQ0=None, estx0=None, estB = None, estC = None,
            PLDS_lik = None, SR_lik = None, dim=None, versionadd=None,
            ):

        self.scalQ=scalQ
        self.scalQ0=scalQ0
        self.maxiter=maxiter
        self.maxtim = maxtim
        self.difflikthresh = difflikthresh
        self.upx0 = upx0
        self.upQ=upQ
        self.upA = upA
        self.upQ0 = upQ0
        self.upC = upC
        self.upB = upB
        self.Adiag = Adiag
        self.regA=regA
        self.backtracking = backtracking
        self.backtrack_diff = backtrack_diff
        self.estA = estA
        self.estQ = estQ
        self.estQ0 = estQ0
        self.estx0 = estx0
        self.estC = estC
        self.estB = estB
        self.seedtest = seedtest
        self.residuals = residuals
        self.EMiter = np.zeros([Ncrossval, MAXDIM-MINDIM+1])*np.nan

        self.MINDIM=MINDIM
        self.MAXDIM=MAXDIM
        self.Ncrossval=Ncrossval
        self.path=path
        self.seed=seed
        if versionadd is None:
            self.name = 'day_' + np.str(data.day) + '_attention_' + np.str(data.att) + '_block_' + np.str(
                data.block)
        else:
            self.name = 'day_' + np.str(data.day) + '_attention_' + np.str(data.att) + '_block_' + np.str(data.block) + np.str(versionadd)
        self.dim = dim

        self.list_ee_xx = None

        if PLDS_lik is None:
            self.PLDS_lik = np.zeros([self.Ncrossval, data.D, self.MAXDIM - self.MINDIM + 1]) * np.nan
        else:
            self.PLDS_lik = PLDS_lik
        if SR_lik is None:
            self.SR_lik = np.zeros([self.Ncrossval, data.D]) * np.nan
        else:
            self.SR_lik = SR_lik
        np.random.seed(seed)
        Ntrain = np.array([np.round(data.Ttrials * .9)], dtype=int)[0]

        self.PLDS_MSE = [None] * (self.MAXDIM-self.MINDIM + 1)
        self.PLDS_pred = [None] * (self.MAXDIM-self.MINDIM + 1)

        self.SR_MSE = np.zeros([data.D, Ncrossval]) * np.nan
        self.SR_MSE_const = np.zeros([data.D, Ncrossval]) * np.nan
        self.SR_pred = np.zeros([np.max(data.counts0), data.D, data.Ttrials, self.Ncrossval]) * np.nan;
        self.RESIDUALS = np.zeros([np.sum(data.counts0), data.D, Ncrossval])*np.nan
        self.TRAINTRIALS = np.zeros([Ntrain, Ncrossval], dtype=int)
        self.TESTTRIALS = np.zeros([data.Ttrials - Ntrain, Ncrossval], dtype=int)
        for nncross in range(Ncrossval):
            traintrials_tmp = np.sort(np.random.choice(data.Ttrials, Ntrain, replace=False))
            mask = np.ones(data.Ttrials, dtype=bool)
            mask[traintrials_tmp] = False
            testtrials_tmp = np.arange(data.Ttrials)[mask]

            self.TRAINTRIALS[:, nncross] = np.array([traintrials_tmp], dtype=int)[0]
            self.TESTTRIALS[:, nncross] = np.array([testtrials_tmp], dtype=int)[0]
        if saveresults:
            try:
                os.mkdir(self.path)
            except OSError:
                print ("already existing (overwriting) %s" % self.path)
            else:
                print ("created the directory %s " % self.path)
            print('class DATAtoMODEL saving itself under ', self.path + 'DtM_'+self.name + '.pk')
            pickle.dump(self, open(self.path + 'DtM_'+self.name + '.pk', 'wb'))

    def PLDS_model(self, data, printit=True,
                   fig=None, norm=False, saveresults=False, multiproc=False, num_proc=None):
        if multiproc:
            # list all processes
            if self.list_ee_xx is None:
                args = helperarg_PLDSfitting()
                args.par(data, DtM=self, printit=printit, fig=fig, norm=norm)
                self.list_ee_xx = define_list_preproc(self, args)
            mob = mp.Pool(processes=num_proc)
            print('ready to multiprocess with %i processors' %num_proc)
            start = time.time()
            mob.starmap(helper_PLDSfitting, self.list_ee_xx)
            mob.close()
            end = time.time()
            print('****** total time for PLDS fitting: ', (end - start), '******')

        else:
            start = time.time()
            for ee in range(self.Ncrossval):
                print('######################### iteration '+np.str(ee)+'#########################################')
                for mm in range(self.MINDIM, self.MAXDIM+1):
                    print('###################### fit ', np.str(mm)+' dimension ######################################')
                    ### train it ###
                    _, _, _, _, _ , emiter = model_train(data=data, DtM=self, xdim=mm, ee=ee, printit=printit, fig=fig,
                                                       saveresults=saveresults, norm=norm)
                    print('emiter', emiter)
                    self.EMiter[ee, mm - self.MINDIM] = emiter
            end = time.time()
            print('****** total time for PLDS fitting (min): ', (end-start)/60, '******')
        if saveresults:
            print('PLDS model saving itself under ', self.path + 'DtM_'+self.name + '.pk')
            pickle.dump(self, open(self.path + 'DtM_'+self.name + '.pk', 'wb')) # maybe not necessary?

    def PLDS_computeMSE(self, data, saveresults, printit=False, XDIM=None, nfold=None, multiproc=False, num_proc=None):
        if XDIM is None:
            XDIM = np.arange(self.MINDIM, self.MAXDIM + 1)
        if nfold is None:
            nfold = np.arange(self.Ncrossval)
        if multiproc:
            for xx in range(len(XDIM)):
                xdim = XDIM[xx]
                self.PLDS_pred[xdim - self.MINDIM] = np.zeros([np.max(data.counts0), data.D, data.Ttrials, self.Ncrossval]) * np.nan;
                self.PLDS_MSE[xdim - self.MINDIM] = np.zeros([data.D, xdim, self.Ncrossval])*np.nan;
            mob = mp.Pool(processes=num_proc)
            print('ready to multiprocess with %i processors' %num_proc)
            print(self.list_ee_xx)
            start = time.time()
            mob.starmap(helper_MSE, self.list_ee_xx)
            mob.close()
            end = time.time()
            print('****** total time for MSE/PRED computation (min): ', (end-start)/60, '******')
            args = self.list_ee_xx[0][2]
            for xx in range(len(XDIM)):
                xdim = XDIM[xx]
                for eii in range(len(nfold)):
                    ee = nfold[eii]
                    mse = np.load(args.DtM.path + 'mp_MSE_model_'+np.str(xdim)+'_cf_'+np.str(ee)+'_'+args.DtM.name + '.npy')
                    pred = np.load(args.DtM.path + 'mp_PRED_model_'+np.str(xdim)+'_cf_'+np.str(ee)+'_'+args.DtM.name + '.npy')
                    self.PLDS_pred[xdim - self.MINDIM][:, :, :, ee] = np.copy(pred)
                    if xdim==1:
                        self.PLDS_MSE[xdim - self.MINDIM][:, 0, ee] = np.copy(mse)
    
                    else:
                        self.PLDS_MSE[xdim-self.MINDIM][:, :, ee] = mse[:,:,1] # save the cumulative error for adding each dimension (first dimension is the estimated neuron, second dimension is the dimension
            if saveresults:
                print('PLDS model saving itself under ', self.path + 'DtM_' + self.name + '.pk')
                pickle.dump(self, open(self.path + 'DtM_'+self.name + '.pk', 'wb'))
            
            
        else:
            for xx in range(len(XDIM)):
                xdim = XDIM[xx]
                self.PLDS_pred[xdim - self.MINDIM] = np.zeros([np.max(data.counts0), data.D, data.Ttrials, self.Ncrossval]) * np.nan;
                self.PLDS_MSE[xdim - self.MINDIM] = np.zeros([data.D, xdim, self.Ncrossval])*np.nan;
                for eii in range(len(nfold)):
                    ee = nfold[eii]
                    print('cross-fold '+np.str(ee) + ', ' +np.str(xdim) + ' dimension(s)')
                    MOD = pickle.load(open(self.path + self.name + '_PLDS_ncross_' + np.str(ee) + '_xdim_' + np.str(xdim) + '.pk', 'rb'))
                    MODall = fit_to_all_trials(data_trial=data.data_trial, MOD=MOD, counts0=data.counts0, X=data.X,
                                               seedtest=self.seedtest)
                    estCdeg, Cdeg_arot, estxdeg, estx0deg, estAdeg, xdeg_arot, As, AvT, Au, cho_est, evecest = \
                        PLDS_rotations(MODall, scal=1, plotit=False, printit=printit)
                    # model fit with A-rotated dimensions
                    mse, pred = model_test_lno(MODall=MODall, testtrials=self.TESTTRIALS[:, ee], seedtest=1, data_trial=data.data_trial,
                                       X=data.X, counts0=data.counts0,
                                       rotate=(MODall.xdim > 1), cho_est=cho_est, evecest=evecest,
                                       As=As, AvT=AvT, Au=Au, saveresults=False,
                                       path=self.path, name=self.name, ee=ee, pred =True)
                    self.PLDS_pred[xdim - self.MINDIM][:, :, :, ee] = np.copy(pred)
                    if xdim==1:
                        self.PLDS_MSE[xdim - self.MINDIM][:, 0, ee] = np.copy(mse)
    
                    else:
                        self.PLDS_MSE[xdim-self.MINDIM][:, :, ee] = mse[:,:,1] # save the cumulative error for adding each dimension (first dimension is the estimated neuron, second dimension is the dimension
                    if saveresults:
                        print('PLDS model saving itself under ', self.path + 'DtM_' + self.name + '.pk')
                        pickle.dump(self, open(self.path + 'DtM_'+self.name + '.pk', 'wb'))

    def PLDS_likelihood(self, data):
        # PLDS predictive firing rate
        for ee in range(self.Ncrossval):
            for xdim in range(self.MINDIM, self.MAXDIM+1):
                MODtrain = pickle.load(
                    open(self.path + self.name + '_PLDS_ncross_' + np.str(ee) + '_xdim_' + np.str(xdim) + '.pk', 'rb'))

                #### test it on held out dataset ####
                MOD_test = PLDS()
                MOD_test.par(xdim=MODtrain.xdim, ydim=data.D, seed=self.seedtest, n_step=data.counts0,
                             est=True,
                             y=data.data_trial, Ttrials=data.Ttrials,
                             C=MODtrain.estC, Q0=MODtrain.estQ0, A=MODtrain.estA, Q=MODtrain.estQ, x0=MODtrain.estx0,
                             B=MODtrain.estB, X=data.X)

                # estimate latent for testing trials
                MOD_test.estx, _ = MOD_test.Estep(C_est=False, estA=False, estQ=False, estQ0=False, B_est=False,
                                                  estx0=False)
                logFR = logFR_predict(data, MOD_test.B.T, latent=MOD_test.estx, C=MOD_test.C.T)
                self.PLDS_lik[ee, :, xdim - self.MINDIM] = loglik_pred(data, logFR, trials=self.TESTTRIALS[:, ee])

class runDatatoModel:
    def par(self, DATAFILE, path_save, name, data = None, DtM = None, pred=None, simulate=False, notes=None):
        # DATAfile indicates day, attentional condition, block
        self.DATAFILE=DATAFILE
        self.data = data
        self.DtM = DtM
        self.pred = pred
        self.simulate = simulate
        self.path_save = path_save
        self.name = name
        self.notes = notes

    def load_data(self, count, coFR, remov, path_get,
                  ST_N_timebins, ST_aft_stim_windows=1, ST_onoff_stim=True, ST_inc_direction=True, isi_crit=True,
                  betaGT=None, mGT=None, wGT=None, AGT=None,
                  D_fake=None, constGT=None, scalcoefGT=None, mod_dimGT=None, mod_sigGT=None, inform=None, GT_change=None,
                  behavior_kickout=True, change_to=True, check_adapt=True, code_kickout=None, times=None):

        if self.simulate:
            path_save = self.path_save+'simulation/'
            name = 'data_'+ self.name + '_dim_'+np.str(mod_dimGT)
        else:
            path_save = self.path_save + 'data/'
            name = 'data_'+ self.name

        self.data = DATA()
        self.data.par(coFR=coFR, remov=remov, name = name, path_get=path_get, path_save=path_save,
                      ST_N_timebins= ST_N_timebins, ST_aft_stim_windows=ST_aft_stim_windows, ST_onoff_stim=ST_onoff_stim,
                      ST_inc_direction=ST_inc_direction,
                      isi_crit=isi_crit,
                      betaGT=betaGT, mGT=mGT, wGT=wGT, AGT=AGT)

        self.data.preprocess(count, self.DATAFILE,
                             behavior_kickout=behavior_kickout, change_to=change_to, check_adapt=check_adapt,
                             code_kickout=code_kickout)

        if self.simulate:
            self.data.simulate_data(D_fake=D_fake, scalcoefGT=scalcoefGT, constGT=constGT,
                                    mod_dimGT=mod_dimGT, mod_sigGT=mod_sigGT,
                                    plotit=True, times=times)
            fig, ax = plt.subplots(1, 2, figsize=(12, 3))
            xxdim = 0
            for tt in range(self.data.Ttrials):
                ax[0].plot(self.data.mGT[:, xxdim, tt])
            ax[0].set_title('true modulator 1st dim, all trials')
            ax[1].plot(self.data.mGT[:, xxdim, 0], self.data.data_trial[:, np.argmax(self.data.wGT[xxdim,:]), 0] / (
                        np.nanmin(self.data.data_trial[:, np.argmax(self.data.wGT[xxdim,:]), 0]) + .0001), 'o', label='modulated')
            ax[1].plot(self.data.mGT[:, xxdim, 0], self.data.data_trial[:, np.argmin(self.data.wGT[xxdim,:]), 0] / (
                        np.nanmin(self.data.data_trial[:, np.argmin(self.data.wGT[xxdim,:]), 0]) + .0001), 'o', label='not modulated')
            ax[1].set_xlabel('1st dim modulator')
            ax[1].set_ylabel('FR')
            ax[1].legend()

    def plot_activitybystim(self):
        XTMP, YTMP = concat_trials(self.data.data_trial, self.data.counts0, self.data.X)

        fig2, ax2 = plt.subplots(2, 2, figsize=(17, 8))
        for rr in range(2):
            ax2[0, rr - 1].plot([0, self.data.ST_aft_stim_windows], [0, 0], '--k')
            xtmp = XTMP[XTMP[:, 0] == rr, :]
            ytmp = YTMP[XTMP[:, 0] == rr, :]
            for nn in range(self.data.D):
                nmean = np.nan
                nmean2 = np.nan
                for rrii in range(self.data.ST_aft_stim_windows):
                    tmp = ytmp[xtmp[:, rrii + 1 + self.data.ST_inc_direction] == 1, nn]
                    tmp[tmp == 0] = .000001
                    ax2[0, rr - 1].plot([rrii, rrii + 1], [nmean, np.mean(np.log(tmp))], 'o-',
                                        color=plt.cm.coolwarm(nn / self.data.D))
                    ax2[1, rr - 1].plot([rrii, rrii + 1], [nmean2, np.mean(tmp)], 'o-',
                                        color=plt.cm.coolwarm(nn / self.data.D))
                    nmean = np.mean(np.log(tmp))
                    nmean2 = np.mean(tmp)
            ax2[0, rr - 1].set_xlim(0, self.data.ST_aft_stim_windows + 1)
            ax2[1, rr - 1].set_xlim(0, self.data.ST_aft_stim_windows + 1)
            ax2[1, rr - 1].set_xlabel('stimulus time window')
            ax2[0, rr - 1].set_ylabel('log FR')
            ax2[1, rr - 1].set_ylabel('FR')
        ax2[0, 0].set_title('contrast low')
        ax2[0, 1].set_title('contrast high')

        fig3, ax3 = plt.subplots(1, 1, figsize=(17, 2))
        ytmp = np.mean(YTMP[(XTMP[:, 0] == 1) &
                            (np.sum(XTMP[:, (1 + self.data.ST_inc_direction):(self.data.ST_N_timebins + 1 + self.data.ST_inc_direction)], axis=1) == 1), :], axis=0) - \
               np.mean(YTMP[(XTMP[:, 0] == 0) &
                            (np.sum(XTMP[:, (1 + self.data.ST_inc_direction):(self.data.ST_N_timebins + 1 + self.data.ST_inc_direction)], axis=1) == 1), :], axis=0)

        var = np.sqrt(.5 * (np.var(YTMP[(XTMP[:, 0] == 1) &
                                        (np.sum(XTMP[:, (1 + self.data.ST_inc_direction):(self.data.ST_N_timebins + 1 + self.data.ST_inc_direction)], axis=1) == 1),
                                   :], axis=0)) +
                      np.var(YTMP[(XTMP[:, 0] == 0) &
                                  (np.sum(XTMP[:, (1 + self.data.ST_inc_direction):(self.data.ST_N_timebins + 1 + self.data.ST_inc_direction)], axis=1) == 1), :],
                             axis=0))

        ax3.plot([0, self.data.D], [0, 0], '--', color='grey')
        ax3.set_xlabel('neurons')
        ax3.set_ylabel('FR high contr - low')
        for nn in range(self.data.D):
            ax3.plot([nn, nn], [ytmp[nn] - var[nn] / 2, ytmp[nn] + var[nn] / 2], '-',
                     color=plt.cm.coolwarm(nn / self.data.D))
            ax3.plot(nn, ytmp[nn], 'o', color=plt.cm.coolwarm(nn / self.data.D))
        blow = np.min(ytmp) - np.max(var / 2)
        bup = np.max(ytmp) + np.max(var / 2)
        ax3.set_ylim(blow, bup)

        if (self.data.ST_aft_stim_windows > self.data.ST_N_timebins):
            for rr in range(self.data.ST_N_timebins + 1 + self.data.ST_inc_direction, self.data.ST_aft_stim_windows + 1 + self.data.ST_inc_direction):
                fig3, ax3 = plt.subplots(1, 1, figsize=(17, 2))
                ytmp = np.mean(YTMP[XTMP[:, rr] == 1, :], axis=0) - np.mean(YTMP[XTMP[:, 0] == 0, :], axis=0)
                var = np.sqrt(
                    .5 * (np.var(YTMP[XTMP[:, rr] == 1, :], axis=0) + np.var(YTMP[XTMP[:, 0] == 0, :], axis=0)))

                ax3.plot([0, self.data.D], [0, 0], '--', color='grey')
                ax3.set_xlabel('neurons')
                ax3.set_ylabel('FR ' + np.str(rr - self.data.ST_N_timebins) + ' after stim off - off')
                for nn in range(self.data.D):
                    ax3.plot([nn, nn], [ytmp[nn] - var[nn] / 2, ytmp[nn] + var[nn] / 2], '-',
                             color=plt.cm.coolwarm(nn / self.data.D))
                    ax3.plot(nn, ytmp[nn], 'o', color=plt.cm.coolwarm(nn / self.data.D))
                ax3.set_title('bin ' + np.str(rr))
                ax3.set_ylim(blow, bup)
        if self.data.ST_inc_direction:
            fig4, ax4 = plt.subplots(1, 1, figsize=(17, 2))
            ytmp = np.abs(np.mean(YTMP[(XTMP[:, 1] == 1) & (XTMP[:, 0] == 1), :], axis=0) - np.mean(
                YTMP[(XTMP[:, 1] == 0) & (XTMP[:, 0] == 1), :], axis=0))
            var = np.sqrt(.5 * (np.var(YTMP[(XTMP[:, 1] == 1) & (XTMP[:, 0] == 1), :], axis=0) + np.var(
                YTMP[(XTMP[:, 1] == 0) & (XTMP[:, 0] == 1), :], axis=0)))
            ax4.plot([0, self.data.D], [0, 0], '--', color='grey')
            for nn in range(self.data.D):
                ax4.plot([nn, nn], [ytmp[nn] - var[nn] / 2, ytmp[nn] + var[nn] / 2], '-',
                         color=plt.cm.coolwarm(nn / self.data.D))
                ax4.plot(nn, ytmp[nn], 'o', color=plt.cm.coolwarm(nn / self.data.D))
            ax4.set_xlabel('neurons')
            ax4.set_title('FR abs diff with direction (only high contr)')
            ax4.set_ylim(blow, bup)

    def create_model(self, MINDIM=1, MAXDIM=3, Ncrossval=1,
                     scalQ=.01, scalQ0=.001, maxiter=10, maxtim=1000, difflikthresh=.001,
                     upx0=True, upQ0=True, upQ=True, upA=True, upC=True, upB=True, Adiag=False, regA=True,
                     backtracking=True, backtrack_diff=0):
        if self.simulate:
            path = self.path_save+'simulation/models/'+self.name + '/'
        else:
            path = self.path_save+'data/models/'+self.name + '/'

        self.DtM = DATAtoMODEL()
        self.DtM.par(data=self.data, MINDIM=MINDIM, MAXDIM=MAXDIM, Ncrossval=Ncrossval, path=path, saveresults=True, seed=0,
                scalQ=scalQ, scalQ0=scalQ0, maxiter=maxiter, maxtim=maxtim, difflikthresh=difflikthresh,
                upx0=upx0, upQ0=upQ0, upQ=upQ, upA=upA, upC=upC, upB=upB, Adiag=Adiag, regA=regA,
                backtracking=backtracking, backtrack_diff=backtrack_diff)
        self.DtM.C_starting = None
        self.DtM.A_starting = None

    def fit_SR(self, ee, Ttrain=None, method = 'Newt', CV_meth= 'median', tol = .0001,
                  LAM=np.array([0, 0.000001, .0001, .001, .01, .1, 1, 10], dtype='int'),
                  Ncross=5, plotit=False, comp_SR_pois=True, comp_SR_loglin=True):
        if Ttrain is None:
            Ttrain = np.array(np.round(self.data.Ttrials * .9), dtype='int')  # number of trials for training

        self.pred = learn_SR()
        self.pred.par()
        self.pred.learn(self.data.data_trial, self.data.X, self.data.D, self.data.Ttrials, LAM, Ncross, counts0=self.data.counts0,
                   Ttrain=Ttrain, train=self.DtM.TRAINTRIALS[:, ee],
                   comp_SR_pois=comp_SR_pois, comp_SR_loglin=comp_SR_loglin,
                   method=method, tol=tol,
                   CV_meth=CV_meth, plotit=plotit, disp=plotit)
        if plotit:
            fig5, ax5 = plt.subplots(1,1,figsize=(15,2))
            ax5.set_title('converged?')
            for nn in range(self.data.D):
                ax5.plot(nn, self.pred.SUCCES[nn], 'o', color=plt.cm.coolwarm(nn/self.data.D))

        self.pred.predict(self.data.data_trial, self.data.X, self.data.D, self.data.Ttrials, Ttrain=Ttrain, test=self.DtM.TESTTRIALS[:, ee],
             counts0=self.data.counts0, comp_SR_pois=comp_SR_pois, comp_SR_loglin=comp_SR_loglin)
        self.DtM.SR_MSE[:, ee] = self.pred.SR_MSE_pois
        self.DtM.SR_pred[:, :,:,ee] = self.pred.PRED_pois

        # likelihood
        logFR = logFR_predict(self.data, self.pred.beta_pois, latent=None, C=None)
        self.DtM.SR_lik[ee, :] = loglik_pred(self.data, logFR, trials=self.DtM.TESTTRIALS[:, ee])
        # residuals
        _, ytmp = concat_trials(self.data.data_trial - self.pred.PRED_pois, self.data.counts0, self.data.X)
        self.DtM.RESIDUALS[:, :, ee] = ytmp

    def starting_values(self, estA=None):
        # save beta for starting values in PLDS
        self.DtM.estB = self.pred.beta_pois.T
        # get starting values for C through PCA
        self.DtM.C_starting = [None] * (self.DtM.MAXDIM - self.DtM.MINDIM + 1)
        for xdim in range(self.DtM.MINDIM,self.DtM.MAXDIM+1):
            C_starting_ee = np.zeros([self.data.D, xdim, self.DtM.Ncrossval])*np.nan
            for ee in range(self.DtM.Ncrossval):
                pcobj_res = decomposition.PCA(whiten=True,n_components=xdim)
                pcobj_res.fit(self.DtM.RESIDUALS[:, :, ee])
                loadings = pcobj_res.components_
                if loadings.shape[0]==xdim:
                    loadings = loadings.T
                # outliers?
                cut = (np.percentile(np.abs(loadings), 75))
                loadings[np.abs(loadings)>cut] = cut
                if xdim==1:
                    C_starting_ee[:,:,ee] = loadings/np.sqrt(np.sum(loadings**2))
                else:
                    C_starting_ee[:, :, ee] = loadings / np.sqrt(np.sum(loadings ** 2, axis=0))
            self.DtM.C_starting[xdim-self.DtM.MINDIM] = C_starting_ee
        if estA is not None:
            self.DtM.A_starting= [None] * (self.DtM.MAXDIM - self.DtM.MINDIM + 1)
            self.DtM.A_starting[0] = estA

    def plot_SR_coef(self):
        if self.simulate==False:
            print('only for simulated data with known ground truth')
        else:
            INF = np.array(self.data.D / 2, dtype='int')
            # for fake data with ground truth
            fig, ax = plt.subplots(1, 3, figsize=(15, 5))
            ax[0].plot(self.data.betaGT[:, :INF], self.pred.beta_pois[:, :INF], 'or');
            ax[0].plot(self.data.betaGT[:, INF:], self.pred.beta_pois[:, INF:], 'ok');
            ax[0].set_aspect('equal')
            if self.pred.beta is not None:
                ax[1].plot(self.data.betaGT[:, :INF], self.pred.beta[:, :INF], 'or');
                ax[1].plot(self.data.betaGT[:, INF:], self.pred.beta[:, INF:], 'ok');
            ax[0].set_title('pois beta')
            ax[1].set_title('linreg beta')
            ax[1].set_title('offset beta')
            ax[0].set_xlabel('true coefficients')
            ax[0].set_ylabel('estimated coefficients')
            ax[0].plot([-2, 5], [-2, 5], '--', color='grey')
            ax[1].plot([-2, 5], [-2, 5], '--', color='grey')
            if self.pred.beta_cont is not None:
                ax[2].plot(self.data.mfr, self.data.mfr, '-', color='grey')
                ax[2].plot(self.data.mfr[:INF], self.pred.beta_cont[:INF], 'or', label='stim driven');
                ax[2].plot(self.data.mfr[INF:], self.pred.beta_cont[INF:], 'ok', label='stim ind');
            ax[2].set_aspect('equal')
            ax[2].legend()
            ax[2].set_xlabel('true mean firing rate')
            ax[2].set_ylabel('offset estimated')

    def plot_SR_pred(self, nn=None, nn2=None):
        # prediction

        fig, ax = plt.subplots(1, 4, figsize=(15, 3))
        ax[0].boxplot(self.pred.SR_MSE_pois, positions=np.array([0]))
        if self.pred.beta is not None:
            ax[0].boxplot(self.pred.SR_MSE, positions=np.array([1]))
            ax[0].boxplot(self.pred.SR_MSE_cont, positions=np.array([2]))
        ax[0].set_xlim(-1, 3)
        if self.pred.beta is not None: ax[0].set_ylim(0, np.max(self.pred.SR_MSE))
        ax[0].set_xticks(np.arange(3))
        ax[0].set_xticklabels(('pois', 'lin', 'offset'))
        ax[1].boxplot(self.pred.SR_MSE_pois)
        ax[1].set_title('poisson')
        ax[2].set_title('log linear')
        ax[3].set_title('offset')
        if self.pred.beta is not None:
            ax[2].boxplot(self.pred.SR_MSE)
            ax[3].boxplot(self.pred.SR_MSE_cont)

        if self.pred.beta is not None:
            print('const ', np.median(self.pred.SR_MSE_cont))
            print('lin reg ', np.median(self.pred.SR_MSE))
        print('pois ', np.median(self.pred.SR_MSE_pois))

        # var criteria
        if nn is None:
            NN = np.argsort(
                np.mean(np.nanvar(self.pred.PRED_pois, axis=0), axis=1) / np.mean(np.nanmean(self.data.data_trial, axis=2), axis=0))
            nn = NN[-1]
            nn2 = NN[0]
            print('chose neuron with smallest and largest predicted FF')


    def fit_PLDS(self, printit=True, multiproc=False, num_proc=None, computeMSE = True, computelik=True):
        self.DtM.PLDS_model(data=self.data, printit=printit, saveresults=True, fig=None, norm=False,
                            multiproc=multiproc, num_proc=num_proc)
        if computelik:
            self.DtM.PLDS_likelihood(self.data)
        if computeMSE:
            print('compute MSE')
            _ = self.DtM.PLDS_computeMSE(self.data, saveresults=True, printit=True, multiproc=multiproc, num_proc=num_proc)

    def plot_latent(self, xdim, ee):
        # look at latents and compare to GT parameters

        MOD = pickle.load(
            open(self.DtM.path + self.DtM.name + '_PLDS_ncross_' + np.str(ee) + '_xdim_' + np.str(xdim) + '.pk', 'rb'))

        fig, ax = plt.subplots(1, 2+self.simulate, figsize=(10+self.simulate*5, 4))
        tmp = np.mean(np.nanmean(self.data.data_trial, axis=0), axis=1)
        ax[0].plot(MOD.estC, tmp / np.max(tmp), '.')
        ax[0].set_title('max FR over C')
        ax[0].set_xlabel('C')
        ax[0].set_ylabel('FR normalized')
        if self.simulate:
            ax[1].plot(self.data.betaGT, self.data.betaGT, '-', color='grey')
            ax[2].plot(self.data.wGT[0, :], self.data.wGT[0, :], '-', color='grey')
            for rr in range(MOD.xdim):
                ax[2].plot(self.data.wGT[0, :], MOD.estC[:, rr], 'o', color=plt.cm.coolwarm(rr / MOD.xdim),
                           label='latent dim ' + np.str(rr))
            ax[1].plot(self.data.betaGT, self.pred.beta_pois, 'ob')
            ax[1].plot(self.data.betaGT.T, MOD.estB, 'or')
            ax[1].plot(0, 0, '-b', label='SR')
            ax[1].plot(0, 0, '-r', label='PLDS')
            ax[2].legend()
            ax[1].legend()
            ax[1].set_title('stimulus coefficient')
            ax[2].set_title('C over true w')
            ax[1].set_xlabel('true B')
            ax[1].set_ylabel('estimated B')
        else:
            ax[1].plot(MOD.estC[:,0], self.pred.beta_pois.T, 'o')
            ax[1].set_xlabel('estimated C (first dimension')
            ax[1].set_ylabel('estimated B (from SR)')
        if self.simulate:
            fig2, ax2 = plt.subplots(1, np.max([2, MOD.xdim]), figsize=(16, 4))
            fig3, ax3 = plt.subplots(1, np.max([2, MOD.xdim]), figsize=(16, 4))
            for rr in range(MOD.xdim):
                ax2[rr].set_title('dim %.0f' % (rr + 1))
                ax3[rr].set_title('dim %.0f' % (rr + 1))
                for tt in range(MOD.estx.shape[2]):
                    ax2[rr].plot(MOD.estx[:, rr, tt], '-', color=plt.cm.coolwarm(tt / MOD.estx.shape[2]));
                    ax3[rr].plot(self.data.mGT[:, 0, self.DtM.TRAINTRIALS[tt, ee]], MOD.estx[:, rr, tt], '.',
                                 color=plt.cm.coolwarm(tt / MOD.estx.shape[2]));
            ax2[0].set_xlabel('time')
            ax2[0].set_ylabel('fitted latent')
            ax3[0].set_xlabel('real modulator')
            ax3[0].set_ylabel('fitted latent')
        else:
            fig2, ax2 = plt.subplots(1, np.max([2, MOD.xdim]), figsize=(16, 4))
            for tt in range(MOD.estx.shape[2]):
                for rr in range(MOD.xdim):
                    ax2[rr].plot(MOD.estx[:, rr, tt], '-', color=plt.cm.coolwarm(tt / MOD.estx.shape[2]));
                    ax2[rr].set_xlabel('time')
                    ax2[rr].set_ylabel('estimated latent')

        print(self.DtM.EMiter[ee, :], ' iterations in EM')
        if self.simulate:
            print('est A: ', MOD.estA, ' compared to real A=', self.data.AGT)
            print('est Q: ', MOD.estQ, 'compared to real Q=', self.data.mod_sigGT ** 2)
            print('est Q0: ', MOD.estQ0, 'compared to realQ0=', (self.data.mod_sigGT/10) ** 2)
        else:
            print('est A: ', MOD.estA)
            print('est Q: ', MOD.estQ)
            print('est Q0: ', MOD.estQ0)

    def plot_latent_by_stim(self, ee, xdim):
        MOD = pickle.load(
            open(self.DtM.path + self.DtM.name + '_PLDS_ncross_' + np.str(ee) + '_xdim_' + np.str(xdim) + '.pk', 'rb'))
        XTMP, YTMP = concat_trials(self.data.data_trial[:, :, self.DtM.TRAINTRIALS[:, ee]], self.data.counts0[self.DtM.TRAINTRIALS[:, ee]],
                                   self.data.X[:, :, self.DtM.TRAINTRIALS[:, ee]])
        lat, _ = concat_trials(self.data.data_trial[:, :, self.DtM.TRAINTRIALS[:, ee]], self.data.counts0[self.DtM.TRAINTRIALS[:, ee]], MOD.estx)
        for xx in range(MOD.xdim):
            fig, ax = plt.subplots(1, 2, figsize=(15, 3))
            for rr in range(XTMP.shape[1]):

                ax[0].boxplot(np.mean(YTMP[XTMP[:, rr] == 0, :], axis=1),
                              positions=np.array([rr * 2]))
                ax[0].boxplot(np.mean(YTMP[XTMP[:, rr] == 1, :], axis=1),
                              positions=np.array([rr * 2 + .5]))

                ax[1].boxplot(lat[XTMP[:, rr] == 0, xx],
                              positions=np.array([rr * 2]))
                ax[1].boxplot(lat[XTMP[:, rr] == 1, xx],
                              positions=np.array([rr * 2 + .5]))
            ax[0].legend()
            ax[0].set_title('mean pop activity')
            ax[1].set_title('latent values')
            ax[0].set_xlim(-1, XTMP.shape[1] * 2)
            ax[1].set_xlim(-1, XTMP.shape[1] * 2)
            ax[0].set_ylabel('%.0f latent dim' % (xx + 1))
        return fig, ax

    def plot_MSE(self, ee):
        fig, ax = plt.subplots(2, 1, figsize=(10, 8))

        xtmp_SR, ytmp_SR = concat_trials(self.data.data_trial[:, :, self.DtM.TESTTRIALS[:, ee]],
                                         self.data.counts0[self.DtM.TESTTRIALS[:, ee]],
                                         self.pred.PRED_pois[:, :, self.DtM.TESTTRIALS[:, ee]])
        SR_MSE = np.sqrt(np.sum((xtmp_SR - ytmp_SR) ** 2, axis=0))
        ax[0].boxplot(SR_MSE, positions=np.array([0]));
        ax[1].plot([0, self.DtM.MAXDIM + 1], [1, 1], '--', color='grey')

        for xdim in range(self.DtM.MINDIM, self.DtM.MAXDIM + 1):
            xtmp, ytmp = concat_trials(self.data.data_trial[:, :, self.DtM.TESTTRIALS[:, ee]],
                                       self.data.counts0[self.DtM.TESTTRIALS[:, ee]],
                                       self.DtM.PLDS_pred[xdim - 1][:, :, self.DtM.TESTTRIALS[:, ee], ee])
            ax[0].boxplot(np.sqrt(np.sum((xtmp - ytmp) ** 2, axis=0)), positions=np.array([xdim]));
            ax[1].boxplot(np.sqrt(np.sum((xtmp - ytmp) ** 2, axis=0)) / SR_MSE, positions=np.array([xdim]));

        ax[0].set_xlim([-1, self.DtM.MAXDIM + 1])
        ax[1].set_xlim([0, self.DtM.MAXDIM + 1])
        ax[0].set_xticks(np.arange(self.DtM.MAXDIM + 1))
        ax[0].set_xticklabels(['SR', 'PLDS-1dim', 'PLDS-2dim', 'PLDS-3dim'])
        ax[1].set_xticks(np.arange(1, self.DtM.MAXDIM + 1))
        ax[1].set_xticklabels(['PLDS-1dim', 'PLDS-2dim', 'PLDS-3dim', 'PLDS-4dim'])
        ax[1].set_ylabel('relative to SR')
        ax[0].set_ylabel('MSE')
        return fig, ax

    def plot_PLDS_pred(self, ee, xdim, ind=None):
        # PLDS
        pred_pois, ytmp = concat_trials(self.data.data_trial[:, :, self.DtM.TESTTRIALS[:, ee]],
                                        self.data.counts0[self.DtM.TESTTRIALS[:, ee]],
                                        self.DtM.PLDS_pred[xdim - 1][:, :, self.DtM.TESTTRIALS[:, ee], ee])
        # SR
        pred_SR, _ = concat_trials(self.data.data_trial[:, :, self.DtM.TESTTRIALS[:, ee]], self.data.counts0[self.DtM.TESTTRIALS[:, ee]],
                                   self.pred.PRED_pois[:, :, self.DtM.TESTTRIALS[:, ee]])
        # stimulus X
        x, _ = concat_trials(self.data.data_trial[:, :, self.DtM.TESTTRIALS[:, ee]], self.data.counts0[self.DtM.TESTTRIALS[:, ee]],
                             self.data.X[:, :, self.DtM.TESTTRIALS[:, ee]])
        if ind is None:
            ind =  np.argmax(np.var(pred_pois, axis=0)/np.var(pred_SR, axis=0)) # np.argmin(np.mean((pred_pois-ytmp)**2, axis=0)/np.mean(ytmp, axis=0)) #
            print('neuron with highest variance in PLDS compared to SR %.0f' %(ind+1))
        print('neuron with lowest MSE in SR normalized by FR %.0f' % (np.argmin(np.mean((pred_SR-ytmp)**2, axis=0)/np.mean(ytmp, axis=0))+1))

        fig, ax = plt.subplots(1,1, figsize=(15,5))
        tmp = ytmp[:,ind]
        ax.plot([0, len(tmp)], [np.mean(tmp), np.mean(tmp)], '--', color='grey')
        ax.plot(tmp, 'k', label='data')
        ax.set_title('neuron %.0f' %ind)
        tmp_pred = pred_pois[:,ind]
        ax.plot([0, len(tmp_pred)], [np.mean(tmp_pred), np.mean(tmp_pred)], '--', color='green')
        ax.plot(tmp_pred, '-g',label='PLDS')
        tmp_pred = pred_SR[:,ind]
        ax.plot([0, len(tmp_pred)], [np.mean(tmp_pred), np.mean(tmp_pred)], '--', color='orange')
        ax.plot(tmp_pred, '-r',label='SR')
        ax.set_xlabel('time')
        ax.set_ylabel('activity')
        ax.legend()

        fig, ax = plt.subplots(1,1, figsize=(15,2))
        ax.plot(x*np.arange(1,x.shape[1]+1));
        ax.set_xlabel('time')
        ax.set_ylabel('stimulus')

        fig, ax = plt.subplots(1,1, figsize=(15,4))
        ax.plot(np.mean(ytmp, axis=0), 'k', label='data');
        ax.plot(np.mean(pred_pois, axis=0), 'g-', label='PLDS');
        ax.plot(np.mean(pred_SR, axis=0), 'r', label='SR');
        ax.legend()
        ax.set_xlabel('neuron')
        ax.set_ylabel('mean activity')

class learn_SR:
    def par(self):
        self.PRED_loglin = None
        self.PRED_cont = None
        self.PRED_pois = None
        self.SR_MSE_loglin = None
        self.SR_MSE_cont = None
        self.SR_MSE_pois = None
        self.beta_loglin = None
        self.beta0 = None
        self.beta_cont = None
        self.beta_pois = None
        self.saveLAM = None
        self.trackLIK = np.array([np.nan])
        self.trackB = np.array([np.nan])

    def lik(self, b, XTMP, ytmp, lam=1, trackit=False):
        # computes unnormalized negative log-likelihood given Poisson probability of data
        if len(XTMP.shape)>1:
            xb = XTMP.dot(b)
            bb = b.T.dot(b)
        else:
            xb = XTMP*b
            bb = b**2
        if any(xb > 700):
            tmp = -999999999999999999
            print('hit exp bound in lik')
        else:
            tmp = xb * ytmp - np.exp(xb) - bb * lam

        if trackit:
            self.trackLIK = np.concatenate((self.trackLIK, np.array([-np.sum(tmp)])))
            self.trackB = np.concatenate((self.trackB, np.array(b)))

        tmp = -np.sum(tmp)

        return tmp

    def dlik(self, b, XTMP, ytmp, lam=1):
        if any(XTMP.dot(b) > 700):
            tmp = -999999999999999999
            print('hit exp bound in derivative')
        else:
            tmp = -((XTMP.T.dot(ytmp)) - (XTMP.T.dot(
                np.exp(XTMP.dot(b)))) - b * lam * 2)
        return tmp

    def Hlik(self, b, XTMP, ytmp, lam=1):
        dotprod = XTMP.dot(b)
        if any(dotprod > 700):
            tmp = -999999999999999999
            print('hit exp bound in hessian')
        else:
            res = np.einsum('ts,tu,t->su', XTMP, XTMP, np.exp(dotprod))
        return -(res - lam * 2)

    def learn(self, SPIKES, X, D, Ttrials, LAM, Ncross, Ttrain, counts0=None, train=None, comp_SR_pois=False, comp_SR_loglin=False,
              method = 'BFGS', tol=.0001, disp=True, CV_meth='mean', plotit=False, betastart=None):
        self.LAM=LAM
        SUCCES = np.zeros(D)*np.nan
        if plotit: fig, ax = plt.subplots(2,3,figsize=(15,8))
        saveLAM = np.zeros([D, len(self.LAM), Ncross])*np.nan
        saveLAM2 = np.zeros([len(self.LAM), Ncross])*np.nan


        # estimate B (doing regression on training trials)
        LAMplot = np.copy(self.LAM)
        LAMplot[self.LAM == 0] = .00001
        for cc in range(Ncross):
            traintrials, testtrials = traintest(Ttrials, Ttrain, seed=cc)

            if len(SPIKES.shape) == 3:
                XTMP, YTMP = concat_trials(SPIKES[:, :, traintrials], counts0[traintrials], X[:, :, traintrials])
            else:
                XTMP = X[traintrials, :]
                YTMP = SPIKES[traintrials, :]

            if len(SPIKES.shape) == 3:
                XTMPtest, YTMPtest = concat_trials(SPIKES[:, :, testtrials], counts0[testtrials], X[:, :, testtrials])
            else:
                XTMPtest = X[testtrials, :]
                YTMPtest = SPIKES[testtrials, :]
            YTMP[YTMP <= 0] = .00000001
            YTMPtest[YTMPtest <= 0] = .00000001
            ##############################################
            # cross-validate to check which lambda to use#
            ##############################################
            for ll in range(len(self.LAM)):
                if comp_SR_loglin:
                    print('ERROR, linear model not implemented here in minimal version')

                if comp_SR_pois:
                        for nn in range(D):
                            b0 = np.random.randn(X.shape[1])*.01
                            b0[b0>.1] = .1
                            if method == 'BFGS':
                                res = minimize(self.lik, x0=b0, jac=self.dlik, args=(XTMP, YTMP[:, nn], self.LAM[ll]),
                                               method='BFGS', options={'disp': False}, tol=tol)
                            if method == 'Newt':
                                res = minimize(self.lik, x0=b0, jac=self.dlik, hess=self.Hlik, args=(XTMP, YTMP[:, nn], self.LAM[ll]),
                                               method='Newton-CG',
                                               options={'disp': False}, tol=tol)
                            tmp = XTMPtest.dot(res.x)
                            if CV_meth=='mean':
                                saveLAM[nn, ll, cc] = np.mean((np.log(YTMPtest[:,nn])- tmp) ** 2)
                            if CV_meth=='median':
                                saveLAM[nn, ll, cc] = np.median((np.log(YTMPtest[:,nn])- tmp) ** 2)


        if (len(LAM) > 1)&plotit:
            if comp_SR_loglin:
                ax[0,0].plot(LAMplot, np.mean(saveLAM2, axis=1) / np.max(np.mean(saveLAM2, axis=1)), 'o-')
                ax[0,1].plot(LAMplot, np.var(saveLAM2, axis=1), '-o')  # / np.max(np.mean(lam_mse, axis=1)))
                ax[0,0].set_title('max-mean-normalized MSE on testing for each neurons')
                ax[0,0].set_xlabel('lambda')
                ax[0,1].set_title('max-mean-normalized var on testing for each neurons')
                ax[0,1].set_xlabel('lambda')
                ax[0,0].set_xscale('log')
                ax[0,1].set_xscale('log')
                print('best linreg lam= ', LAMplot[np.argmin(np.mean(saveLAM2, axis=1))])
            if comp_SR_pois:
                for nn in range(D):
                    ax[1,0].plot(LAMplot, np.mean(saveLAM[nn, :,:] , axis=1) / np.max(np.mean(saveLAM[nn, :,:] , axis=1)),
                                 color=plt.cm.coolwarm(nn/D))
                    ax[1,1].plot(LAMplot, np.var(saveLAM[nn, :,:] , axis=1), color=plt.cm.coolwarm(nn/D))
                    ax[1,0].set_title('max-mean-normalized MSE on testing for each neurons')
                    ax[1,0].set_xlabel('lambda')
                    ax[1,1].set_title('max-mean-normalized var on testing for each neurons')
                    ax[1,1].set_xlabel('lambda')
                    ax[1,2].plot(nn, LAMplot[np.argmin(np.mean(saveLAM[nn, :,:] , axis=1))], 'o', color=plt.cm.coolwarm(nn/D))
                    ax[1,2].set_xlabel('neuron')
                    ax[1,2].set_title('best lambda for Pois model')
                    ax[1,0].set_xscale('log')
                    ax[1,1].set_xscale('log')
                    ax[1,2].set_ylim(np.min(LAMplot)-np.min(LAMplot)/10, np.max(self.LAM)+np.max(LAMplot)/10)
                    ax[1,2].set_yscale('log')
                print('best Pois lambdas = ', self.LAM[np.argmin(np.mean(saveLAM, axis=2), axis=1)])
        ###############################################
        ####### fit model with chosen lambda ##########
        ###############################################
        if (train is None):
            traintrials, testtrials0 = traintest(Ttrials, Ttrain)
        if (train is not None):
            traintrials = train
            testtrials0 = None

        if len(SPIKES.shape)==3:
            XTMP, YTMP = concat_trials(SPIKES[:, :, traintrials], counts0[traintrials], X[:, :, traintrials])
        else:
            XTMP = X[traintrials,:]
            YTMP = SPIKES[traintrials,:]

        if comp_SR_loglin:
            YTMP[YTMP <= 0] = .00000001
            beta, beta0 = np.polyfit(XTMP, np.log(YTMP), lam=LAMplot[np.argmin(np.mean(saveLAM2, axis=1))], deg=1)
        beta_cont = np.mean(YTMP, axis=0)
        if comp_SR_pois:
            beta_pois = np.zeros([X.shape[1], D]) * np.nan
            for nn in range(D):
                if betastart is not None:
                    b0=np.concatenate((betastart, np.array([np.random.randn(1)*.01])))
                else:
                    b0 = np.random.randn(X.shape[1]) * .01
                    b0[b0 > .1] = .1

                if method == 'BFGS':
                    res = minimize(self.lik, x0=b0, jac=self.dlik, args=(XTMP, YTMP[:, nn], self.LAM[np.argmin(np.mean(saveLAM[nn,:,:], axis=1))]),
                                   method='BFGS', options={'disp': disp}, tol=tol)
                if method == 'Newt':
                    res = minimize(self.lik, x0=b0, jac=self.dlik, hess=self.Hlik, args=(XTMP, YTMP[:, nn], self.LAM[np.argmin(np.mean(saveLAM[nn,:,:], axis=1))]),
                                   method='Newton-CG',
                                   options={'disp': disp}, tol=tol)

                SUCCES[nn] = res.success
                beta_pois[:,nn] = res.x
                self.lambda_used = self.LAM[np.argmin(np.mean(saveLAM[nn,:,:], axis=1))]
        self.SUCCES = SUCCES
        if comp_SR_loglin:
            self.beta_loglin = beta
            self.beta0 = beta0
        self.beta_cont = beta_cont
        if comp_SR_pois:
            self.beta_pois = beta_pois
        self.saveLAM = saveLAM2
        self.saveLAM_pois = saveLAM

        return testtrials0

    def predict(self, SPIKES, X, D, Ttrials, counts0=None, Ttrain=None, test=None, comp_SR_pois=False, comp_SR_loglin=False):
        if test is not None:
            testtrials = test
        else:
            testtrials = np.arange(Ttrain, Ttrials)

        if len(X.shape) == 3:
            PRED_loglin = np.zeros([X.shape[0], D, Ttrials]) * np.nan
            logPRED = np.zeros([X.shape[0], D, Ttrials]) * np.nan
            PRED_pois = np.zeros([X.shape[0], D, Ttrials]) * np.nan
            logPRED_pois = np.zeros([X.shape[0], D, Ttrials]) * np.nan
            PRED_cont = np.zeros([X.shape[0], D, Ttrials]) * np.nan
            logPRED_cont = np.zeros([X.shape[0], D, Ttrials]) * np.nan
            mse = np.zeros([len(testtrials), D]) * np.nan
            mse_cont = np.zeros([len(testtrials), D]) * np.nan
            mse_pois = np.zeros([len(testtrials), D]) * np.nan
        else:
            PRED_loglin = np.zeros([Ttrials, D]) * np.nan
            logPRED = np.zeros([Ttrials, D]) * np.nan
            PRED_pois = np.zeros([Ttrials, D]) * np.nan
            logPRED_pois = np.zeros([Ttrials, D]) * np.nan
            PRED_cont = np.zeros([Ttrials, D]) * np.nan
            logPRED_cont = np.zeros([Ttrials, D]) * np.nan

        ### PREDICTION ###
        if len(X.shape) == 3:
            for tt in range(Ttrials):
                PRED_cont[:counts0[tt], :, tt] = np.repeat(self.beta_cont, counts0[tt]).reshape(D, counts0[tt]).T
                tmp = self.beta_cont
                tmp[tmp<=0] = .0001
                logPRED_cont[:counts0[tt], :, tt] = np.log(tmp)
                if comp_SR_pois:
                    tmp = X[:counts0[tt], :, tt].dot(self.beta_pois)  # predicted firing rate
                    logPRED_pois[:counts0[tt], :, tt] = tmp
                    tmp[tmp > 10] = 10
                    PRED_pois[:counts0[tt], :, tt] = np.exp(tmp)
                    if np.sum(np.isnan(PRED_pois[:counts0[tt], :, tt]))>0:
                        print('warning nans in prediction ', tt)
            # compute MSE for testing trials
            if SPIKES is not None:
                count = 0
                for tt in testtrials:
                    if comp_SR_loglin:
                        tmp = SPIKES[:counts0[tt], :, tt] - PRED_loglin[:counts0[tt], :,tt]
                    tmp_cont = SPIKES[:counts0[tt], :, tt] - PRED_cont[:counts0[tt], :, tt]
                    if comp_SR_pois:
                        tmp_pois = (SPIKES[:counts0[tt], :, tt] - PRED_pois[:counts0[tt],:, tt])
                    for dd in range(D):
                        if comp_SR_loglin:
                            mse[count, dd] = np.sum((tmp[:, dd])**2) / counts0[tt]
                        mse_cont[count, dd] = np.sum((tmp_cont[:, dd])**2) / counts0[tt]
                        if comp_SR_pois:
                            mse_pois[count, dd] = np.sum((tmp_pois[:, dd])**2) / counts0[tt]
                    count +=1
            if comp_SR_loglin:
                self.SR_MSE_loglin = np.mean(mse, axis=0)
                self.SR_MSE_cont = np.mean(mse_cont, axis=0)
            if comp_SR_pois:
                self.SR_MSE_pois = np.mean(mse_pois, axis=0)
        if len(X.shape)<3:
            PRED_cont = self.beta_cont
            if SPIKES is not None: self.SR_MSE_cont = np.sum((SPIKES[testtrials, :] - PRED_cont) ** 2)
            tmp = self.beta_cont
            tmp[tmp <= 0] = .0001
            logPRED_cont = np.log(tmp)
            if comp_SR_pois:
                logPRED_pois = X.dot(self.beta_pois)  # predicted firing rate
                tmp = np.copy(logPRED_pois)
                tmp[tmp > 10] = 10
                PRED_pois = np.exp(tmp)
                if np.sum(np.isnan(PRED_pois)) > 0:
                    print('warning nans in prediction ')
                if SPIKES is not None: self.SR_MSE_pois = np.sum((SPIKES[testtrials,:] - PRED_pois[testtrials,:]) ** 2)
        self.PRED_loglin = PRED_loglin
        self.PRED_cont = PRED_cont
        self.PRED_pois = PRED_pois
        self.logPRED = logPRED
        self.logPRED_cont = logPRED_cont
        self.logPRED_pois = logPRED_pois


####################### helper functions ############################

def helper_PLDSfitting(ee, mm, args):
    print('###################### fit ', np.str(mm) + ' dimension ######################################')
    ### train it ###
    _, _, _, _, _, emiter = model_train(data=args.data, DtM=args.DtM, xdim=mm, ee=ee,
                                               printit=args.printit,
                                               fig=args.fig,
                                               saveresults=True, norm=args.norm)
    print('emiter', emiter)
    
def helper_MSE(ee, xdim, args):

    print('cross-fold '+np.str(ee) + ', ' +np.str(xdim) + ' dimension(s)')
    MOD = pickle.load(open(args.DtM.path + args.DtM.name + '_PLDS_ncross_' + np.str(ee) + '_xdim_' + np.str(xdim) + '.pk', 'rb'))
    MODall = fit_to_all_trials(data_trial=args.data.data_trial, MOD=MOD, counts0=args.data.counts0, X=args.data.X,
                               seedtest=args.DtM.seedtest)
    estCdeg, Cdeg_arot, estxdeg, estx0deg, estAdeg, xdeg_arot, As, AvT, Au, cho_est, evecest = \
        PLDS_rotations(MODall, scal=1, plotit=False, printit=args.printit)
    # model fit with A-rotated dimensions
    mse, pred = model_test_lno(MODall=MODall, testtrials=args.DtM.TESTTRIALS[:, ee], seedtest=1, data_trial=args.data.data_trial,
                       X=args.data.X, counts0=args.data.counts0,
                       rotate=(MODall.xdim > 1), cho_est=cho_est, evecest=evecest,
                       As=As, AvT=AvT, Au=Au, saveresults=False,
                       path=args.DtM.path, name=args.DtM.name, ee=ee, pred =True)
    
    np.save(args.DtM.path + 'mp_MSE_model_'+np.str(xdim)+'_cf_'+np.str(ee)+'_'+args.DtM.name + '.npy', mse)
    np.save(args.DtM.path + 'mp_PRED_model_'+np.str(xdim)+'_cf_'+np.str(ee)+'_'+args.DtM.name + '.npy', pred)
    
def define_list_preproc(DtM, args):
    if (DtM.MAXDIM - DtM.MINDIM) > 0:
        tmp1, tmp2 = np.meshgrid(np.arange(DtM.Ncrossval), np.arange(1, DtM.MAXDIM - DtM.MINDIM + 2))
        tmp1 = tmp1.reshape(DtM.Ncrossval * (DtM.MAXDIM - DtM.MINDIM+1))
        tmp2 = tmp2.reshape(DtM.Ncrossval * (DtM.MAXDIM - DtM.MINDIM+1))
        list_ee_xx = [[tmp1[xx], tmp2[xx], args] for xx in
                      range(DtM.Ncrossval * (DtM.MAXDIM - DtM.MINDIM+1))]
    else:
        list_ee_xx = [[xx, 1, args] for xx in range(DtM.Ncrossval)]
    return list_ee_xx

def logFR_predict(data, beta, latent=None, C=None, X = None, ttrial=None, testtrials=None):
    # ttrial ... index of all trials
    if ttrial is None:
        print('ttrial is None')
        tind = np.arange(data.Ttrials)
    else:
        tind = np.array(ttrial, dtype='int')
    if testtrials is None:
        testtrials = np.arange(data.Ttrials)
    if X is None: 
        if testtrials is None:
            X = data.X
            counts0 = data.counts0
        else:
            X = data.X[:,:,testtrials]
            counts0 = data.counts0[testtrials]
    else:
        counts0 = data.counts0
    if latent is None:
        print('latent is None')
        latent = X*0
        C = beta*0
    if len(latent.shape) == 2:
        print('len(latent.shape) == 2')
        tmp = np.zeros([latent.shape[0], latent.shape[1], len(testtrials)])*np.nan
        tmp[:, :, ttrial] = latent
        latent = np.copy(tmp)
    
    # predictive firing rate
    logFR = np.zeros([np.max(counts0), data.D, len(tind)])*np.nan
    for ttrial in tind:
        if C.shape[0]==1:
            latC = latent[:counts0[ttrial],:,ttrial]*C
        else:
            latC = latent[:counts0[ttrial],:,ttrial].dot(C)
        logFR[:counts0[ttrial],:,ttrial] = X[:counts0[ttrial],:,ttrial].dot(beta)+latC
    return logFR

def loglik_pred(data, logFR, trials=None, testtrials=None, nn=None, norm=True):
    # compute likelihood of held out data

    if trials is None:
        trials = np.arange(data.Ttrials)
    if testtrials is None:
        testtrials = trials
    if nn is None:
        nn = np.arange(data.D)
    yL = np.zeros(len(nn))
    for tt in range(len(trials)):

        yL += np.sum(data.data_trial[:data.counts0[testtrials[tt]],nn,testtrials[tt]]*
                         logFR[:data.counts0[testtrials[tt]],nn,trials[tt]]-\
                    np.exp(logFR[:data.counts0[testtrials[tt]],nn,trials[tt]]), axis=0)
        if norm:
            for nnii in range(len(nn)):
                tmp = np.array(data.data_trial[:data.counts0[testtrials[tt]],nn[nnii],testtrials[tt]], dtype='int')
                for ii in range(len(tmp)):
                    yL[nnii]-= math.log(math.factorial(tmp[ii]))
    return yL

def get_latent_mu_cov(xmu, MOD_test_nn,
                      C_est=False, estA=False, estQ=False, estQ0=False, B_est=False):
    iHL = np.zeros([np.max(MOD_test_nn.n_step)*MOD_test_nn.xdim, np.max(MOD_test_nn.n_step)*MOD_test_nn.xdim, MOD_test_nn.Ttrials])*np.nan
    for ttrial in range(MOD_test_nn.Ttrials):
        Ctilest = MOD_test_nn.expand_C(est=C_est, ttrial=ttrial)
        Sest = MOD_test_nn.expand_S(estA=estA, estQ=estQ, estQ0=estQ0, ttrial=ttrial)
        _, ystack = MOD_test_nn.expand_xy(ttrial=ttrial)
        dtil = MOD_test_nn.expand_d(est=B_est, ttrial=ttrial)
        nonan = np.isnan(xmu[:, 0, ttrial]) == False
        MOD_test_nn.estx[nonan, :, ttrial] = xmu[nonan, :, ttrial]
        xmuttrialstack, _ = MOD_test_nn.expand_xy(est=True, ttrial=ttrial)
        ihl = MOD_test_nn.invnegHLik(xmuttrialstack, Sest, Ctilest, dtil, ttrial=ttrial)
        iHL[:ihl.shape[0],:ihl.shape[0],ttrial] = ihl
    return MOD_test_nn, iHL

def lognorm(mu, sig):
    E = np.exp(mu+np.diag(sig)/2)
    COV = np.outer(mu,mu)*(np.exp(sig)-1)
    return E, COV


class dimensionality_test():

    def par(self, DtM, data, Nsamps=1000, pathsave=None):
        self.Nsamps = Nsamps
        self.samLL = np.zeros([data.D, DtM.Ncrossval, DtM.MAXDIM-DtM.MINDIM+1])*np.nan
        self.latent_sampl = [[None] * (DtM.Ncrossval)]* (DtM.MAXDIM) # latent_sampl[dimension][crossfold]
        if pathsave is None: self.pathsave = DtM.path + 'DIM/'
        else: self.pathsave = pathsave

    def x_FR_liks(self, data, MODall, MOD_test_nn, xmu, nnout, testtrials):
        logpred = logFR_predict(data, beta=MODall.B[nnout, :].T, latent=xmu,
                                C=MODall.C.T, ttrial=np.arange(MOD_test_nn.Ttrials), testtrials=testtrials)
        pred = np.exp(logpred)
        loglik = loglik_pred(data, logFR=logpred, trials=np.arange(MOD_test_nn.Ttrials), testtrials=testtrials,
                             nn=nnout, norm=True)
        return loglik, pred

    def sampl_x_FR_liks(self, data, MODall, MOD_test_nn, nnout, testtrials, iHL, savesamp=False, ee=None):
        if ee is None:
            ee=0
        # sampling
        LOGLIK = np.zeros(self.Nsamps) * np.nan
        if savesamp:
            self.latent_sampl[MODall.xdim - 1][ee] = np.zeros([np.max(MOD_test_nn.n_step), MOD_test_nn.xdim, MOD_test_nn.Ttrials, self.Nsamps])*np.nan
        for tt in range(self.Nsamps):
            # create a sample of the latent space
            xtmp = np.zeros(MOD_test_nn.estx.shape) * np.nan
            for ttrial in range(MOD_test_nn.Ttrials):
                xmuttrialstack, _ = MOD_test_nn.expand_xy(est=True, ttrial=ttrial)
                tmpHL = iHL[:MOD_test_nn.n_step[ttrial] * MOD_test_nn.xdim,
                                                                  :MOD_test_nn.n_step[ttrial] * MOD_test_nn.xdim,
                                                                  ttrial]
                try:
                    xtmp[:MOD_test_nn.n_step[ttrial], :, ttrial] = \
                        np.random.multivariate_normal(xmuttrialstack, tmpHL). \
                            reshape(MOD_test_nn.n_step[ttrial], MOD_test_nn.xdim)
                except np.linalg.LinAlgError:
                    print('LINALG ERROR in SVD during multi-normal! (use diag instead)')
                    tmpHL = np.eye(MOD_test_nn.n_step[ttrial] * MOD_test_nn.xdim)
                    xtmp[:MOD_test_nn.n_step[ttrial], :, ttrial] = \
                        np.random.multivariate_normal(xmuttrialstack, tmpHL). \
                            reshape(MOD_test_nn.n_step[ttrial], MOD_test_nn.xdim)


            LOGLIK[tt], _ = self.x_FR_liks(data, MODall, MOD_test_nn, xtmp, nnout, testtrials)
            if savesamp:
                self.latent_sampl[MODall.xdim-1][ee][:,:,:,tt] = xtmp
        return LOGLIK

    def loglik_sampling_byneuron(self, DtM, data, MODall, nnout, ee, plotit=False, savesamp=False):
        # compute the estimated latent trajectory from the training-neurons, for the testing-trials
        xmu, MOD_test_nn = compute_latent(nnout, MODall, 0, data.X, data.counts0, data.data_trial,
                                          DtM.TESTTRIALS[:, ee])
        # compute the likelihood of the held out neuron's activity under the model
        MOD_test_nn, iHL = get_latent_mu_cov(xmu, MOD_test_nn,
                                             C_est=False, estA=False, estQ=False, estQ0=False, B_est=False)
        # 1) using only the predicted latent-mean
        loglik, pred = self.x_FR_liks(data, MODall, MOD_test_nn, MOD_test_nn.estx, nnout, DtM.TESTTRIALS[:, ee])
        # 2) using sampling method to create distribution of FR from distribution of latent values
        LOGLIK = self.sampl_x_FR_liks(data, MODall, MOD_test_nn, nnout, DtM.TESTTRIALS[:, ee], iHL, ee=ee, savesamp=savesamp)
        # 3) compute analytical mean of the FR given latent-distribution
        if plotit:
            plt.figure()
            plt.hist(LOGLIK, 25);
            plt.plot([loglik, loglik], [0, self.Nsamps / 10], 'r')
            plt.plot([np.mean(LOGLIK), np.mean(LOGLIK)], [0, self.Nsamps / 10], 'k')
            plt.title('neuron ' + np.str(nnout[0]) + ' log lik summed all trials')
            print('neurons GR w=', (data.wGT[:, nnout].T[0]))
            print('neurons GR A=', np.diag(data.A))
            print('### estimation ###')
            print('neurons est w=', MODall.C[nnout, :])
            if MODall.xdim > 1:
                print('neurons est A=', np.linalg.eig(MODall.A)[0])
            else:
                print('neurons est A=', MODall.A[0])
        return np.mean(LOGLIK)

    def test(self, DtM, data, xdim=None, whichneuron = None, crossfold=None, savesamp=False, saveme=False,
             multiproc=False, num_proc=None, name='tmp_test'):
        if xdim is None:
            xdim = np.arange(1, DtM.MAXDIM + 1)
        if crossfold is None:
            crossfold = np.arange(DtM.Ncrossval)
        if whichneuron is None:
            whichneuron = np.arange(data.D)

        if multiproc:
            args = helperarg_DIM()
            args.par(DIM=self, DtM=DtM,data=data, whichneuron=whichneuron)
            mob = mp.Pool(processes=num_proc)
            print('ready to multiprocess')
            list_ee_xx = define_list_preproc_DIM(DtM, args)
            start = time.time()
            mob.starmap(helper_DIM, list_ee_xx)
            mob.close()
            end = time.time()
            print('****** total time for DIM assessment: ', (end - start), '******')
            for ee in crossfold:
                for xx in range(len(xdim)):
                    for nn in range(len(whichneuron)):
                        name_LL = 'crossfold' + np.str(ee) + '_xdim' + np.str(xdim[xx]) + '_neuron' + np.str(
                            whichneuron[nn])
                        if os.path.isfile(args.DIM.pathsave + name_LL + '.npy'):
                            self.samLL[whichneuron[nn], ee, xx] = np.load(args.DIM.pathsave + name_LL + '.npy')
            if saveme:
                print('test saving itself under ', DtM.path + 'DIM_' + name + '.pk')
                pickle.dump(self, open(DtM.path + 'DIM_' + name + '.pk', 'wb'))
            print('---------------')
        else:
            start = time.time()
            for ee in crossfold:
                print('cross-fold: ', ee)
                for xx in xdim:
                    print('xdim=', xx)

                    MOD = pickle.load(
                        open(DtM.path + DtM.name + '_PLDS_ncross_' + np.str(ee) + '_xdim_' + np.str(xx) + '.pk',
                             'rb'))
                    MODall = fit_to_all_trials(data.data_trial, MOD, data.counts0, data.X, seedtest=0)
                    mod, _ = concat_trials(data.data_trial, data.counts0, MODall.estx)

                    if MOD.xdim == 1:
                        print('est A', MOD.estA)
                    else:
                        print('est A', np.linalg.eig(MOD.estA)[0])
                    for nn in whichneuron:
                        nnout = np.array([nn])
                        self.samLL[nnout, ee, xx - 1] = self.loglik_sampling_byneuron(DtM,data, MODall, nnout, ee,
                                                                              plotit=False, savesamp=savesamp)
                    print(' ')
                    if saveme:
                        print('test saving itself under ',DtM.path + 'DIM_' + name + '.pk')
                        pickle.dump(self, open(DtM.path + 'DIM_' + name + '.pk', 'wb'))
                print('---------------')
            end = time.time()
            print('****** total time for DIM assessment: ', (end - start), '******')

class helperarg_DIM():
    def par(self, DIM, DtM,data, whichneuron):
        self.DIM = DIM
        self.DtM=DtM
        self.data=data
        self.whichneuron = whichneuron
        self.savesamp = False

def helper_DIM(ee, xx, nn, args, saveresults=True):
    name = 'crossfold' + np.str(ee) + '_xdim' + np.str(xx) + '_neuron' + np.str(nn)
    print('processing :' + name)
    print(' ')
    MOD = pickle.load(
        open(args.DtM.path + args.DtM.name + '_PLDS_ncross_' + np.str(ee) + '_xdim_' + np.str(xx) + '.pk',
             'rb'))
    MODall = fit_to_all_trials(args.data.data_trial, MOD, args.data.counts0, args.data.X, seedtest=0)
    mod, _ = concat_trials(args.data.data_trial, args.data.counts0, MODall.estx)
    if MOD.xdim == 1:
        print('est A', MOD.estA)
    else:
        print('est A', np.linalg.eig(MOD.estA)[0])
    nnout = args.whichneuron[nn]
    if isinstance(nnout, np.ndarray)==False:
        nnout = np.array([nnout])
    samLL = args.DIM.loglik_sampling_byneuron(args.DtM, args.data, MODall, nnout, ee,
                                                                      plotit=False, savesamp=args.savesamp)

    if saveresults:
        try:
            os.mkdir(args.DIM.pathsave)
        except OSError:
            print("already existing (overwriting) %s" % args.DIM.pathsave)
        else:
            print("created the directory %s " % args.DIM.pathsave)
        np.save(args.DIM.pathsave+ name+'.npy', samLL)

    return samLL

def define_list_preproc_DIM(DtM, args):
    if (DtM.MAXDIM - DtM.MINDIM) > 0:
        tmp1, tmp2, tmp3 = np.meshgrid(np.arange(DtM.Ncrossval), np.arange(1, DtM.MAXDIM - DtM.MINDIM + 2), args.whichneuron)
        tmp1 = tmp1.reshape(DtM.Ncrossval * (DtM.MAXDIM - DtM.MINDIM+1) * len(args.whichneuron))
        tmp2 = tmp2.reshape(DtM.Ncrossval * (DtM.MAXDIM - DtM.MINDIM+1) * len(args.whichneuron))
        tmp3 = tmp3.reshape(DtM.Ncrossval * (DtM.MAXDIM - DtM.MINDIM+1) * len(args.whichneuron))
        list_ee_xx = [[tmp1[xx], tmp2[xx], tmp3[xx], args] for xx in
                      range(DtM.Ncrossval * (DtM.MAXDIM - DtM.MINDIM+1) * len(args.whichneuron))]
    else:
        tmp1, tmp2 = np.meshgrid(np.arange(DtM.Ncrossval), args.whichneuron)
        tmp1 = tmp1.reshape(DtM.Ncrossval * len(args.whichneuron))
        tmp2 = tmp2.reshape(DtM.Ncrossval * len(args.whichneuron))
        list_ee_xx = [[tmp1[xx], 1, tmp2[xx], args] for xx in range(DtM.Ncrossval)]
    return list_ee_xx


def traintest(Ttrials, Ttrain, seed=None):
    # Ttrials can be a set of possible trials or a number of possible trials
    # Ttrain should be the number of trials
    if seed is None: seed = np.array(np.random.choice(1000, 1), dtype='int')
    np.random.seed(seed)
    traintrials = np.random.choice(Ttrials, Ttrain, replace=False)
    testtrials = np.arange(Ttrials)
    mask = np.ones(Ttrials, dtype='bool')
    mask[traintrials] = False
    testtrials = testtrials[mask]
    return traintrials, testtrials

def concat_trials(SPIKES, counts0, X=None):
    YTMP = np.copy(SPIKES[:counts0[0], :, 0])
    if X is not None: XTMP = np.copy(X[:counts0[0], :, 0])
    else: XTMP = X

    for tt in range(1, len(counts0)):
        YTMP = np.concatenate([YTMP, SPIKES[:counts0[tt], :, tt]])
        if X is not None: XTMP = np.concatenate([XTMP, X[:counts0[tt], :, tt]])
    return XTMP, YTMP


def load_MAIN(count: object, ee: object, DATAFILE, home: object = None, printit: object = False, need_to_rerun_target_response: object = False,
              path_save: object = None, path_get: object = None, direction=False, simulation: object = False, name: object = None,
              comp_inf_too=True, code_kickout=np.array([9, 15, 2, 3, 11, 10, 5, 8, 14, 6, 7, 12, 13, 14, 15]), nameabr='tmp') -> object:

    print('count = ', count, 'where name = ', name)
    ### get main file ###
    if simulation:
        if os.path.isfile(path_save + 'main_simulation_' + name + '.pk'):
            print('   load MAIN file')
            MAIN = pickle.load(open(path_save + 'main_simulation_' + name + '.pk', 'rb'))
            print('   main name = ', MAIN.name)
        else:
            print('   main file not constructed for simulation of' + name)
    else:
        if os.path.isfile(path_save + 'main_' + name + '.pk'):
            print('   load MAIN file')
            MAIN = pickle.load(open(path_save + 'main_' + name + '.pk', 'rb'))
        else:
            print('   main file not constructed for ' + path_save + 'main_' + name)
            return None, None, None, name
    if MAIN.simulate:
        MAIN.DtM.path = path_save + 'simulation/models/' + MAIN.name + '/'
    else:
        MAIN.DtM.path = path_save + 'data/models/' + MAIN.name + '/'
    ### get dimensionality file ###
    if os.path.isfile(path_save + 'DIM_' + MAIN.name + '.pk'):
        print('   load DIM file')
        DIM = pickle.load(open(path_save + 'DIM_' + MAIN.name + '.pk', 'rb'))
    else:
        if os.path.isdir(path_save + '/DIM'):
            print('   need to run assemble dim assessment data into file for ' + 'DIM_' + MAIN.name)
            DIM = dimensionality_test()
            if MAIN.simulate:
                DIM.par(MAIN.DtM, MAIN.data, Nsamps=1, pathsave=path_save + 'simulation/models/' + name + '/DIM/')
            else:
                DIM.par(MAIN.DtM, MAIN.data, Nsamps=1, pathsave=path_save + 'data/models/' + name + '/DIM/')
            whichneuron = np.arange(MAIN.data.D)
            args = helperarg_DIM()
            args.par(DIM=DIM, DtM=MAIN.DtM, data=MAIN.DtM, whichneuron=whichneuron)
            crossfold = np.arange(MAIN.DtM.Ncrossval)
            xdim = np.arange(1, MAIN.DtM.MAXDIM + 1)
            for eetmp in crossfold:
                for xx in range(len(xdim)):
                    for nn in range(len(whichneuron)):
                        name_LL = 'crossfold' + np.str(eetmp) + '_xdim' + np.str(xdim[xx]) + '_neuron' + np.str(
                            whichneuron[nn])
                        if os.path.isfile(args.DIM.pathsave + name_LL + '.npy'):
                            DIM.samLL[whichneuron[nn], eetmp, xx] = np.load(args.DIM.pathsave + name_LL + '.npy')
            print('MAIN saving itself under ', DIM.pathsave + 'DIM_' + MAIN.name + '.pk')
            pickle.dump(DIM, open(DIM.pathsave + 'DIM_' + MAIN.name + '.pk', 'wb'))
        else:
            print('   DIM file not there and files not constructed for ' + MAIN.name)
            DIM = None

    if printit:
        print('neurons: ', MAIN.data.D)
        print('number of trials: ', MAIN.data.Ttrials)
        print('number of total time points: ', np.sum(MAIN.data.counts0))
        print('number of total time points used for fitting in crossfold', ee, ': ',
              np.mean(np.sum(MAIN.data.counts0[MAIN.DtM.TRAINTRIALS[:, ee]], axis=0)))
        totwin = (MAIN.data.ST_N_timebins)
        print('number of stimulus presentations available to fit low contrast stimulus condition: ', \
              np.sum(np.sum(np.sum(MAIN.data.X[:, :totwin, :], axis=1) == 1, axis=0) / totwin))
        print('number of stimulus presentations available to fit high contrast stimulus condition: ', \
              np.sum(np.sum(np.sum(MAIN.data.X[:, totwin:(2 * totwin), :], axis=1) == 1, axis=0) / totwin))
        print('number of time points used for testing in crossfold', ee, ': ',
              np.mean(np.sum(MAIN.data.counts0[MAIN.DtM.TESTTRIALS[:, ee]], axis=0)))
        print('kept %.0f percent of units' % (100 * np.mean(MAIN.data.mask_neuron)))
        if MAIN.notes is not None:
            print(MAIN.notes)
        print('stimulus coefficients: ', MAIN.data.Xvar)
        print('lambda values: ', MAIN.pred.LAM)

    print('   ')
    if need_to_rerun_target_response:
        ######### FIXED PARAMETERS #################
        N_timebins = 4  # time bins per 200ms stimulus presentation
        aft_stim_windows = 1  # how many time bins am I estimating coefficients for (N_timebins+?)
        coFR = 0  # minimum firing rate
        # inter spike interval
        isi_crit = False  # use throwout criteria for ISI-violations

        notes = 'redo response to target stimulus'
        simulate = False

        ############ RUN ####################
        print(notes)

        print('count = ', count)

        name = nameabr+'_day_' + np.str(DATAFILE[count, 0]) + '_attention_' + np.str(DATAFILE[count, 1]) + \
               '_block_' + np.str(DATAFILE[count, 2])
        file = runDatatoModel()
        file.par(DATAFILE, path_save=path_save,
                 name=name, simulate=simulate, notes=notes)
        ###### load data #########

        # 1 is correct hit, 4 is miss, 3 is correct reject, 5 is false alarm
        file.load_data(count, coFR=coFR, remov=True, path_get=path_get,
                       ST_N_timebins=N_timebins, ST_aft_stim_windows=aft_stim_windows, ST_onoff_stim=False,
                       ST_inc_direction=direction,
                       isi_crit=isi_crit,
                       code_kickout=code_kickout, change_to=True,
                       check_adapt=False, behavior_kickout=False)
        print('no longer using out=3')
        if comp_inf_too:
            file.data.comp_informative()

        if printit:
            fig, ax = plt.subplots(1, 2, figsize=(15, 4))
            ax[0].plot(file.data.dprim)
            ax[0].plot([0, 100], [0, 0], '--k')
            ax[1].hist(file.data.dprim)
            ax[0].set_title("d'")
    else:
        file=MAIN

    return file, MAIN, DIM, name


def loglik(real, pred):
    logpred = np.copy(pred)
    logpred[logpred==0] = .000001
    logpred = np.log(logpred)
    return np.sum(real*logpred-pred, axis=0)
