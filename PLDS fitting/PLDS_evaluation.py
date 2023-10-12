# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 11:55:33 2018

@author: caroline
"""

import numpy as np
import matplotlib.pyplot as plt

from PLDS import PLDS
from sklearn.decomposition import FactorAnalysis
import pickle


def model_train(data, DtM, xdim, ee, fig=None, norm=False, printit=False, saveresults=False):
    if xdim < 1:
        print('error, latent dimension must be >=1')
    ###############################################################################################################
    ######################################### set up model: #######################################################
    ###############################################################################################################
    np.random.seed(ee)
    data_trial = data.data_trial[:, :, DtM.TRAINTRIALS[:,ee]]
    Xtrain = data.X[:, :, DtM.TRAINTRIALS[:,ee]]
    # create model
    MOD = PLDS()
    # initialize A and C
    if DtM.estC is not None:
        estC = DtM.estC
    else:
        if DtM.C_starting is not None:
            estC = DtM.C_starting[xdim-DtM.MINDIM][:,:,ee]
        else:
            estC=None
    if DtM.estA is not None:
        estA = DtM.estA
    else:
        if DtM.A_starting is not None:
            estA = DtM.A_starting[xdim-DtM.MINDIM]
        else:
            estA=None
    # initialize noise paramterers
    if DtM.estQ is None:
        estQ = np.eye(xdim)*DtM.scalQ
    else:
        estQ = np.copy(DtM.estQ)
    if DtM.estQ0 is None:
        estQ0 = np.eye(xdim)*DtM.scalQ0
    else:
        estQ0 = np.copy(DtM.estQ0)
    MOD.par(xdim, ydim=np.copy(data_trial.shape[1]), estx0=DtM.estx0,
            est=True, X=Xtrain, Ttrials=len(DtM.TRAINTRIALS[:,ee]), y=data_trial, n_step=data.counts0[DtM.TRAINTRIALS[:,ee]], seed=ee,
            estA=estA, estQ=estQ, estQ0=estQ0, estC=estC, estB=DtM.estB)

    ###############################################################################################################
    ######################################### fit model: ##########################################################
    ###############################################################################################################
    if printit: print('################### fit PLDS to data ########################')
    # initialize
    if (xdim>1)&(DtM.residuals)&(DtM.estC is not None):
        factor = FactorAnalysis(n_components=MOD.xdim, random_state=ee).fit(DtM.RESIDUALS[:,:,ee])
        MOD.estC = factor.components_.T/np.max(np.abs(factor.components_))

    normLik, fig, diffLik, allSIG, iiwhile = MOD.runEM(upA=DtM.upA, upB=DtM.upB, upQ=DtM.upQ, upQ0=DtM.upQ0, upx0=DtM.upx0, upC=DtM.upC, regA=DtM.regA,
                                                      Adiag=DtM.Adiag, Xtrain=Xtrain,
                                                       backtrack_diff=DtM.backtrack_diff, maxiter=DtM.maxiter,
                                                       maxtim=DtM.maxtim, fig=fig,
                                                       difflikthresh=DtM.difflikthresh, printit=printit,
                                                       backtracking=DtM.backtracking, norm=norm)

    if saveresults:
        print('\n \n model is being saved under: \n',
              DtM.path + DtM.name +'_PLDS'+ '_ncross_' + np.str(ee) + '_xdim_' + np.str(MOD.xdim) + '.pk',
              '\n \n')
        pickle.dump(MOD, open(DtM.path + DtM.name +'_PLDS'+ '_ncross_' + np.str(ee) + '_xdim_' + np.str(MOD.xdim) + '.pk', 'wb'))
    else:
        print('not saving this')
    return MOD, normLik, fig, diffLik, allSIG, iiwhile

def compute_latent(nnout, MODall, seedtest, X, counts0, data_trial, testtrials):
    mask = np.ones(MODall.ydim, dtype=bool)
    mask[nnout] = False
    # create variables without the left out neuron
    datacuttimtest_nn = data_trial[:, mask, :]
    # create testing model without left out neuron
    MOD_test_nn = PLDS()
    if len(testtrials)==1:
        n_step = np.array(counts0[testtrials])
    else:
        n_step = counts0[testtrials]
    MOD_test_nn.par(MODall.xdim, MODall.ydim - len(nnout), seed=seedtest, est=True,
                    y=datacuttimtest_nn[:, :, testtrials], Ttrials=len(testtrials), n_step=n_step,
                    X=X[:, :, testtrials],
                    C=MODall.C[mask, :], Q0=MODall.Q0, A=MODall.A, Q=MODall.Q, x0=MODall.x0, B=MODall.B[mask, :])

    # estimate the latent from the majority of neurons
    xfin, _ = MOD_test_nn.Estep(C_est=False, estA=False, estQ=False, estQ0=False, B_est=False,
                                estx0=False)
    return xfin, MOD_test_nn

def model_test_lno(MODall, testtrials, seedtest, data_trial, X, counts0, path=None, name=None, ee=None,
                   whichneuron=None,
                   rotate=False, cho_est=None, evecest=None, As=None, AvT=None, Au=None, saveresults=False,
                   pred=False):
    ###############################################################################################################
    # test model: 
    ###############################################################################################################
    if pred:
        PRED = np.zeros(MODall.y.shape)*np.nan
    if whichneuron is None:
        whichneuron = np.arange(MODall.ydim)
    if rotate:  # if true that error is estimated for every added dimension separatedly and dimensions are ordered
                # depending on their temporal component
        MSE = np.zeros([len(whichneuron), MODall.xdim, 2])*np.nan
    else:
        MSE = np.zeros([len(whichneuron)])*np.nan
    for nnoutii in range(len(whichneuron)):
        nnout = np.array([whichneuron[nnoutii]])
        # compute the latent on all but the nnoutii neuron
        xfin, MOD_test_nn = compute_latent(nnout, MODall, seedtest, X, counts0, data_trial, testtrials)

        # test prediction for activity of remaining neuron
        if rotate:
            mse_plds = np.zeros([MOD_test_nn.Ttrials, MODall.xdim])*np.nan
            mse_plds_cum = np.zeros([MOD_test_nn.Ttrials, MODall.xdim])*np.nan
        else:
            mse_plds = np.zeros([MOD_test_nn.Ttrials])*np.nan
            mse_plds_cum = None
        if rotate:
            for ttrial in range(MOD_test_nn.Ttrials):
                data_tt = data_trial[:MOD_test_nn.n_step[ttrial], nnout,testtrials[ttrial]]
                # rotate latent x and mapping matrix C
                estxdeg = (np.linalg.inv(cho_est).dot(evecest.T).dot(xfin[:MOD_test_nn.n_step[ttrial],:,ttrial].T)).T
                xdeg_arot = np.diag(np.sqrt(As)).dot(AvT).dot(estxdeg.T).T
                estCdeg = MODall.C.dot(evecest.dot(cho_est))
                Cdeg_arot = estCdeg.dot(Au).dot(np.diag(np.sqrt(As)))
                for xx in range(MODall.xdim):
                    pred_tt = np.exp(xdeg_arot[:, xx]*(Cdeg_arot[nnout, xx]) +
                                    MODall.d[:MOD_test_nn.n_step[ttrial], nnout, testtrials[ttrial]])
                    mse_plds[ttrial, xx] = np.sum((pred_tt -data_tt)**2) / MOD_test_nn.n_step[ttrial]
                    pred_tt_cum = np.exp(xdeg_arot[:, :(xx+1)].dot(Cdeg_arot[nnout, :(xx+1)].T) +
                                    MODall.d[:MOD_test_nn.n_step[ttrial], nnout, testtrials[ttrial]])
                    mse_plds_cum[ttrial, xx] = np.sum((pred_tt_cum - data_tt)**2) / MOD_test_nn.n_step[ttrial]
                if pred:
                    PRED[:MODall.n_step[testtrials[ttrial]], nnoutii, testtrials[ttrial]] = pred_tt_cum[:,0]

        else:
            for ttrial in range(MOD_test_nn.Ttrials):
                data_tt = data_trial[:MOD_test_nn.n_step[ttrial], nnout,testtrials[ttrial]]
                pred_tt = np.exp(xfin[:MOD_test_nn.n_step[ttrial],:,ttrial].dot(MODall.C[nnout,:])+
                                 MODall.d[:MOD_test_nn.n_step[ttrial], nnout, testtrials[ttrial]])
                mse_plds[ttrial] = np.sum((pred_tt-data_tt)**2) /\
                                   MOD_test_nn.n_step[ttrial]

                if pred:

                    PRED[:MODall.n_step[testtrials[ttrial]], nnoutii, testtrials[ttrial]] = pred_tt[:,0]


        if rotate:
            MSE[nnoutii, :, 0] = np.nanmean(mse_plds, axis=0) # error for each dimension by itself
            MSE[nnoutii, :, 1] = np.nanmean(mse_plds_cum, axis=0) # if an increasing number of dimensions is used
        else:
            MSE[nnoutii] = np.nanmean(mse_plds)
    if saveresults: pickle.dump(MSE, open(path + name + 'MSE_PLDS_ncross_' + np.str(ee) + '_xdim_' + np.str(MODall.xdim) + '.pk', 'wb'))
    if pred==False: PRED = None
    return MSE, PRED

def fit_to_all_trials(data_trial, MOD, counts0, X, seedtest):
    MODall = PLDS()
    MODall.par(xdim=MOD.xdim, ydim=MOD.ydim, seed = seedtest,
               est = True, y = data_trial, Ttrials=len(counts0),
                n_step=counts0,C = MOD.estC, Q0 = MOD.estQ0,A = MOD.estA,
               Q = MOD.estQ, x0 = MOD.estx0, B=MOD.estB, X=X)
    # estimate the latent
    MODall.estx, _ = MODall.Estep(C_est=False, estA=False, estQ=False, estQ0=False, B_est=False,
                                                estx0=False)
    return MODall


def PLDS_rotations(MODall, scal=1, plotit=False, printit=False):
    R = MODall.B.shape[1]
    # restructure so Q is identity
    if plotit:
        fig, ax = plt.subplots(1, 3, figsize=(18, 4))

    # remove degeneracy
    if MODall.xdim == 1:
        estCdeg = MODall.C * np.sqrt(MODall.Q)
        Cdeg_arot = np.copy(estCdeg)
        estxdeg = MODall.estx / np.sqrt(MODall.Q)
        estx0deg = MODall.x0 / np.sqrt(MODall.Q)
        if plotit:
            ax[0].plot(estCdeg)
            ax[0].set_title('C')
        estAdeg = np.copy(MODall.A)
        xdeg_arot = np.copy(estxdeg)
        As = np.copy(MODall.A)
        AvT = np.ones(1)
        Au = np.ones(1)
        cho_est = np.sqrt(MODall.Q)
        evecest = np.ones(1)
        if printit: print('A rotated for Q=I: ', estAdeg)
    else:
        evaluest, evecest = np.linalg.eig(MODall.Q)
        cho_est = np.diag(np.sqrt(evaluest / scal))
        estAdeg = np.diag(1 / np.sqrt(evaluest)).dot(evecest.T).dot(MODall.A).dot(evecest.dot(cho_est))

        estCdeg = MODall.C.dot(evecest.dot(cho_est))
        estxdeg = np.zeros([MODall.maxn_step, MODall.xdim, MODall.Ttrials]) * np.nan
        for ttrial in range(MODall.Ttrials):
            estxdeg[:(MODall.n_step[ttrial]), :, ttrial] = (np.linalg.inv(cho_est).dot(evecest.T).dot( \
                MODall.estx[:MODall.n_step[ttrial], :, ttrial].T)).T
        estx0deg = (np.diag(1 / np.sqrt(evaluest)).dot(evecest.T).dot(MODall.x0.T)).T

        # rotate parameters to correpond to the two A eigenspectra using svd
        Au, As, AvT = np.linalg.svd(estAdeg)
        Cdeg_arot = estCdeg.dot(Au)
        if printit: print('A singular values: ', As)
        if printit: print('A rotated eigenvalues: ', np.sort(np.linalg.eig(estAdeg)[0]))

        xdeg_arot = np.zeros([MODall.maxn_step, MODall.xdim, MODall.Ttrials]) * np.nan
        for ttrial in range(MODall.Ttrials):
            xdeg_arot[:(MODall.n_step[ttrial]), :, ttrial] = AvT.dot(estxdeg[:MODall.n_step[ttrial], :, ttrial].T).T

        if plotit:

            ax[0].plot(Cdeg_arot[:, 0], '.')
            ax[0].set_title('A-rotated-C1')
            ax[1].plot(Cdeg_arot[:, 1], '.')
            ax[1].set_title('A-rotated-C2')
            if MODall.xdim > 2:
                ax[3].plot(Cdeg_arot[:, 2], '.')
                ax[3].set_title('A-rotated-C3')

    if printit:
        print('x0 rotated for Q=I ', estx0deg)
        print('')

    if plotit:
        plt.figure(figsize=(17, 4))
        cmap = plt.cm.get_cmap('RdYlGn')
        ax[0] = plt.subplot2grid((1, 2), (0, 0))
        ax[1] = plt.subplot2grid((1, 2), (0, 1))
        ax[0].set_title('estimated stimulus response coefficients')
        for ii in range(R):
            ax[0].plot(MODall.B[:, ii], '--', color=cmap((ii / (R))), label=(ii + 1))
            ax[1].boxplot(MODall.B[:, ii], positions=np.array([ii + 1]), patch_artist=True, \
                          boxprops=dict(facecolor=cmap(ii / (R))))
            ax[0].legend()
        ax[1].set_xlim(0, R + 1)
        ax[1].set_title('distribution of stimulus coefficients for each stimulus')
    return estCdeg, Cdeg_arot, estxdeg, estx0deg, estAdeg, xdeg_arot, \
           As, AvT, Au, cho_est, evecest


