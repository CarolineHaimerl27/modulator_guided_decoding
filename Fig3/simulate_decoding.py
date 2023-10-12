
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb
import pickle
from simulate_response import simulate_response
from matplotlib import cm

#from matplotlib import cm
#import math

class simulate_decoding:

    def par(self, seed=1, Nsim=500, Nsimseed=0, T=10000, D=50, Ninf = np.array([6]), Ninact = np.zeros(1), ustim=np.array([1,4]), scalw=1,
            resol=1, mean=np.array([0,1,2], dtype=int),
            SIGM = np.concatenate([np.zeros(1), np.round(np.arange(0,.5, .05), 2), np.round(np.arange(0.5,1.5, .3), 1)]),
            beta = 0, sig_base = .5, BASE=np.array([2]), up=1.05, DOWN=np.array([.95]), name='SIMULATION',
            TRAIN = None, m_dim=1, w_nois=0, sigm_add=0):

        self.seed = seed
        self.Nsim = Nsim
        self.Nsimseed = Nsimseed
        self.T = T
        self.D = D
        self.Ninf =Ninf
        self.ustim = ustim
        self.scalw = scalw
        self.resol = resol
        self.mean = mean
        self.s = np.tile(ustim, np.int(T/2))

        self.beta = beta
        self.sig_base = sig_base
        self.BASE = BASE
        self.up=up
        self.DOWN = DOWN
        self.SIGM = SIGM
        self.Ninf = Ninf 
        self.Ninact = Ninact
        if TRAIN is None:
            self.TRAIN = np.array([T-10], dtype='int') 
        else:
            self.TRAIN = TRAIN
        self.DECODER = None
        self.name = name
        self.m_dim = m_dim
        self.w_nois = w_nois
        self.sigm_add = sigm_add

    # DECODING SIMULATION
    def simulation(self, path, compMCML=False, compMG=False, compRG=False,compSO=False, want_relsigm=True, multiunit=False):
        for nnsim in range(self.Nsimseed, self.Nsim+self.Nsimseed):
            for ddii in range(len(self.D)):
                for dd in range(len(self.DOWN)):
                    down = self.DOWN[dd]
                    for oo in range(len(self.BASE)):
                        offset_base = self.BASE[oo]
                        for ninf in self.Ninf:
                            for mmean in self.mean:
                                if mmean ==0:
                                    mean_out = False
                                if mmean ==1:
                                    mean_out = True
                                for ss in range(len(self.SIGM)):
                                    POP = simulate_response()
                                    if mmean <2:
                                        sig_m = self.SIGM[ss]
                                    else:
                                        sig_m = 0
                                    POP.par(self.D[ddii], self.T, sig_m, self.beta, self.up, down,
                                            self.sig_base, offset_base, ninf, Ninact = self.Ninact[ddii], seed = nnsim,
                                            m_dim=self.m_dim)
                                    POP.create_population(w_nois=self.w_nois)
                                    if mmean==2:
                                        mean_out = -(self.SIGM[ss]**2*POP.w**2/2)
                                        mean_out[POP.w==0] = 0
                                    POP.response(self.s, self.ustim, mean_out=mean_out, want_relsigm=want_relsigm,
                                                 sigm_add=self.sigm_add, multiunit=multiunit)
                                    for ttrain in range(len(self.TRAIN)):
                                        train = np.arange(0, self.TRAIN[ttrain])
                                        test = np.arange(self.TRAIN[ttrain],self.T)
                                        self.DECODER = POP.decode(train, test, self.s, self.ustim,
                                                             DECODER=self.DECODER,
                                                                  compMCML=compMCML, compMG=compMG, compRG=compRG,
                                                                  compSO=compSO)

            if path is not None:
                print('saved itself after iteration ' + np.str(nnsim))
                pickle.dump(self, open(path + self.name + '.pk', 'wb'))


        self.DECODER = self.DECODER.reset_index()
        self.DECODER = self.DECODER.drop('index', axis=1)
        if path is not None: pickle.dump(self, open(path + self.name + '.pk', 'wb'))

    def visualization(self, normvar, saveplot=False, downind=0, figpath=None):
        if normvar & (np.max(self.DECODER.sigma)<90):
            self.DECODER.sigma = norm_sig(self)
        # set style and color
        my_cols = {'ML': 'black', 'ML-sign-only': 'black', 'decode': 'green', 'MC-ML': 'red',
                   'modulator-guided': 'green', 'modulator-guided_MLthresh': 'black', 'modulator-guided_modMLthresh': 'black',
                   'rate-guided': 'blue', 'SO': 'grey', 'MM-ML': 'orange', 'sum': 'black'}
        sb.set_context('poster', rc={'legend.frameon':False,
                               'axes.labelsize': 30,
                               'axes.titlesize': 0,
                               'legend.fontsize': 24
                               })




        if len(self.SIGM)>1:
            # encoding-decoding tradeoff
            plt.rc('xtick', labelsize=25)
            plt.rc('ytick', labelsize=25)
            fig, ax = plt.subplots(1,1, figsize=(10, 8))
            splot = sb.lineplot(x='sigma', y='accuracy', hue='decoder', data=self.DECODER[((self.DECODER['mean_out']<0)|
                                                                                ((self.DECODER['mean_out']==0)&(self.DECODER['sigma']==0)))&
                                                                          (self.DECODER['down']==self.DOWN[downind])&(
                                                                         (self.DECODER['decoder']=='MC-ML')|
                                                                         (self.DECODER['decoder']=='decode'))],
                        ci=95, palette=my_cols, ax=ax)
            sax = splot.axes
            sax.set_xscale('log')
            sb.despine()
            plt.xlabel('modulator strength (% variance)')
            plt.ylabel('accuracy (% correct)')
            plt.ylim(50,101);

            if saveplot:
                plt.savefig(figpath + "tradeoff.svg")
                plt.savefig(figpath + "tradeoff.pdf")

            # decoder performance over modulator strength
            if np.sum(self.mean==0)>0:
                plt.figure(figsize=(10, 8))
                splot = sb.lineplot(x='sigma', y='accuracy', hue='decoder',
                            data=self.DECODER[(self.DECODER['mean_out'] == 0) &  
                                         (self.DECODER['down'] == self.DOWN[downind]) & (
                                                 (self.DECODER['decoder'] == 'MC-ML') |
                                                 (self.DECODER['decoder'] == 'MM-ML') |
                                                 (self.DECODER['decoder'] == 'modulator-guided') |
                                                 (self.DECODER['decoder'] == 'rate-guided') |
                                                 (self.DECODER['decoder'] == 'SO')
                                         )],
                            ci=95, palette=my_cols, legend=False)
                plt.ylim(50, 101)
                plt.title('mean and variance')
                sax = splot.axes
                sax.set_xscale('log')
                plt.xlabel('modulator strength (% variance)')
                plt.ylabel('accuracy (% correct)')
                sb.despine()
                if saveplot:
                    plt.savefig(figpath + "accuracy_meanvariance.svg")
                    plt.savefig(figpath + "accuracy_meanvariance.pdf")
            if np.sum(self.mean == 1) > 0:
                plt.figure(figsize=(10, 8))
                splot = sb.lineplot(x='sigma', y='accuracy', hue='decoder', data=self.DECODER[((self.DECODER['mean_out'] < 0) | (
                            (self.DECODER['mean_out'] == 0) & (self.DECODER['sigma'] == 0))) &  
                                                                              (self.DECODER['down'] == self.DOWN[downind]) & (
                                                                                      (self.DECODER['decoder'] == 'MC-ML') |
                                                                                      (self.DECODER['decoder'] == 'MM-ML') |
                                                                                      (self.DECODER['decoder'] == 'modulator-guided') |
                                                                                      (self.DECODER['decoder'] == 'rate-guided') |
                                                                                      (self.DECODER['decoder'] == 'SO')
                                                                              )],
                            ci=95, palette=my_cols)
                plt.ylim(50, 101);
                plt.title('variance')
                plt.xlabel('modulator strength (% variance)')
                plt.ylabel('accuracy (% correct)')
                sax = splot.axes
                sax.set_xscale('log')
                sb.despine()

                if saveplot:
                    plt.savefig(figpath + "accuracy_variance.svg")
                    plt.savefig(figpath + "accuracy_variance.pdf")
            if np.sum(self.mean == 2) > 0:
                plt.figure(figsize=(10, 8))
                splot = sb.lineplot(x='mean_out', y='accuracy', hue='decoder',
                            data=self.DECODER[(self.DECODER['mean_out'] > 0) &  
                                         (self.DECODER['down'] == self.DOWN[downind]) & (
                                                 (self.DECODER['decoder'] == 'MC-ML') |
                                                 (self.DECODER['decoder'] == 'MM-ML') |
                                                 (self.DECODER['decoder'] == 'modulator-guided') |
                                                 (self.DECODER['decoder'] == 'rate-guided') |
                                                 (self.DECODER['decoder'] == 'SO')
                                         )],
                            ci=95, palette=my_cols)
                plt.ylim(50, 101)
                plt.title('mean')
                sax = splot.axes
                plt.xlabel('modulator strength (% variance)')
                plt.ylabel('accuracy (% correct)')
                sax.set_xscale('log')
                sb.despine()
                if saveplot:
                    plt.savefig(figpath + "accuracy_mean.svg")
                    plt.savefig(figpath + "accuracy_mean.pdf")

def norm_sig(obj, plotit=False):
    fr_base = np.array(obj.DECODER['FR_base'], dtype=float)
    sigma = np.array(obj.DECODER['sigma'], dtype=float)
    down = np.array(obj.DECODER['down'], dtype=float)
    # convert to percentage of variance:
    mu_stim = .5 * (np.exp(fr_base * obj.up) + np.exp(fr_base * down))
    V_stim = .5 * (np.exp(fr_base * obj.up) ** 2 + np.exp(fr_base * down) ** 2) - mu_stim ** 2
    V_mod = np.exp(2 * sigma ** 2 * obj.scalw ** 2) - np.exp(sigma ** 2 * obj.scalw ** 2)

    rel_sigma = np.sqrt(np.exp(2 * sigma ** 2 * obj.scalw ** 2) -
                                    np.exp(sigma ** 2 * obj.scalw ** 2)) / \
                            np.sqrt(mu_stim + V_stim + np.exp(2 * sigma ** 2 * obj.scalw ** 2) -
                                    np.exp(sigma ** 2 * obj.scalw ** 2)) * 100
    if plotit:
        plt.figure(figsize=(8,5))
        plt.plot(sigma, rel_sigma, 'o')
        plt.xlabel('modulator strength')
        plt.ylabel('% of variance from modulator')
        plt.title('example transformation from absolute to relative mod-var')

    return rel_sigma




def vis_Ninf(SIM, types, yy=None, YY=np.array([0]), normsig=False, log=False, fit_p = 2):
    if normsig:
        if any(SIM.SIGM > 50):
            print('already normalized')
        else:
            SIM.DECODER.sigma = norm_sig(SIM)
            SIM.SIGM = np.unique(SIM.DECODER.sigma)
    sim = SIM.DECODER[SIM.DECODER.decoder == types[yy]]

    # change variable scaling for interpretability
    sim.Ninf = sim.Ninf / SIM.D * 100  # from decoder --> already multiplied by 2!
    rel_ninf = SIM.Ninf * 2 / SIM.D * 100
    SIGM = np.copy(SIM.SIGM)
    if log: SIGM[SIGM <= 0] = .01

    # compute mean perf
    sd = sim.groupby(['Ninf', 'sigma'])['accuracy'].mean().reset_index()
    dec = sd.pivot(index='Ninf', values='accuracy', columns='sigma')

    fig, ax = plt.subplots(1, 3, figsize=(20, 5))
    sb.heatmap(dec, cmap="YlGnBu", ax=ax[0])
    ax[0].set_title(types[yy])

    if log:
        lSIGM = np.log(SIGM)
    else:
        lSIGM = np.copy(SIGM)
    cmap = cm.coolwarm
    for yy0 in YY:
        sim = SIM.DECODER[SIM.DECODER.decoder == types[yy0]]
        # change variable scaling for interpretability
        sim.Ninf = sim.Ninf / SIM.D * 100  # from decoder --> already multiplied by 2!
        rel_ninf = SIM.Ninf * 2 / SIM.D * 100
        SIGM = np.copy(SIM.SIGM)
        if log: SIGM[SIGM <= 0] = .01

        # compute mean perf
        sd = sim.groupby(['Ninf', 'sigma'])['accuracy'].mean().reset_index()
        dec = sd.pivot(index='Ninf', values='accuracy', columns='sigma')

        for nn in range(len(SIM.Ninf)):
            ax[1].plot(dec.loc[rel_ninf[nn], :], '.', color=cmap(nn / len(SIM.Ninf)), label=rel_ninf[nn])
            p2 = np.poly1d(np.polyfit(lSIGM, dec.loc[rel_ninf[nn], :], fit_p))
            fit = p2(lSIGM[1:])
            ax[1].plot(SIGM[1:], fit, '-', color=cmap(nn / len(SIM.Ninf)))
            ax[1].plot(SIGM[np.argmax(fit)-1], np.max(fit), 'o', color=cmap(nn / len(SIM.Ninf)))
    ax[1].legend()
    ax[1].set_xlabel('modulator strength')
    ax[1].set_ylabel('accuracy')
    ax[1].set_ylim(45, 105)
    if log: ax[1].set_xscale('log')

    X, Y = np.meshgrid(SIGM, rel_ninf)
    Z = np.copy(dec)
    ax[2].contourf(X, Y, Z, levels=np.arange(40, 101, 1)) 

    ax[2].set_xlabel('sigma')
    ax[2].set_ylabel('percentage informative')
    ax[2].set_title(types[yy])

    for nn in range(len(SIM.Ninf)):
        ind = sd[sd.Ninf == rel_ninf[nn]].accuracy.idxmax()
        ax[0].plot(np.where(SIM.SIGM == sd.sigma[ind])[0] + .5, nn + .5, 'or')
        ax[2].plot(SIM.SIGM[SIM.SIGM == sd.sigma[ind]], rel_ninf[nn], 'or')

    if log: ax[2].set_xscale('log')



