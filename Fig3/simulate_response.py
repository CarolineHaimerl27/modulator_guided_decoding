# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 14:20:57 2018

@author: caroline
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import minimize
from convergence import FLD_perf


class simulate_response:
    
    def par(self, D, T, sig_m, beta, up, down, sig_base, offset_base, Ninf, resol=1, Ninact=np.zeros(1),
            seed=0, m1=None, m2=None, w = None, relvar = None, m_dim=1):
        self.D = D
        self.T = T
        self.sig_m = sig_m # modulation variance
        self.beta = beta # if 0 --> w is rescaled informativeness, if 1 w is unrelated to informativeness
        self.resol = resol # in how many steps is the modulation discretized within a trial (e.g. how many stimulus presentation wihtin a trial)
        self.up = up # increased mean response (log)
        self.down = down # decreased mean response (log)
        self.sig_base = sig_base # standard deviation of baseline firing rate
        self.offset_base = offset_base #  mean of baseline firing rate
        self.Ninf = Ninf # Number of informative neurons
        self.seed = seed
        self.sig_m = sig_m
        self.dec = None
        self.mRES = None
        self.Ninact = Ninact
        self.m1 = m1
        self.m2 = m2
        self.w = w
        self.relvar = relvar
        self.m_dim = m_dim
        
    def set_baseline(self, scal_inact = 1):
        np.random.seed(self.seed)
        tmp = np.random.random(self.D)*self.sig_base+self.offset_base
        if self.Ninact>0:
            tmp[self.Ninf:(self.Ninf+self.Ninact)] = np.random.random(self.Ninact)*scal_inact
        if np.sum(tmp<0)>0:
            print(np.sum(tmp<0), ' corrected cases of negative baseline firing rate')
        self.baseline = np.abs(tmp)

    def response_mean(self):
        self.M = np.array([self.baseline, self.baseline]).T
        for ii in range(self.Ninf):
            self.M[ii,:] = self.M[ii,:]*np.array([self.down,self.up])
            self.M[-(ii+1),:] = self.M[-(ii+1),:]*np.array([self.up, self.down])
        # save the mean stimulus response
        self.m1 = np.exp(self.M[:,0])
        self.m2 = np.exp(self.M[:, 1])
        
    def create_population(self, w_nois = 0):
        self.set_baseline()
        self.response_mean()
        self.dec_ML()
        self.w = np.abs(self.dec+np.random.randn(self.D)*w_nois)
        if self.beta==1:
            ind = np.argsort(np.abs(self.dec))
            self.w[ind] = self.w[ind][::-1]
        if self.beta==.5:
            self.w = np.random.choice(self.w, self.D)


    def response(self, s, ustim, seed_sample=None, mean_out=False, sig_m=None, want_relsigm=True, sigm_add=0, multiunit=False):
        ####################################
        ########## encoding model ##########
        ####################################
        if sig_m is None:
            sig_m = np.copy(self.sig_m)

        # STIMULUS RESPONSE
        RES = np.zeros([self.D, self.T])*np.nan
        for tt in range(self.T): 
            RES[:,tt] = np.exp(self.M[:,np.where(s[tt]==ustim)[0]].T[0])

        self.d_orig = np.abs(self.m1-self.m2)/\
                np.sqrt(.5*(self.m1+self.m2)) # dprime expected given only Poisson process
            
        # MODULATION
        if seed_sample is None:
            np.random.seed(self.seed)
        else:
            np.random.seed(seed_sample)
        if self.m_dim>1:
            self.m = np.random.randn(self.T * self.resol * self.m_dim).reshape(self.T * self.resol, self.m_dim) * sig_m
            self.w = np.concatenate((self.w, self.w)).reshape(self.m_dim, self.D)
        else:
            self.m = np.random.randn(self.T * self.resol)* sig_m

        # correction for mean increase
        if type(mean_out) is bool:
            if mean_out ==False:
                self.meanmod = np.zeros([self.D])
            if mean_out == True:
                if self.m_dim==1:
                    self.meanmod = (sig_m**2*self.w**2/2)
                    self.meanmod[self.w==0] = 0
                else:
                    self.meanmod = np.sum(sig_m**2*self.w**2/2, axis=0)
                    self.meanmod[self.w[0,:]==0] = 0
        else:
            self.meanmod = np.copy(mean_out)

        # SPIKING
        if self.m_dim==1:
            tmp = RES*np.exp(np.outer(self.w,self.m).T-[self.meanmod,]*self.T).T
        else:
            tmp = RES*np.exp(self.w.T.dot(self.m.T).T-[self.meanmod,]*self.T).T
        # add noise
        tmp = tmp + np.random.randn(self.D*self.T).reshape(self.D, self.T)*(sigm_add)
        # guarantee that still positive
        tmp[tmp<0] = 0
        if np.sum(tmp>10000)>0:
            print('exp bound hit, times', np.sum(tmp>10000))
            tmp[tmp>10000] = 10000
        self.mRES = np.random.poisson(tmp)

        # informativeness measures
        self.d_mod = np.abs(np.mean(self.mRES[:,s==ustim[0]], axis=1)-np.mean(self.mRES[:,s==ustim[1]], axis=1))/\
                np.sqrt(.5*(np.var(self.mRES[:,s==ustim[0]], axis=1)+np.var(self.mRES[:,s==ustim[1]], axis=1))) # dprime expected given only Poisson process

        if self.m_dim==1:
            self.S_FLD = FLD_perf(self.dec, sig_m, np.array([self.m1, self.m2]), self.w, self.D, meanout=mean_out)
        if want_relsigm:
            self.relvar = self.rel_SIGM(np.array([self.sig_m]))
        if multiunit:
            tmp = np.zeros([np.array(self.D/2).astype('int'), self.T])*np.nan
            wtmp = np.zeros([np.array(self.D/2).astype('int')])*np.nan
            m1 = np.zeros([np.array(self.D/2).astype('int')])*np.nan
            m2 = np.zeros([np.array(self.D/2).astype('int')])*np.nan
            nnii=0
            for nn in range(0, self.D,2):
                tmp[nnii,:]= np.sum(self.mRES[nn:(nn+2),:],axis=0)
                wtmp[nnii] = np.sum(self.w[nn:(nn+2)])
                m1[nnii] = np.sum(self.m1[nn:(nn+2)])
                m2[nnii] = np.sum(self.m2[nn:(nn+2)])
                nnii+=1
            self.w = np.copy(wtmp)
            self.meanmod = (sig_m**2*self.w**2/2)
            self.meanmod[self.w==0] = 0
            self.mRES = np.copy(tmp)
            self.m1 = np.copy(m1) 
            self.m2 = np.copy(m2)
            self.dec_ML()
            self.D = np.array(self.D/2).astype('int')



    def vis_response(self, s, ustim):
        fig = plt.figure(figsize=(17,10))
        ax1 = plt.subplot2grid((2,2), (0,0), colspan=2)
        ax2 = plt.subplot2grid((2,2), (1,0), colspan=1)
        ax3 = plt.subplot2grid((2,2), (1,1), colspan=1)
        im = ax1.imshow(self.mRES)
        ax1.set_title('simulated activity')
        ax1.set_xlabel('trials')
        ax1.set_ylabel('neuron')
        fig.colorbar(im, ax=ax1)
        ax2.plot(self.m1, 'r', label='lambda stim1 (no mod)')
        ax2.plot(self.m2, 'b', label='lambda stim2 (no mod)')
        ax2.plot(np.mean(self.mRES[:,ustim[0]==s], axis=1), '--r', label='mean stim 1 (with mod)')
        ax2.plot(np.mean(self.mRES[:,ustim[1]==s], axis=1), '--b', label='mean stim 2 (with mod)')
        ax2.plot(np.mean(self.mRES, axis=1), 'k', label='overall mean (with mod)')
        ax2.plot(np.mean(self.mRES[:,ustim[0]==s], axis=1), '--r')
        ax2.plot(np.mean(self.mRES[:,ustim[1]==s], axis=1), '--b')
        ax2.plot(np.abs(self.m1-self.m2), 'g', label='diff pf stim response(with mod)')
        ax2.legend()
        ax2.set_xlabel('neurons')
        ax2.set_ylabel('mean response')
        ax3.plot(self.d_orig, label='d before mod')
        ax3.plot(self.d_mod, label='d with mod')
        ax3.set_title('informative neurons (d>.5) %.2f perc' %(np.mean(self.d_orig>.5)))
        ax3.legend()
        ax3.set_xlabel('neurons')
        ax3.set_ylabel("d'")
        plt.show()

    def rel_SIGM(self, sigma, notarray=True, m1 = None, m2 = None, w = None):
        if notarray:
            sigma = np.array(sigma)
        if m1 is None:
            m1 = self.m1
        if m2 is None:
            m2 = self.m2
        if w is None:
            w = self.w
        tmp = np.zeros(len(sigma))
        ms =  (m1 + m2) / 2
        m2s =  (m1**2 + m2**2) / 2
        for ss in range(len(sigma)):
            total = ms+m2s*np.exp(sigma[ss]**2*w**2-1)
            mod_dep = np.exp(sigma[ss]**2*w**2)-1
            tmp[ss] = np.max(mod_dep/total)
        return tmp*100

    ##########################################################################
    ######################## DECODERS ########################################
    ##########################################################################

    # Poisson ML
    def dec_ML(self):
        dec = self.m1/self.m2
        dec[dec<=0] = 0.00001 # this should really never be the case
        self.dec = np.log(dec)
        self.thresh = -np.sum(self.m1-self.m2)
        

    # MC-ML
    def dec_modML(self, test):
        if self.m_dim==1:
            tmp = np.outer(self.w, self.m[test]).T
        else:
            tmp = self.w.T.dot(self.m[test,:].T).T
        self.thresh_tt = -np.sum(np.multiply(np.exp(tmp-[self.meanmod,]*len(test)).T,\
                            np.array([(self.m1-self.m2),]*len(test)).T), axis=0)

    # signs
    def dec_SO(self, train, s, ustim):
        ind1 = np.where(s[train] == ustim[0])[0]
        ind2 = np.where(s[train] == ustim[1])[0]
        if (len(ind1))>1 &(len(ind2)>1):
            sm1 = np.sign(np.mean(self.mRES[:,ind1], axis=1)-np.mean(self.mRES[:,ind2], axis=1))
        else:
            sm1 = np.sign(self.mRES[:,ind1]-self.mRES[:,ind2])
        return sm1

    # modulator-guided decoder
    def dec_MG(self, train, test, est_thresh=True):
        # 1. direction of decoding weights
        if self.m_dim == 1:
            dec_m = (self.mRES[:,train].dot(self.m[train])) / len(train)
        else:
            dec_m = np.sum((self.mRES[:, train].dot(self.m[train,:])), axis=1) / (len(train)*self.m_dim)
        # 2. unbias
        if self.sig_m>0:
            estw = dec_m / (self.sig_m ** 2 * (self.m1 + self.m2) / 2)
        else:
            estw = dec_m / ((self.m1 + self.m2) / 2)
        # 3. apply signs
        self.dec_m = estw* self.dsigns

        # 4. find modulation-dependent threshold
        # estimate response to stimulus
        if est_thresh:
            meandiff = np.sum(np.abs(self.m1 - self.m2)[self.dsigns > 0]) / (self.Ninf)
            meandiff = (self.dsigns > 0) * meandiff
            meandiff[self.dsigns < 0] = np.sum(np.abs(self.m1 - self.m2)[self.dsigns < 0]) / (self.Ninf)
            if self.m_dim == 1:
                tmp = np.outer(self.w, self.m[test]).T
            else:
                tmp = self.w.T.dot(self.m[test,:].T).T
            self.thresh_m = -np.sum(np.multiply(np.exp(tmp-[self.meanmod,]*len(test)).T,
                                                np.array([(np.sign(self.dec)*meandiff), ] * len(test)).T), axis=0)
        else:
            self.thresh_m = np.copy(self.thresh_tt)

    # rate-guided decoder
    def dec_RG(self, train, test, s):
        # 1. direction
        dec_r = np.mean(self.mRES[:,train], axis=1)
        # 2. weight to match ML so that the ML threshold can be used
        dec_r = dec_r/np.sqrt(np.sum((dec_r**2))) * np.sqrt(np.sum((self.dec**2)))
        # 3. signs
        self.dec_r = dec_r * self.dsigns


    ##########################################################################
    ######################## ACCURACY ########################################
    ##########################################################################
        
    def test_decoder(self, test, dec, thresh, s, ustim):
        tmp = np.mean((s[test]==ustim[0])==((dec.dot(self.mRES[:,test])+thresh)>0))
        return tmp*100

    def decode(self, train, test, s, ustim, DECODER=None,
               compMCML=False, compMG=False, compRG=False, compSO=False):
        if self.relvar is None:
            relvar = self.sig_m
        else:
            relvar = self.relvar[0]

        cols = ['decoder', 'sigma', 'relsigma', 'training', 'accuracy', 'Ninf', 'FR_base', 'down', 'mean_out', 'Ninact', 'MSE_direction', 'perc_sign']
        if DECODER is None:
            DECODER = pd.DataFrame([], columns=cols)

        # compute signs
        self.dsigns = self.dec_SO(train, s, ustim)

        # MM-ML
        DECODER = DECODER.append(pd.DataFrame([['MM-ML', self.sig_m, relvar, len(train),
                                                self.test_decoder(test, self.dec,
                                                                  -np.sum((self.m1 - self.m2) * np.exp((self.sig_m ** 2 * self.w ** 2) / 2 - self.meanmod)),
                                                                                                   s, ustim),
                                                self.Ninf * 2, self.offset_base, self.down, -self.meanmod[0], self.Ninact, np.array(0), np.array(100)]],
                                              columns=cols))

        if compMCML|compMG:
            # MC-ML
            self.dec_modML(test)

            DECODER = DECODER.append(pd.DataFrame([['MC-ML', self.sig_m, relvar, len(train),
                                                    self.test_decoder(test, self.dec, self.thresh_tt, s, ustim),
                                                    self.Ninf * 2, self.offset_base, self.down, -self.meanmod[0],
                                                    self.Ninact, np.array(0), np.array(100)]], columns=cols))
        # modulator-guided
        if compMG:
            self.dec_MG(train, test)

            DECODER = DECODER.append(pd.DataFrame([['modulator-guided', self.sig_m, relvar, len(train),
                                                    self.test_decoder(test, self.dec_m, self.thresh_m, s, ustim),
                                                    self.Ninf * 2, self.offset_base, self.down, -self.meanmod[0],
                                                    self.Ninact, np.mean((self.dec_m-self.dec)**2),
                                                    np.mean(np.sign(self.dec_m[np.abs(self.dec)>0])==np.sign(self.dec[np.abs(self.dec)>0]))*100]], columns=cols))
        if compRG:
            # rate-guided
            self.dec_RG(train, test, s)

            DECODER = DECODER.append(pd.DataFrame([['rate-guided', self.sig_m, relvar, len(train),
                                                    self.test_decoder(test, self.dec_r, self.thresh, s, ustim),
                                                    self.Ninf * 2, self.offset_base, self.down, -self.meanmod[0],
                                                    self.Ninact, np.mean((self.dec_r-self.dec)**2), np.mean(np.sign(self.dec_r[np.abs(self.dec)>0])==np.sign(self.dec[np.abs(self.dec)>0]))*100]], columns=cols))
        if compSO:
            # sign-only
            DECODER = DECODER.append(pd.DataFrame([['sign-only', self.sig_m, relvar, len(train),
                                                    self.test_decoder(test, self.dsigns / np.sqrt(np.sum(self.dsigns ** 2)) * np.sqrt(np.sum(self.dec ** 2)),
                                                                                                           0, s, ustim),
                                                    self.Ninf * 2, self.offset_base, self.down, -self.meanmod[0],
                                                    self.Ninact, np.array(0), np.mean(np.sign(self.dsigns[np.abs(self.dec)>0])==np.sign(self.dec[np.abs(self.dec)>0]))*100]], columns=cols))


        return DECODER




