import numpy as np
import matplotlib.pyplot as plt

def theo_var_what(T, lam, sigma, w, plotit=False, unbiased=False):
    if unbiased==False: print('currently just unbiased version')
    const = np.mean(lam**2, axis=0)*np.exp(sigma**2*w**2)*(4*sigma**2*w**2+1)/(np.mean(lam, axis=0)**2*sigma**2)+ \
            (sigma ** 2 * w ** 2 + 1)/(np.mean(lam, axis=0)*sigma**2)-w**2
    if plotit:
        plt.figure()
        tt = np.arange(1,T)
        plt.plot(tt, np.outer(1/tt, const), '--', color='grey')
        plt.plot(tt, np.mean(np.outer(1/tt, const), axis=1), '-k')
        plt.xscale('log')
    return const

# compute predicted performance through FLD
def FLD_perf(d, sigma, lam, w, N, meanout=True):
    # assumes a binary s
    if meanout:
        nom = np.sum(d*np.diff(lam,axis=0))**2
        Cov_s0 = np.outer(lam[0,:], lam[0,:])*(np.exp(sigma**2*np.outer(w,w))-1)+np.eye(N)*lam[0,:]
        Cov_s1 = np.outer(lam[1,:], lam[1,:])*(np.exp(sigma**2*np.outer(w,w))-1)+np.eye(N)*lam[1,:]
    else:
        print('mean-in version not yet implemented - need to derive covariance')
        nom = np.sum(d*np.exp(sigma**2*w**2/2)*np.diff(lam,axis=0))**2
    dnom = d.dot(Cov_s0).dot(d)+d.dot(Cov_s1).dot(d)
    FLD = nom/dnom
    return FLD

# compute predicted performance through FLD
def d_perf(sigma, lam, w, meanout=True):
    # assumes a binary s
    if meanout:
        nom = np.diff(lam,axis=0)**2
        Cov_s0 = lam[0]**2*(np.exp(sigma**2*w**2)-1)+lam[0]
        Cov_s1 = lam[1]**2*(np.exp(sigma**2*w**2)-1)+lam[1]
    else:
        print('mean-in version not yet implemented - need to derive covariance')
    dnom = .5*(Cov_s0+Cov_s1)
    dprim = nom/dnom
    return dprim, nom, dnom



def sim_est_w(Nsim, T, N, sigma, lam, w, Ttrain, s, d):
    w_hat = np.zeros([len(Ttrain), N, Nsim]) * np.nan
    PERF_control = np.zeros([Nsim]) * np.nan
    PERF_w = np.zeros([len(Ttrain), Nsim]) * np.nan
    PERF_w_unb = np.zeros([len(Ttrain), Nsim]) * np.nan
    for nsim in range(Nsim):
        np.random.seed(nsim)
        m = np.random.randn(T) * sigma

        k = np.zeros([T, N]) * np.nan
        for tt in range(T):
            k[tt, :] = np.random.poisson(lam[s[tt], :] * np.exp(m[tt] * w - sigma ** 2 * w ** 2 / 2))

        for tt in range(len(Ttrain)):
            w_hat[tt, :, nsim] = np.sum(k[:Ttrain[tt], :].T * m[:Ttrain[tt]], axis=1) / Ttrain[tt]

        m = np.random.randn(T) * sigma
        ktest = np.zeros([T, N]) * np.nan
        for tt in range(T):
            ktest[tt, :] = np.random.poisson(lam[s[tt], :] * np.exp(m[tt] * w - sigma ** 2 * w ** 2 / 2))

        for tt in range(len(Ttrain)):
            thresh = -np.sum(np.diff(lam, axis=0) * np.exp(
                np.outer(m, w_hat[tt, :, nsim]) - w_hat[tt, :, nsim] ** 2 * sigma ** 2 / 2), axis=1)
            PERF_w[tt, nsim] = np.mean(((ktest.dot(w_hat[tt, :, nsim] * np.sign(d)) + thresh) < 0) == (s == 0))

            wunb = w_hat[tt, :, nsim] / (np.mean(lam, axis=0) * sigma ** 2)  # unbiased estimate of w
            thresh = -np.sum(np.diff(lam, axis=0) * np.exp(np.outer(m, wunb) - wunb ** 2 * sigma ** 2 / 2), axis=1)
            PERF_w_unb[tt, nsim] = np.mean(((ktest.dot(wunb * np.sign(d)) + thresh) < 0) == (s == 0))

        thresh = -np.sum(np.diff(lam, axis=0) * np.exp(np.outer(m, w) - w ** 2 * sigma ** 2 / 2), axis=1)
        PERF_control[nsim] = np.mean(((ktest.dot(d) + thresh) < 0) == (s == 0))

    return w_hat, PERF_w, PERF_w_unb, PERF_control

def norm(x):
    return (x-np.min(x))/(np.max(x)-np.min(x))

