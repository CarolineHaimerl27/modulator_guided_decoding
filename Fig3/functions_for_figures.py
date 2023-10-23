from simulate_decoding import simulate_decoding
from simulate_response import simulate_response
import numpy as np
from datetime import date
import pickle
import seaborn as sb
import matplotlib.pyplot as plt
from convergence import theo_var_what

# color scheme
my_cols = {'ML': 'black', 'ML-sign-only': 'black', 'MC-ML': 'red', \
           'modulator-guided': 'green',\
           'rate-guided': 'blue', 'sign-only': 'grey', 'MM-ML': 'orange', 'sum': 'black',}

#########################################################################################
######################## Figure 2 ######################################################
#########################################################################################
# Figure 2:Accuracy of sign estimation, for simulated data.A.Mean % correctly attributed signs for informativeneurons
# as a function of number of training trials with varying modulator strength (percentage of spike countvariance of the
# informative neurons accounted for by the modulator). Decoding signs are learned within a fewtens of trials.B. Mean
# performance of RG and SO decoders as the number of inactive neurons is increased. TheRG decoder downweights inactive
# neurons, thus allowing it to maintain better performance than the SO decoder.

def Fig2A_sim(path='simulations/', name = None, Nsim=100, Nsimseed=0, SIGM=None, D=None, Ninf=None):
    if name is None: 'SIMULATION_learning_signs_'+str(date.today())
    # training
    print('running: '+name)

    if D is None: D = np.array([1000])
    if Ninf is None: Ninf = np.array([6])
    if SIGM is None: SIGM = np.array([0, .01, .4, .7, 1.5])
    TRAIN = np.concatenate([np.arange(2, 50, 10), np.arange(50, 100, 10)])
    m_dim=1
    w_nois=0
    sigm_add=0

    SIM = simulate_decoding()
    SIM.par(Nsim=Nsim, Nsimseed=Nsimseed, D = D, Ninf = Ninf, Ninact = D-50,
            SIGM=SIGM, mean = np.array([1], dtype=int), # mean =1 means that we have the mean corrected model
            name=name, TRAIN=TRAIN,
            m_dim=m_dim, w_nois=w_nois, sigm_add=sigm_add)
    SIM.simulation(path=path, compMCML=False, compMG=False, compRG=False,compSO=True)

def Fig2A_plot(SIM=None, name=None, savefig=False, relsigma=True):
    if SIM is None:
        if name is None:
            print('error, give simulation object or path/name')
        else:
            SIM = pickle.load(open(name + '.pk', 'rb'))
    SIM.DECODER['perc_sign']= np.array(SIM.DECODER['perc_sign'], dtype='float')
    if relsigma:
        plt.figure(figsize=(5,2))
        bins = np.array([-.01, .01, .25, 1.25])
        for bb in range(len(bins)):
            plt.plot([bins[bb], bins[bb]], [0, 50], '--k')
        plt.hist(SIM.DECODER['relsigma'], 100, density=True)
        plt.xlabel('rel mod strength of simulations')
        plt.ylabel('frequency')
        plt.title('rel sigma and discretization')
        hist = np.digitize(SIM.DECODER['relsigma'], bins)
        dbins = np.diff(bins)
        SIM.DECODER['relsigma'] = bins[hist]-dbins[hist-1]/2
        var = 'relsigma'
    else:
        var='sigma'
    pal = sb.cubehelix_palette(len(np.unique(SIM.DECODER[var])), start=2, rot=0, dark=0, light=.85, reverse=True)
    plt.rc('xtick', labelsize=25)
    plt.rc('ytick', labelsize=25)
    plt.figure(figsize=(10,8))
    plt.plot([0, 100], [90, 90], '--', color='grey')
    plt.ylim(50,100)
    plt.yticks(np.array([50, 70,90], dtype='int'),np.array([50, 70,90], dtype='int')) 
    sb.lineplot(x='training', y='perc_sign', hue=var, data=SIM.DECODER[(SIM.DECODER['decoder']=='sign-only')],
                palette=pal, markers=True)
    sb.despine()
    if savefig: plt.savefig(name+".svg")
    if savefig: plt.savefig(name+".pdf")

def Fig2B_sim(path='simulations/', name = None, Nsim=100, Nsimseed=0, alldec=False):
    if name is None: name = 'SIMULATION_addNinact_' + str(date.today())
    print('running: ' + name)

    D = np.array([50, 1100, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000,
                  10000])
    Nact = 50
    Ninf = np.array([6]) # will be doubled, so total of 12 informative neurons
    SIGM = np.array([2])
    w_nois = 0
    sigm_add = 0
    m_dim=1

    SIM = simulate_decoding()
    SIM.par(Nsim=Nsim, Nsimseed=Nsimseed, D=D, Ninf=Ninf, Ninact=D - Nact,
            SIGM=SIGM, mean=np.array([1], dtype=int),  # mean =1 means that we have the mean corrected model
            name=name, m_dim=m_dim, w_nois=w_nois, sigm_add=sigm_add)

    SIM.simulation(path=path, compMCML=alldec, compMG=alldec, compRG=True, compSO=True, want_relsigm=True)

def Fig2B_plot(SIM=None, name=None, savefig=False):
    if SIM is None:
        if name is None:
            print('error, give simulation object or path/name')
        else:
            SIM = pickle.load(open(name + '.pk', 'rb'))
    plt.figure(figsize=(10,8))
    plt.plot([0, 10000], [50, 50], '--', color='grey')
    sb.lineplot(x='Ninact', y='accuracy', hue='decoder', data=SIM.DECODER,\
               ci=95, palette=my_cols, markers=True)
    sb.despine()
    if savefig: plt.savefig(name+'.svg')

#########################################################################################
######################## Figure 3 ######################################################
#########################################################################################
# Figure 3:Comparison of decoders.A.Comparison of decoding accuracy of different decoders on simulateddata. Curves
# indicate mean accuracy, and its 95%confidence interval. We simulated5000cells in total, of which50were active cells
# and of those12(24%of active) were informative cells. Baseline firing rates were set similarfor all active neurons.
# B.Increasing modulator strength has opposite effects on encoding and decoding accuracy.It will decrease the FLD ratio
# (encoding accuracy), but it will also increase decoding accuracy by decreasing thevariance of the unbiased estimate
# of decoding weights (1Var(|a(MG unbias)|)), these two effects jointly produce themaximum in accuracy of the MG-decoder.
# C.Performance of different decoders as a function of the number of informative neurons (in % informative neurons of
# active neurons). The strength of modulation was set fixed to the MG-optimal strength (see A). Other parameters are the
# same as in A.

def Fig3A_sim(path='simulations/', name = None, Nsim=100, Nsimseed=0,
              w_nois=0, sigm_add=0, m_dim=1, multiunit=False, nsigm=20,
             D = np.array([5000]), Nact = 50, Ninf = np.array([6])):
    # changes to...
    # w_nois will add gaussian noise to the relationship between modulation strength w and optimal decoding weights
    # sigm_add will add additive noise to the firing rate
    # m_dim sets the dimensionality of the modulator
    if name is None: name = 'SIMULATION_accuracy_' + str(date.today())
    print('running: ' + name)

    SIGM = np.logspace(-1,1,nsigm) 
    print('sigma is ', SIGM)
    w_nois = w_nois
    sigm_add = sigm_add
    m_dim = m_dim

    SIM = simulate_decoding()
    SIM.par(Nsim=Nsim, Nsimseed=Nsimseed, D=D, Ninf=Ninf, Ninact=D - Nact,
            SIGM=SIGM, mean=np.array([1], dtype=int),  # mean =1 means that we have the mean corrected model
            name=name, m_dim=m_dim, w_nois=w_nois, sigm_add=sigm_add)

    SIM.simulation(path=path, compMCML=True, compMG=True, compRG=True, compSO=True, want_relsigm=True,
                  multiunit=multiunit)

def Fig3A_plot(SIM=None, name=None, savefig=False, relsigmause=True, thresh_sigm=-1):
    if SIM is None:
        if name is None:
            print('error, give simulation object or path/name')
        else:
            SIM = pickle.load(open(name + '.pk', 'rb'))
    plt.figure(figsize=(10, 8))

    if relsigmause:
        relsigma = SIM.DECODER.groupby('sigma').mean().relsigma 
        sigma = np.unique(SIM.DECODER.sigma)
        for ss in range(len(sigma)):
            SIM.DECODER.relsigma[SIM.DECODER.sigma == sigma[ss]] = relsigma[sigma[ss]]

        sb.lineplot(x='relsigma', y='accuracy', hue='decoder',
                    data=SIM.DECODER.loc[SIM.DECODER.relsigma>thresh_sigm,:], errorbar=('ci', 95), palette=my_cols, markers=True)
        plt.xlabel('relative modulator strength')
    else:
        sb.lineplot(x='sigma', y='accuracy', hue='decoder',
                    data=SIM.DECODER.loc[SIM.DECODER.sigma>thresh_sigm,:], errorbar=('ci', 95), palette=my_cols, markers=True)
        plt.xlabel('modulator strength')
    sb.despine()
    plt.xscale('log')
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    
    if savefig & relsigmause:
        plt.savefig(name+"_3A_relsigma.pdf")
        plt.savefig(name+"_3A_relsigma.svg")
    elif savefig:
        plt.savefig(name+"_3A_sigma.pdf")
        plt.savefig(name+"_3A_sigma.svg")


def Fig3B_plot(SIM=None, name=None, name2=None, name3=None, name4=None, savefig=False, Nsim=10, thresh_sigm=-1, abssigmatoo=False):
    if SIM is None:
        if name is None:
            print('error, give simulation object or path/name')
        else:
            SIM = pickle.load(open(name + '.pk', 'rb'))
    np.random.seed(0)

    if name2 is not None:
        SIM2 = pickle.load(open(name2 + '.pk', 'rb'))
        SIM.DECODER = SIM.DECODER.append(SIM2.DECODER)
        print('sigma1: ', SIM.SIGM)
        print('sigma2: ', SIM2.SIGM)
        SIM.SIGM = np.unique(np.concatenate((SIM.SIGM, SIM2.SIGM)))
        print('sigma merged: ', SIM.SIGM)
    if name3 is not None:
        SIM3 = pickle.load(open(name3 + '.pk', 'rb'))
        SIM.DECODER = SIM.DECODER.append(SIM3.DECODER)
        print('sigma1: ', SIM.SIGM)
        print('sigma2: ', SIM3.SIGM)
        SIM.SIGM = np.unique(np.concatenate((SIM.SIGM, SIM3.SIGM)))
        print('sigma merged: ', SIM.SIGM)
    if name4 is not None:
        SIM4 = pickle.load(open(name4 + '.pk', 'rb'))
        SIM.DECODER = SIM.DECODER.append(SIM4.DECODER)
        print('sigma1: ', SIM.SIGM)
        print('sigma2: ', SIM4.SIGM)
        SIM.SIGM = np.unique(np.concatenate((SIM.SIGM, SIM4.SIGM)))
        print('sigma merged: ', SIM.SIGM)

    beta = 0
    sigma = SIM.SIGM[SIM.SIGM>thresh_sigm]
    ALL = np.zeros([len(sigma), 2]) * np.nan
    for ss in range(len(sigma)):
        print('doing sigma='+np.str(sigma[ss]))
        if sigma[ss]==0:
            continue
        PERFFLD = np.zeros(Nsim)
        for nnsim in range(Nsim):
            POP = simulate_response()
            POP.par(SIM.D[0], SIM.T, sigma[ss], beta, SIM.up, SIM.DOWN[0],
                    SIM.sig_base, SIM.BASE[0], SIM.Ninf[0], Ninact = SIM.Ninact[0],
                    seed = nnsim)
            POP.create_population()
            POP.response(SIM.s, SIM.ustim, mean_out=SIM.mean)
            const = theo_var_what(SIM.T, np.array([POP.m1, POP.m2]),
                                  sigma[ss], POP.w, plotit=False, unbiased=True)

            PERFFLD[nnsim] = POP.S_FLD
        ALL[ss,:] = np.array([np.mean(PERFFLD), np.sum(const)])

    dec = SIM.DECODER[(SIM.DECODER.decoder == 'modulator-guided')]
    perf = dec.groupby('sigma').mean().accuracy

    relsigma = SIM.DECODER.groupby('sigma').mean().relsigma 

    fig, ax = plt.subplots(3, 1, figsize=(7, 14))
    for aa in range(3):
        ax[aa].set_xscale('log')
    ax[1].set_yscale('log')
    ax[0].plot(relsigma[SIM.SIGM>thresh_sigm], ALL[:, 0], '-r', linewidth=3)
    ax[1].plot(relsigma[SIM.SIGM>thresh_sigm], 1 / ALL[:, 1], '-', linewidth=3, color='green')
    ax[2].plot(relsigma[SIM.SIGM>thresh_sigm], perf[SIM.SIGM>thresh_sigm], '-', color='green', linewidth=3)
    ax[0].set_ylabel('FLD')
    ax[1].set_ylabel('1/Var')
    ax[2].set_ylabel('accuracy')
    ax[2].set_xlabel('relative sigma')
    if savefig: plt.savefig(name + "_3B_relsigma.pdf")

    fig2, ax2 = plt.subplots(3, 1, figsize=(7, 14))

    for aa in range(3):
        ax2[aa].set_xscale('log')
    ax2[1].set_yscale('log')
    ax2[0].plot(SIM.SIGM[SIM.SIGM>thresh_sigm], ALL[:, 0], '-r', linewidth=3)
    ax2[1].plot(SIM.SIGM[SIM.SIGM>thresh_sigm], 1 / ALL[:, 1], '-', color='green', linewidth=3)
    ax2[2].plot(SIM.SIGM[SIM.SIGM>thresh_sigm], perf[SIM.SIGM>thresh_sigm], '-', color='green', linewidth=3)
    ax2[0].set_ylabel('SNR')
    ax2[1].set_ylabel('1/Var')
    ax2[2].set_ylabel('accuracy')
    ax2[2].set_xlabel('sigma')
    for aa in range(3):
        ax[aa].tick_params(axis='x', labelsize= 20)
        ax[aa].tick_params(axis='y', labelsize= 20)
        ax2[aa].tick_params(axis='x', labelsize= 20)
        ax2[aa].tick_params(axis='y', labelsize= 20)

    if savefig:
        plt.savefig(name+"_3B_sigma.pdf")
        plt.savefig(name + "_3B_sigma.svg")

    if abssigmatoo==False:
        plt.close()

    return fig, ax, fig2, ax2

def Fig3C_sim(path='simulations/', name = None, Nsim=100, Nsimseed=0):
    if name is None: name = 'SIMULATION_diffNinf_' + str(date.today())
    print('running: ' + name)
    D = np.array([5000])
    Nact = 50
    Ninf = np.array([2, 4, 6, 8, 10, 12, 14, 16]) # will be doubled
    SIGM = np.array([2])
    w_nois = 0
    sigm_add = 0
    m_dim = 1

    SIM = simulate_decoding()
    SIM.par(Nsim=Nsim, Nsimseed=Nsimseed, D=D, Ninf=Ninf, Ninact=D - Nact,
            SIGM=SIGM, mean=np.array([1], dtype=int),  # mean =1 means that we have the mean corrected model
            name=name, m_dim=m_dim, w_nois=w_nois, sigm_add=sigm_add)

    SIM.simulation(path=path, compMCML=True, compMG=True, compRG=True, compSO=True, want_relsigm=True)

def Fig3C_plot(SIM=None, name=None, savefig=False):
    if SIM is None:
        if name is None:
            print('error, give simulation object or path/name')
        else:
            SIM = pickle.load(open(name + '.pk', 'rb'))
    plt.figure(figsize=(10, 8))
    sb.lineplot(x='Ninf', y='accuracy', hue='decoder', data=SIM.DECODER, \
                ci=95, palette=my_cols, markers=True)
    sb.despine()
    if savefig: plt.savefig('accuracy_Nininf.pdf')

