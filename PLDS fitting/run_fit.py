import pickle
import numpy as np
from functions import runDatatoModel

# which dataset
count = 16

# path
path_get = '.../data/'
path_save = 'results/'

#################################################
################ PARAMETERS #####################
#################################################

######### FITTING PARAMETERS #################
MINDIM=1
MAXDIM=1 # CHANGE
Ncrossval= 1 # 3 # CHANGE
Nsamps=10000
# multiprocess?
multiproc = False
num_proc=  3# CHANGE28
# fitting: number of max iterations
maxiter = 5 # CHANGE 20

######### OPTIMIZED (FIXED) PARAMETERS #################
LAM=np.array([0], dtype='int') # np.array([0, .0001, .001, .01, .1, 1, 10], dtype='int')
N_timebins = 4 # time bins per 200ms stimulus presentation
aft_stim_windows= 1 # how many time bins am I estimating coefficients for (N_timebins+?)
coFR = 0 # minimum firing rate
direction = False # should stimulus direction be a predictor
# inter spike interval
isi_crit = False # use throwout criteria for ISI-violations
ISI_percmax = 10 # % violations are tolerated
ISI_minms=3 # below ISI_minms ms is considered a ISI violation


############ RUN ####################
DATAFILE = np.load(path_get+'DATAFILE_BR.npy', allow_pickle=True)
name = 'day_'+np.str(DATAFILE[count,0])+'_attention_'+\
       np.str(DATAFILE[count,1])+'_block_'+np.str(DATAFILE[count,2])
print(name, '\n')

file = runDatatoModel()
file.par(DATAFILE, path_save =path_save,
     name = name, simulate=False, notes=None)
###### load data #########

file.load_data(count, coFR = coFR, remov = True, path_get = path_get,
               ST_N_timebins = N_timebins, ST_aft_stim_windows = aft_stim_windows,
               ST_onoff_stim = False, ST_inc_direction = direction,
               isi_crit = isi_crit,
               code_kickout=np.array([9, 3, 2, 11, 10, 5, 8, 14, 6, 7, 12, 13, 14, 15]),
               change_to=True, check_adapt=False, behavior_kickout=False)
file.create_model(MINDIM=MINDIM, MAXDIM=MAXDIM, Ncrossval=Ncrossval,
                 scalQ=.01, scalQ0=.001, maxiter=maxiter, maxtim=1000, difflikthresh=.001,
                 upx0=True, upQ0=True, upQ=True, upA=True, upC=True, upB=True, Adiag=False, regA=False,
                 backtracking=True, backtrack_diff=0)
#### fit the SR model ####
print('**** fitting SR *****')

for ee in range(file.DtM.Ncrossval):
    file.fit_SR(ee=ee, Ttrain=None, method='Newt', CV_meth='median', tol=.0001,
                LAM=LAM, Ncross=10, plotit=False,
                comp_SR_pois=True, comp_SR_loglin=False)


################ set starting parameters ###################
file.starting_values(estA=np.array([.5]))

#### save whole file ######
pickle.dump(file, open(file.path_save + 'main_' + file.name + '.pk', 'wb'))
print('file saved at ', file.path_save)

#### fit the PLDS model ###
file.fit_PLDS(printit=True, multiproc=multiproc, num_proc=num_proc)

#### save whole file ######
pickle.dump(file, open(file.path_save + 'main_' + file.name + '.pk', 'wb'))

# compute informativeness
print('**** informativeness *****')
file.data.comp_informative()

#### save whole file ######
pickle.dump(file, open(file.path_save + 'main_' + file.name + '.pk', 'wb'))
