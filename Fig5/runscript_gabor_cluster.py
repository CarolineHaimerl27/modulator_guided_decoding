import numpy as np
import torch
from ANN_modulator_mnist import traintest, produce_dataset, save_state_dict, create_mask
from ANN_modulator_mnist import MG_Model_multi as model


########## VARYING PARAMETERS ##########
# save and load
path_save = 'models/'
path_mnist = ''# '/Users/carolinehaimerl/Documents/GitReps/MNIST' # ''
name_pre = 'Gabor_v81'
name_tun = ''
train_pre = True
save_pre = True & train_pre
save_traj = True
train_model = {'MG': True, 'weights': True, 'gain': True, 'decoder': True}
print(' I am running version ', name_pre, name_tun, ' and I am pretraining: ', train_pre,
      ', tuning weights: ',   train_model['weights'], ', tuning gain: ',   train_model['gain'],
      ', tuning decoder: ',   train_model['decoder'], ', tuning stochastic gain: ',   train_model['MG'])
seed_pretrain, seed_retrain, seed_task = 100, 200, 300
seed_batch_pre, seed_batch_re, seed_batch_task = 5, 9, 13
######################################
############# TRAINING ###############
######################################
Tpar = {'Emax':1000, 'momentum':0.9, 'weight_decay':0, 'l1_lambda':0, 'optimizer_type': 'adam', 'flex_epoch': True,
        'backprop': True, 'mmsigma': 0, 'samp_rate': 1, 'route': False, 'mmean': 0,
        'Nsets':30, 'Nsamples':10, 'Nbatch':2, 'Nvalperc':0.5, 'lr':1e-4} #
np.save(path_save + 'results/tune_weights_model_' + name_pre +name_tun+  '_Tpar.npy', Tpar)
# pretraining
if train_pre:
    #Tpar_pre = {**Tpar, 'lr':1e-5, 'Nsets':1, 'Nsamples':10, 'Nbatch':2, 'Nvalperc':0.3}
    Tpar_pre = {**Tpar, 'lr':1e-4, 'Nsets':1, 'Nsamples':4000, 'Nbatch':2**5, 'Nvalperc':0.3, 'l1_lambda':0}
    np.save(path_save + 'results/pretrained_model_' + name_pre + '_Tpar.npy', Tpar_pre)
else:
    Tpar_pre = dict(enumerate(
        np.load(path_save + 'results/pretrained_model_' + name_pre + '_Tpar.npy', allow_pickle=True).flatten()))[0]
# task-training
Npar_tun = {}
np.save(path_save + 'results/tune_weights_model_' + name_pre +name_tun+ '_Npar.npy', Npar_tun)
if train_model['gain']|train_model['MG']:
    Tpar_gain = {**Tpar, 'mmean': 1, 'lr': 1e-3} #
    np.save(path_save + 'results/tune_gain_model_' + name_pre +name_tun+ '_Tpar.npy', Tpar_gain) # 1e-4
if train_model['MG']:
    Tpar_stoch = {**Tpar_gain, 'route': True, 'mmsigma': .1, 'samp_rate': 500}
    np.save(path_save + 'results/tune_stoch_model_' + name_pre +name_tun+  '_Tpar.npy', Tpar_stoch)

######################################
################ DATA ################
######################################
# paramters: bounds = from each spatial dimension the last 'bounds' elements are removed to avoid boundary effects
# so the total number of spatial positions will be (Npos-bounds*2)**2
if train_pre|(isinstance(Tpar['lr'], float)==False):
    Gpar_data = {'sig': .7, 'lam': 1.5, 'Norien': 10, 'Npos': 16, 'bounds':2, 'nois':.1,
                 'data_type': 'MNIST','seed0': seed_pretrain, 'pool_orien': None}
    if Gpar_data['data_type'] == 'gabor':
        ######### pre-training data ##########
        Gpar_data['I'] = (56) ** 2  # input dimension
    else:
        Gpar_data.update({'scal': 4, 'downscale': 2})
        Gpar_data['RFsize'] = int(28 / Gpar_data['downscale'])
    data, datatest, label, labeltest, _, _, _, _ = \
        produce_dataset(Gpar_data, pretraining=True, task=None, Tpar_pre=Tpar_pre, Tpar=None, path_mnist=path_mnist,
                        seed0=Gpar_data['seed0'], nsamplespre=Tpar_pre['Nsamples'], nsamplestun=None, testtoo=True)
    np.save(path_save + 'results/' + name_pre + '_Gpar_data.npy', Gpar_data)
else:
    Gpar_data = dict(enumerate(np.load(path_save + 'results/' + name_pre + '_Gpar_data.npy', allow_pickle=True).flatten()))[0]

######################################
############# TASK PAR ###############
######################################
Ttasks = 3 # number of tasks
LOC = np.array([0, 4, 9])# np.arange(Ttasks)#np.random.choice(int((Gpar_data['Npos']-Gpar_data['bounds']*2)**2), Ttasks, replace=False)
tasks, S1, S2 = np.arange(Ttasks), np.arange(Ttasks), np.arange(Ttasks)+1
S2[S2>9] = 0
dtype, device = torch.float, torch.device("cpu")
print('is GPU available? ', torch.cuda.is_available())
if torch.cuda.is_available(): device = torch.device("cuda:0")
# specify the task to train on after pretraing and how many non-task categories should show in the distractors
# plus the number of distractors and choce a different random seed
Gpar_data_task = {**Gpar_data}
Gpar_data_task.update({'data_type': 'gabor', 'Ndistr_cat': 8, \
                       'Ndistractors':  50, # maximum is 143 = int((Gpar_data['Npos']-Gpar_data['bounds']*2)**2-1),
                       'seed0':seed_task,
                       'Ttasks': Ttasks, 'LOC': LOC, 'S1': S1, 'S2': S2})

if (Gpar_data['data_type']!=Gpar_data_task['data_type']):
    print('mixed task-types, pretraining and task-training not the same task')
    #if Gpar_data_task['data_type']=='gabor':
        ######### pre-training data ##########
        #Gpar_data_task['I'] = (56)**2 # input dimension
    if Gpar_data_task['data_type'] == 'MNIST':
        Gpar_data_task.update({'scal': 2, 'downscale': 2})
        Gpar_data_task['RFsize'] = int(28 / Gpar_data_task['downscale'])

np.save(path_save + 'results/'+name_pre + name_tun+'_Gpar_data_task.npy', Gpar_data_task)

######################################
############# NETWORK ################
######################################
if train_pre:
    Gpar_net = {**Gpar_data, 'bounds': 0} #, 'Npos': 31}
    Gpar_net['coupl_redfact'] = 0  # should be a divisor of 'Npos'
    Gpar_net['coupl_offset'] = 0  # how coupling is centered relative to neurons PO (0=lies on it), in % of difference
    Gpar_net['coupl_sigma'] = [.1, .1, .1]  # sigma of gaussian coupling filter
    # 0.3	0.2	0.5
    np.save(path_save + 'results/'+ name_pre + '_Gpar_net.npy', Gpar_net)
    Npar_pre = {'N': Gpar_net['Norien']*Gpar_net['Npos']**2, 'mod_loc': 0, 'Hfact': 4, 'masktype': 'sparse',
                     'Nh': None, # 4*Gpar_net['Norien']*Gpar_net['Npos']**2,
                'Nhidden': 2, 'pool_fact_h2': 3}
    np.save(path_save + 'results/pretrained_model_' + name_pre + '_Npar.npy', Npar_pre)
else:
    Npar_pre = dict(enumerate(
                    np.load(path_save + 'results/pretrained_model_' + name_pre + '_Npar.npy',
                            allow_pickle=True).flatten()))[0]
    Gpar_net = dict(enumerate(
                    np.load(path_save + 'results/' + name_pre + '_Gpar_net.npy', allow_pickle=True).flatten()))[0]
mask = create_mask(Npar_pre, Gpar_data, Gpar_net, dtype, device)

######################################
######################################
############# PRE-TRAINING ###########
######################################
print('pretraining: total dataset size ', Tpar_pre['Nsamples'], ' batch size ', Tpar_pre['Nbatch'], ' learning rate ',
      Tpar_pre['lr'])
model_pre = model(Gpar_data['I'], Npar_pre['N'], C=Gpar_data['Norien'], Gpar=Gpar_net, mask=mask,
                  Nhidden=Npar_pre['Nhidden'], mod_loc=Npar_pre['mod_loc'],
                  dtype=dtype, device=device).to(device)
model_pre.trainingtype()
if train_pre|(isinstance(Tpar['lr'], float)==False):
    loss_fn = torch.nn.NLLLoss()
    mod = Tpar_pre['mmean'] + torch.zeros(np.concatenate(data).shape[0]).unsqueeze(1)
    print('I am here, starting pretraining')
    model_pre, _, _, _, _, _, traj_pre = traintest(data, label, datatest, labeltest,
                                          model_pre, loss_fn, dtype, Tpar=Tpar_pre, mod=mod,
                                         old_datatasktest=[], old_labeltasktest=[], plottraining=False,
                                                   save_traj=save_traj, seed=seed_batch_pre)
else:
    model_pre.load_state_dict(torch.load(path_save+'pretrained_model_'+name_pre+'.pt', map_location=device), strict=False)

# saving
if save_pre:
    torch.save(model_pre.state_dict(), path_save+'pretrained_model_'+name_pre+'.pt')
    np.save(path_save + 'results/pretrained_model_' + name_pre + '_trajectories.npy', traj_pre)

####################################
########## learning rate ###########
####################################

print(Tpar['lr'])
if isinstance(Tpar['lr'], float)==False:
    Tpar_LR = {**Tpar, 'Nsets': 1, 'Nsamples': 100}
    crit = [[] for _ in Tpar['lr']]
    whichlr = []
    ll = 0
    for lr in Tpar['lr']:
        Tpar_LR['lr'] = lr
        print('learning rate ', lr)
        for rr in range(2):
            data, datatest, label, labeltest, _, _, _, _ = \
                produce_dataset(Gpar_data, pretraining=True, task=None, Tpar_pre=Tpar_LR, Tpar=None,
                                path_mnist=path_mnist,
                                seed0=rr, nsamplespre=Tpar['Nsamples']*Tpar['Nsets'], nsamplestun=None,
                                testtoo=True)
            table = np.array([np.arange(10), np.random.choice(10, 10, replace=False)])
            print(table)
            labelperm = table[1, label]
            labeltestperm = table[1, labeltest]

            model_lr = model(Gpar_data['I'], Npar_pre['N'], C=Gpar_data['Norien'], Gpar=Gpar_net, mask=mask,
                              Nhidden=Npar_pre['Nhidden'], mod_loc=Npar_pre['mod_loc'],
                              dtype=dtype, device=device).to(device)
            mod = Tpar_LR['mmean'] + torch.zeros(np.concatenate(data).shape[0], device=device).unsqueeze(1)

            # build up on pretrained model
            model_lr.load_state_dict(torch.load(path_save + 'pretrained_model_' + name_pre + '.pt'), strict=False)
            model_lr.trainingtype(targeting=False, encoding=True,
                                   hidden=True, output=True)
            loss_fn = torch.nn.NLLLoss()
            # training progress
            _, _, _, _, _, _, traj_lr = \
                traintest(data, labelperm, datatest, labeltestperm,
                          model_lr, loss_fn, dtype, Tpar=Tpar_LR,
                          mod=mod, old_datatasktest=[], old_labeltasktest=[],
                          plottraining=False, save_traj=save_traj)

            crit[ll].append(np.mean(traj_lr[0]['correct'][-10:]))
            np.save(path_save + 'results/'+ name_pre + '_LR' +np.str(ll) + '_sim' +np.str(rr)+ '_trajectories.npy',traj_lr)
        whichlr.append(np.mean(crit[ll]))
        ll +=1

    np.save(path_save + 'results/' + name_pre + '_lr_crit.npy', crit)
    cutoff = (np.mean(traj_pre[0]['correct'][-10:])*.7)
    Tpar['lr'] = Tpar['lr'][np.argwhere(np.array(whichlr)>=cutoff)[0][0]]
    print('Tpar_tun[lr]=', Tpar['lr'])
    if train_model['gain']: Tpar_gain['lr'] = Tpar['lr']
    if train_model['MG']: Tpar_stoch['lr'] = Tpar['lr']

####################################
########## RE-TRAINING #############
####################################

if (Gpar_data['data_type']!=Gpar_data_task['data_type']): #& train_pre:
    print('switching task and retraining')
    Tpar_sw = {**Tpar, 'Nsets': 1, 'Nsamples': 1000, 'Nbatch':2**5, 'Nvalperc':0.3, 'lr':1e-4} #
    Gpar_sw = {**Gpar_data_task, 'Ndistractors': 0, 'seed0': seed_retrain, \
               'pool_orien': None} #[Gpar_data_task['S1'], Gpar_data_task['S2']]}

    #     Tpar_pre = {**Tpar, 'lr':1e-5, 'Nsets':1, 'Nsamples':4000, 'Nbatch':2**5, 'Nvalperc':0.3}

    # get data of the new task type:
    data, datatest, label, labeltest, _, _, _, _ = \
        produce_dataset(Gpar_sw, pretraining=True, task=None, Tpar_pre=Tpar_sw, Tpar=None, path_mnist=path_mnist,
                        seed0=Gpar_sw['seed0'], nsamplespre=Tpar_sw['Nsamples'], nsamplestun=None, testtoo=True)
    '''
    _, _, _, _, data, datatest, label, labeltest = \
        produce_dataset(Gpar_sw, pretraining=False, task=0, Tpar_pre=Tpar_sw, Tpar=Tpar_sw, path_mnist=path_mnist,
                        seed0=Gpar_sw['seed0'], nsamplespre=Tpar_sw['Nsamples'], nsamplestun=Tpar_sw['Nsamples'], testtoo=True)
    '''
    mod = Tpar['mmean'] + torch.zeros(np.concatenate(data).shape[0], device=device).unsqueeze(1)

    # build up on pretrained model
    model_pre.trainingtype(targeting=False, encoding=False,
                               hidden=False, output=True)
    loss_fn = torch.nn.NLLLoss()
    # training progress
    model_pre, _, _, _, _, _, traj_sw = \
        traintest(data, label, datatest, labeltest,
                  model_pre, loss_fn, dtype, Tpar=Tpar_sw,
                  mod=mod, old_datatasktest=[], old_labeltasktest=[],
                  plottraining=False, save_traj=True, seed=seed_batch_re)

    torch.save(model_pre.state_dict(), path_save + 'pretrained_model_' + name_pre + name_tun+'_switched.pt')
    np.save(path_save + 'results/pretrained_model_' + name_pre +name_tun+ '_switching_trajectories.npy', traj_sw)


####################################
########## task-training ###########
####################################
if train_model['weights']|train_model['gain']|train_model['MG']|train_model['decoder']:
    for tt in range(Ttasks):
        ################ DATA ################
        s1, s2, loc = S1[tt], S2[tt], LOC[tt]
        _, _, _, _, datatask, datatasktest, labeltask, labeltasktest = \
            produce_dataset(Gpar_data_task, pretraining=False, task=tt, Tpar_pre=Tpar_pre, Tpar=Tpar,
                            path_mnist=path_mnist, seed0=Gpar_data_task['seed0'],
                            nsamplespre=Tpar['Nsamples']*Tpar['Nsets'],
                            nsamplestun=Tpar['Nsamples']*Tpar['Nsets'], testtoo=True)
        print('task ', tt,  ' with ', len(datatask), 'total data points available with s1=', s1, ' and s2=', s2, ' at loc=', loc)
        print('Gpar_data I dimensions', Gpar_data['I'])
        print('Gpar_data_task I dimensions', Gpar_data_task['I'])
        ######### INFORMATIVENESS #############
        act_s1 = (model_pre.relu(torch.mm(torch.tensor(datatask[labeltask == s1], dtype=dtype, device=device), model_pre.gab))).to('cpu')
        act_s2 = (model_pre.relu(torch.mm(torch.tensor(datatask[labeltask == s2], dtype=dtype, device=device), model_pre.gab))).to('cpu')
        var = np.sqrt(.5 * (np.var(act_s1.detach().numpy(), axis=(0)) + np.var(act_s2.detach().numpy(), axis=(0))))
        var[var == 0] = np.nan
        dinf = np.abs(np.mean(act_s1.detach().numpy(), axis=(0)) - np.mean(act_s2.detach().numpy(), axis=(0)))/var
        # task-specific
        np.save(path_save + 'results/' + name_pre + name_tun+ '_task' + np.str(tt) + '_dinf.npy', dinf)
        ############### MODEL WEIGHTS #########
        if train_model['weights']:
            print('1) training all weights')
            model_tun = model(Gpar_data['I'], Npar_pre['N'], C=Gpar_data['Norien'], Gpar=Gpar_net, mask=mask,
                              Nhidden=Npar_pre['Nhidden'], mod_loc=Npar_pre['mod_loc'],
                              dtype=dtype, device=device).to(device)
            mod = Tpar['mmean'] + torch.zeros(np.concatenate(datatask).shape[0], device=device).unsqueeze(1)

            # build up on pretrained model
            if (Gpar_data['data_type']!=Gpar_data_task['data_type']):
                model_tun.load_state_dict(torch.load(path_save+'pretrained_model_'+name_pre+ name_tun+'_switched.pt'), strict=False)
            else:
                model_tun.load_state_dict(
                    torch.load(path_save + 'pretrained_model_' + name_pre + '.pt'), strict=False)
            model_tun.trainingtype(targeting=False, encoding=True,
                                   hidden=True, output=True)
            loss_fn = torch.nn.NLLLoss()
            # training progress
            model_tun, loss_train, loss_test, _, correct, _, traj_tun = \
                traintest(datatask, labeltask, datatasktest, labeltasktest,
                            model_tun, loss_fn, dtype, Tpar=Tpar,
                            mod=mod, old_datatasktest=[], old_labeltasktest=[],
                            plottraining=False, save_traj=save_traj, seed=seed_batch_task)
            # save model weights
            save_state_dict(model_tun, path_save + 'tune_weights_model_' + name_pre+ name_tun+ '_task' + np.str(tt))
            np.save(path_save + 'results/' + 'tune_weights_model_' + name_pre + name_tun+ '_task' + np.str(tt) + '_loss_train.npy',
                    loss_train)
            np.save(path_save + 'results/' + 'tune_weights_model_' + name_pre + name_tun+ '_task' + np.str(tt) + '_loss_test.npy',
                    loss_test)
            np.save(path_save + 'results/' + 'tune_weights_model_' + name_pre + name_tun+ '_task' + np.str(tt) + '_correct_test.npy',
                    correct)
            if save_traj:
                # save the trajectories of each learning
                np.save(
                    path_save + 'results/' + 'tune_weights_model_' + name_pre + name_tun+ '_task' + np.str(tt) + '_trajectories.npy',
                    traj_tun)
        ############### MODEL GAIN ############
        if train_model['gain']:
            print('2) training only coupling')
            model_tun_gain = model(Gpar_data['I'], Npar_pre['N'], C=Gpar_data['Norien'], Gpar=Gpar_net,
                                   mask=mask, Nhidden=Npar_pre['Nhidden'], mod_loc=Npar_pre['mod_loc'],
                                   dtype=dtype, device=device).to(device)
            mod = Tpar_gain['mmean'] + torch.zeros(np.concatenate(datatask).shape[0], device=device).unsqueeze(1)

            # build up on pretrained model
            if (Gpar_data['data_type']!=Gpar_data_task['data_type']):
                model_tun_gain.load_state_dict(torch.load(path_save+'pretrained_model_'+name_pre+ name_tun+'_switched.pt'), strict=False)
            else:
                model_tun_gain.load_state_dict(
                    torch.load(path_save + 'pretrained_model_' + name_pre + '.pt'), strict=False)
            model_tun_gain.trainingtype(targeting=True, encoding=False,
                                        hidden=False, output=False)
            loss_fn = torch.nn.NLLLoss()

            # training progress
            model_tun_gain, loss_train_gain, loss_test_gain, _, correct_gain, _, traj_tun = \
                traintest(datatask, labeltask, datatasktest,
                            labeltasktest,
                            model_tun_gain, loss_fn, dtype, Tpar=Tpar_gain,
                            mod=mod, old_datatasktest=[],
                            old_labeltasktest=[],
                            plottraining=False, save_traj=save_traj, seed=seed_batch_task)
            # save model gain
            save_state_dict(model_tun_gain, path_save + 'tune_gain_model_' + name_pre+ name_tun+ '_task' + np.str(tt))
            np.save(path_save + 'results/' + 'tune_gain_model_' + name_pre + name_tun+ '_task' + np.str(tt) + '_loss_train.npy',
                    loss_train_gain)
            np.save(path_save + 'results/' + 'tune_gain_model_' + name_pre + name_tun+ '_task' + np.str(tt) + '_loss_test.npy',
                    loss_test_gain)
            np.save(path_save + 'results/' + 'tune_gain_model_' + name_pre + name_tun+ '_task' + np.str(tt) + '_correct_test.npy',
                    correct_gain)
            if save_traj:
                # save the trajectories of each learning
                np.save(
                    path_save + 'results/' + 'tune_gain_model_' + name_pre + name_tun+ '_task' + np.str(tt) + '_trajectories.npy',
                    traj_tun)

        ############### MODEL MG #############
        if train_model['MG']:
            print('3) training only coupling and gain')
            model_tun_stoch = model(Gpar_data['I'], Npar_pre['N'], C=Gpar_data['Norien'], Gpar=Gpar_net,
                                    mask=mask, Nhidden=Npar_pre['Nhidden'], mod_loc=Npar_pre['mod_loc'],
                                    dtype=dtype, device=device).to(device)
            mod = Tpar_stoch['mmean'] + torch.zeros(np.concatenate(datatask).shape[0], device=device).unsqueeze(1)

            # build up on pretrained model
            if (Gpar_data['data_type'] != Gpar_data_task['data_type']):
                model_tun_stoch.load_state_dict(
                    torch.load(path_save + 'pretrained_model_' + name_pre + name_tun + '_switched.pt'), strict=False)
            else:
                model_tun_stoch.load_state_dict(
                    torch.load(path_save + 'pretrained_model_' + name_pre + '.pt'), strict=False)
            model_tun_stoch.trainingtype(targeting=True, encoding=False,
                                        hidden=False, output=False)
            loss_fn = torch.nn.NLLLoss()
            print('tau=', model_tun_stoch.tau)
            # training progress
            model_tun_stoch, loss_train_stoch, loss_test_stoch, _, correct_stoch, _, traj_tun = \
                traintest(datatask, labeltask, datatasktest,
                        labeltasktest,
                        model_tun_stoch, loss_fn, dtype, Tpar=Tpar_stoch,
                        mod=mod, old_datatasktest=[],
                        old_labeltasktest=[],
                        plottraining=False, save_traj=save_traj, seed=seed_batch_task)
            # save model stochastic
            save_state_dict(model_tun_stoch, path_save + 'tune_stoch_model_' + name_pre + name_tun+ '_task' + np.str(tt))
            np.save(path_save + 'results/' + 'tune_stoch_model_' + name_pre + name_tun+ '_task' + np.str(tt) + '_loss_train.npy',
                    loss_train_stoch)
            np.save(path_save + 'results/' + 'tune_stoch_model_' + name_pre + name_tun+ '_task' + np.str(tt) + '_loss_test.npy',
                    loss_test_stoch)
            np.save(path_save + 'results/' + 'tune_stoch_model_' + name_pre + name_tun+ '_task' + np.str(tt) + '_correct_test.npy',
                    correct_stoch)
            if save_traj:
                # save the trajectories of each learning
                np.save(
                    path_save + 'results/' + 'tune_stoch_model_' + name_pre + name_tun+ '_task' + np.str(tt) + '_trajectories.npy',
                    traj_tun)


        ############### MODEL decoding ########
        if train_model['decoder']:
            print('4) training only decoding')
            model_tun_dec = model(Gpar_data['I'], Npar_pre['N'], C=Gpar_data['Norien'], Gpar=Gpar_net,
                                  mask=mask, Nhidden=Npar_pre['Nhidden'], mod_loc=Npar_pre['mod_loc'],
                                  dtype=dtype, device=device).to(device)
            mod = Tpar['mmean'] + torch.zeros(np.concatenate(datatask).shape[0], device=device).unsqueeze(1)

            # build up on pretrained model
            if (Gpar_data['data_type'] != Gpar_data_task['data_type']):
                model_tun_dec.load_state_dict(
                    torch.load(path_save + 'pretrained_model_' + name_pre + name_tun + '_switched.pt'), strict=False)
            else:
                model_tun_dec.load_state_dict(
                    torch.load(path_save + 'pretrained_model_' + name_pre + '.pt'), strict=False)
            model_tun_dec.trainingtype(targeting=False, encoding=False,
                                   hidden=False, output=True)
            loss_fn = torch.nn.NLLLoss()
            # training progress
            model_tun_dec, loss_train_dec, loss_test_dec, _, correct_dec, _ , traj_tun = \
                traintest(datatask, labeltask, datatasktest, labeltasktest,
                            model_tun_dec, loss_fn, dtype, Tpar=Tpar,
                            mod=mod, old_datatasktest=[], old_labeltasktest=[],
                            plottraining=False, save_traj=save_traj, seed=seed_batch_task)
            # save model decoding
            save_state_dict(model_tun_dec, path_save + 'tune_dec_model_' + name_pre + name_tun+ '_task' + np.str(tt))
            np.save(path_save + 'results/' + 'tune_dec_model_' + name_pre + name_tun+ '_task' + np.str(tt) + '_loss_train.npy',
                    loss_train_dec)
            np.save(path_save + 'results/' + 'tune_dec_model_' + name_pre + name_tun+ '_task' + np.str(tt) + '_loss_test.npy',
                    loss_test_dec)
            np.save(path_save + 'results/' + 'tune_dec_model_' + name_pre + name_tun+ '_task' + np.str(tt) + '_correct_test.npy',
                    correct_dec)
            if save_traj:
                # save the trajectories of each learning
                np.save(
                    path_save + 'results/' + 'tune_dec_model_' + name_pre + name_tun+ '_task' + np.str(tt) + '_trajectories.npy',
                    traj_tun)





