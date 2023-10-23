import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
import copy
from collections import OrderedDict
import os.path as ospath

dec_colors = {'stoch': 'green', 'weights': 'red', 'dec': 'orange', 'gain': 'blue'}

############### networks ################
def l1_penalty(params, l1_lambda=0.001):
    """Returns the L1 penalty of the params."""
    l1_norm = []
    for p in params:
        if p.requires_grad:
            l1_norm.append(p.abs().sum())
    return l1_lambda*sum(l1_norm)

def coupling_matrix(gab_prop, Gpar_net):
    pos = np.unique(np.stack(gab_prop)[:,0])
    k1 = np.linspace(pos[0] + np.diff(pos[:2]) * Gpar_net['coupl_offset'],
                     pos[-1] - np.diff(pos[:2]) * Gpar_net['coupl_offset'],
                     Gpar_net['Npos'] - (Gpar_net['coupl_redfact']))

    orien = np.unique(np.stack(gab_prop)[:,2])
    k2 = np.linspace(orien[0] + np.diff(orien[:2]) * Gpar_net['coupl_offset'],
                     orien[-1] - np.diff(orien[:2]) * Gpar_net['coupl_offset'],
                     Gpar_net['Norien']- (Gpar_net['coupl_redfact']))
    K = [k1, k1, k2]
    gab_prop = np.stack(gab_prop)

    # over latent dimension 1
    FILTER = []
    PROP = []
    for pp in K[0]:
        for qq in K[1]:
            for oo in K[2]:
                fil = np.exp(-(gab_prop[:,0]-pp)**2/(2*Gpar_net['coupl_sigma'][0]**2))*\
                        np.exp(-(gab_prop[:,1]-qq)**2/(2*Gpar_net['coupl_sigma'][1]**2))*\
                        np.exp(-(gab_prop[:,2]-oo)**2/(2*Gpar_net['coupl_sigma'][2]**2))
                FILTER.append(fil/np.sum(fil))#*len(fil))
                PROP.append([pp,qq,oo])
    FILTER = np.stack(FILTER)/np.sum(np.stack(FILTER),axis=0)
    return FILTER, PROP


class MG_target(torch.nn.Module):
    # the inputs are modulated by exp(modulator*coupling)
    # parameters: weight = the modulator coupling terms
    # inputs: inputs in matrix form #samples x #dimension
    # mod: vector with size #samples
    # nonlinearity: boolean, if true the inputs are passed through an exp-nonlinearity
    def __init__(self, shape, gab_prop, Gpar_net, device):
        super(MG_target, self).__init__()
        FILTER, _ = coupling_matrix(gab_prop, Gpar_net)
        self.FILTER = torch.tensor(FILTER, device=device, dtype=torch.float)
        self.weight = torch.nn.parameter.Parameter(torch.ones([1,FILTER.shape[0]], device=device))
        
    def forward(self, inputs, mod, nonlinearity, varmod = None):
        coupling = torch.mm(self.weight**2, self.FILTER)
        if varmod is None:
            varmod = torch.var(mod)
        if nonlinearity:
            return torch.exp(inputs + torch.mm(mod, coupling) - varmod * coupling**2/2)
        else:
            return inputs*torch.exp(torch.mm(mod, coupling) - varmod * coupling**2 / 2)

class MG_Model_multi(torch.nn.Module):
    # a feedforward network for classification arbitrary # of hidden layers and flexible placing of modulator
    # parameters: I = input dimension, N = number of neurons in first layer, Nh = number of neurons in hidden layers
    #             mask = defines local connections, Nhidden = number of hidden layers, mod_loc = modulator placement
    def __init__(self, I, N, C, Gpar=None, mask=None, Nhidden=1, mod_loc = 0, nonlinearity_mod=False,
                dtype=torch.float, device='cpu'):
        super(MG_Model_multi, self).__init__()
        # Number of input features is I, number of neurons is N
        self.Gpar = Gpar
        self.device=device
        if self.Gpar is None: self.layer_enc = torch.nn.Linear(I, N)
        else:
            # build a front-end Gabor filter bank:
            gabtmp, self.gab_prop = gabor(np.sqrt(I), Gpar)
            self.gab = torch.tensor(np.stack(gabtmp),dtype=dtype, device=self.device).T
            N = self.gab.shape[1]
        # where to apply the modulator
        self.nonlinearity_mod = nonlinearity_mod
        if mod_loc==0:
            self.modulate = MG_target(shape=[1,N], gab_prop=self.gab_prop, Gpar_net=Gpar, device=device)
        else:
            self.modulate = MG_target(shape=[1,N], gab_prop=self.gab_prop, Gpar_net=Gpar, device=device) #
        self.modulate.to(device)
        self.mod_loc = mod_loc
        # hidden layers
        self.Nhidden = Nhidden
        dic_hidden = OrderedDict()
        for nn in range(1, self.Nhidden+1):
            dic_hidden['hidden' + str(nn)] = torch.nn.Linear(mask[nn].shape[0], mask[nn].shape[1])
            dic_hidden['relu' + str(nn)] = torch.nn.ReLU()
        self.hidden = torch.nn.Sequential(dic_hidden)
        # modulator readout
        self.gain = torch.ones([1,mask[-1].shape[1]], device=self.device)
        # output layer
        self.layer_out = torch.nn.Linear(mask[-1].shape[1], C)
        self.softmax = torch.nn.LogSoftmax(dim=1)
        # define relu
        self.relu = torch.nn.ReLU()
        # time constant
        self.tau = 1 # 0
        # move itself to right device
        self.to(self.device)
        # apply masks
        self.mask = mask
        self.applymask()

    def applymask(self):
        # Zero out weights
        with torch.no_grad():

            for nn in range(0, self.Nhidden*2, 2):
                if self.mask[int((nn + 2) / 2)].is_sparse:
                    mask_nn = self.mask[int((nn + 2) / 2)].to_dense().T

                else:
                    mask_nn = self.mask[int((nn + 2) / 2)].T
                if self.hidden[nn].weight.grad is not None:
                    self.hidden[nn].weight.grad.mul_(mask_nn)
                self.hidden[nn].weight.mul_(mask_nn)



    def forward(self, inputs, mod):
        # encoding
        if self.Gpar is None: act = self.layer_enc(inputs)
        else: act = torch.mm(inputs, self.gab)
        if self.mod_loc==0:
            # apply modulator-label in input layer
            act = self.modulate.forward(act, mod, nonlinearity=self.nonlinearity_mod)
        act = self.relu(act)
        ##### hidden layers ####
        for nn in range(0, self.Nhidden * 2, 2):
            # apply weights
            act = self.hidden[nn](act)
            # is this layer being modulated?
            if self.mod_loc == ((nn-1)*2):
                act = self.modulate.forward(act, mod, nonlinearity=self.nonlinearity_mod)
            # apply relu
            act = self.hidden[nn+1](act)
        ##### apply gain ####
        act = act*self.gain
        # output layer
        act = self.layer_out(act)
        act = self.softmax(act)
        return act

    def learn_routing(self,inputs, mod):
        # learn the gain weights for the last hidden layer
        with torch.no_grad():
            ###### first layer ########
            if self.Gpar is None:
                act = self.layer_enc(inputs)
            else:
                act = torch.mm(inputs, self.gab)
            if self.mod_loc == 0:
                act = self.modulate.forward(act, mod, nonlinearity=self.nonlinearity_mod)
            act = self.relu(act)
            ##### hidden layers ####
            for nn in range(0, self.Nhidden * 2, 2):
                # apply weights
                act = self.hidden[nn](act)
                # is this layer being modulated?
                if self.mod_loc == ((nn - 1) * 2):
                    act = self.modulate.forward(act, mod, nonlinearity=self.nonlinearity_mod)
                # apply relu
                act = self.hidden[nn + 1](act)
            ##### compute gain #####
            self.gain, self.tau  = MG_readout(gain=self.gain, act=act, mod=mod, tau=self.tau, device=self.device)

    def backward(self, x, mod, y, loss_fn, optimizer, l1_lambda=0.001):
        # Zero the gradients before running the backward pass.
        optimizer.zero_grad()
        # run forward
        y_pred = self.forward(x, mod)
        # Compute loss.
        loss = loss_fn(y_pred, y) + l1_penalty(self.parameters(), l1_lambda)
        # Backward pass:
        loss.backward()
        # apply mask
        self.applymask()
        # Update the weights using gradient descent.
        optimizer.step()

    def trainingtype(self, targeting=False, encoding=True, hidden=True, output=True):
        # set what is being trained and what is kept fixed
        if encoding&(self.Gpar is None):
            self.layer_enc.weight.requires_grad = True
            self.layer_enc.bias.requires_grad = True
        elif self.Gpar is None:
            self.layer_enc.weight.requires_grad = False
            self.layer_enc.bias.requires_grad = False
        if hidden:
            for nn in range(0, self.Nhidden * 2, 2):
                self.hidden[nn].weight.requires_grad = True
                self.hidden[nn].bias.requires_grad = True
        else:
            for nn in range(0, self.Nhidden * 2, 2):
                self.hidden[nn].weight.requires_grad = False
                self.hidden[nn].bias.requires_grad = False
        if output:
            self.layer_out.weight.requires_grad = True
            self.layer_out.bias.requires_grad = True
        else:
            self.layer_out.weight.requires_grad = False
            self.layer_out.bias.requires_grad = False
        if targeting:
            self.modulate.weight.requires_grad = True
        else:
            self.modulate.weight.requires_grad = False

############### MG readout ###############

# MG readout based on covariance or correlation with m
def MG_readout(gain, act, mod, tau=1, norm=True, device='cpu'):
    with torch.no_grad():
        #### correlation(k,m)
        actstand = act-torch.mean(act,axis=0)
        a = (torch.mm((actstand).T, (mod-torch.mean(mod))) / mod.shape[0]).T
        # cut off any negative correlations at 0
        a[a < 0] = 0
        # normalize to mean=1
        if torch.tensor(norm, device=device)&(torch.sum(a>0)>0): a = a/torch.sum(a)*act.shape[1]
        elif (torch.sum(a>0)==0): a = torch.ones([1,act.shape[1]], device=device)
        # combine with old estimate
        a = (gain*tau+a)/(tau+1)
        # normalize to mean=1
        if norm: a = a/torch.sum(a)*act.shape[1]
        return a, torch.min(torch.tensor([100,tau+1], device=device))

##########################################
############## data loaders ##############
##########################################

def addnois(n, nois):
    return np.random.randn(n)*nois

def get_data2(POS, images, labels, s1, s2, num, F, scal, seed,
              distract=False, nois=0, task = None, distract_diffdig=False):
    if s2 is not None:
        taskimages = images[(labels==s1)|(labels==s2), :]
        tasklabels = labels[(labels == s1) | (labels == s2)]
        if distract_diffdig:
            distractorimages = images[(labels!=s1)&(labels!=s2), :]
        else:
            distractorimages = images.copy()
    else:
        taskimages = images
        tasklabels = labels

    data = []
    datalabel = []
    location = []
    if task is not None:
        pos = POS[task].reshape(1,2)
        posdist = POS[np.arange(POS.shape[0]) != task, :]
    else:
        pos = POS

    # create random seeds
    np.random.seed(seed)
    seeds = np.random.choice(10000, num)

    for ii in range(num):
        np.random.seed(seeds[ii])
        ind = np.random.choice(taskimages.shape[0],1)
        pind = np.random.choice(pos.shape[0], 1)[0]
        images_ii = taskimages[ind]

        data_ii = np.zeros((F*scal)**2)-99
        ###### add digit ######
        mask = np.zeros([F*scal, F*scal], dtype='bool')
        for ff in range(F):
            mask[pos[pind,0]:(pos[pind,0]+F), pos[pind,1]+ff] = True
        keepzeros = (images_ii==0).reshape(images_ii.shape)
        images_ii[keepzeros] = -99
        data_ii[mask.ravel()] = images_ii.ravel()

        ###### add distractors ######
        if distract:
            inddist = np.random.choice(distractorimages.shape[0],posdist.shape[0])
            for dd in range(posdist.shape[0]):
                images_ii = distractorimages[inddist[dd]]
                keepzeros = (images_ii==0).reshape(images_ii.shape)
                images_ii[keepzeros] = -99
                mask = np.zeros([F*scal, F*scal], dtype='bool')
                for ff in range(F):
                    mask[posdist[dd,0]:(posdist[dd,0]+F), posdist[dd][1]+ff] = True
                data_ii[mask.ravel()] = images_ii.ravel()
        # add noise
        data_ii[data_ii==-99] = addnois(np.sum(data_ii==-99), nois)
        data.append(data_ii)
        datalabel.append(tasklabels[ind])
        location.append(pind)
    return np.stack(data), np.concatenate(datalabel), location

##########################################
############## training ##################
##########################################

def getbatches(data, Nbatch, Nsamples, seed):
    # Nsamples is the total size of this sub-dataset
    # Nbatch is the size of one mini-batch
    # set the seed
    np.random.seed(seed)
    # fix a sub-dataset
    samp = np.random.choice(data.shape[0], Nsamples, replace=False)
    # fix the batches
    fact = int(np.floor(Nsamples/Nbatch))
    batches = []
    for bb in range(fact):
        batches.append(np.random.choice(samp, Nbatch, replace=False))
        samp = np.setdiff1d(samp, batches[bb])
    return batches

def traintest(data, label, datatest, labeltest, net0, loss_fn, dtype, Tpar,
              mod=None, old_datatasktest=None, old_labeltasktest=None,
              plottraining=False, cumsamples = False, save_traj=False, seed=None):
    # trains a model *net* and tracks the training progress on the testing data too
    # *data* is used for training, *datatest* is used for testing
    # B... size of one dataset
    # Nbatches... number of datasets
    loss_train, loss_test, correct, trajectories = [], [], [], []
    # testing loss before training
    cqq, lqq = test(datatest, labeltest, net0, loss_fn, dtype, mumean=Tpar['mmean'])
    loss_test.append(lqq)
    correct.append(cqq)
    # test on previous tasks
    if old_datatasktest is not None:
        correctold = [[] for _ in range(len(old_datatasktest))]
        for tt in range(len(correctold)):
            cqq, _ = test(old_datatasktest[tt], old_labeltasktest[tt], net0, loss_fn, dtype, mumean=Tpar['mmean'])
            correctold[tt].append(cqq)
    else: correctold = None
    wnorm = []
    # create a random seed series
    if seed is not None:
        np.random.seed(seed)
        seeds = np.random.choice(Tpar['Nsets']*100, Tpar['Nsets'], replace=False)
    else:
        print('WARNING: seeds are not set!')
        seeds = np.arange(Tpar['Nsets'])
    # take increasing amounts of data
    for qq in range(Tpar['Nsets']):

        if cumsamples==False:
            net = copy.deepcopy(net0)
        # reinitalize optimizer
        if Tpar['optimizer_type']=='adam':
            optimizer = torch.optim.Adam(net.parameters(),
                                     lr=Tpar['lr'], weight_decay=Tpar['weight_decay'])
        elif Tpar['optimizer_type']=='sgd':
            optimizer = torch.optim.SGD(net.parameters(),
                                     lr=Tpar['lr'], momentum=Tpar['momentum'])
        # divide the first set of data (of size Nsamples) into batches of Nbatch for SGD
        if cumsamples:
            batches = getbatches(data, Tpar['Nbatch'], Tpar['Nsamples'], seed=seeds[qq])
        else:
            batches = getbatches(data, Tpar['Nbatch'], Tpar['Nsamples']*(qq+1), seed=seeds[qq])
        # separate into training and evaluation batches
        split = int(np.ceil(Tpar['Nvalperc'] * len(batches)))
        datatrain = [data[bb] for bb in batches[:split]]
        labeltrain = [label[bb] for bb in batches[:split]]
        modtrain = [mod[bb] for bb in batches[:split]]
        dataval = data[np.concatenate(batches[split:])]
        labelval = label[np.concatenate(batches[split:])]
        print(len(datatrain), ' batches for training, ', len(batches[split:]), ' batches for validation')
        # train with this set of data
        net, losstrain, losstest, cqq, traj = train(datatrain, labeltrain, dataval, labelval, datatest, labeltest,
                                                      modtrain, Tpar,
                                                      net, loss_fn, optimizer, dtype,
                                                      plotit=plottraining, save_traj=save_traj)
        # save the norm
        wnorm.append(torch.mean((net.modulate.weight) ** 2))
        # save intermediate loss
        loss_train.append(losstrain)
        loss_test.append(losstest)
        correct.append(cqq)
        trajectories.append(traj)
        # test on previous tasks
        if old_datatasktest is not None:
            for tt in range(len(correctold)):
                cqq, _ = test(old_datatasktest[tt], old_labeltasktest[tt], net, loss_fn, dtype, mumean=Tpar['mmean'])
                correctold[tt].append(cqq)
    return net, loss_train, loss_test, wnorm, correct, correctold, trajectories


def train(datatrain, labeltrain, dataval, labelval, datatest, labeltest, modtrain,
          Tpar, net, loss_fn, optimizer, dtype,
          plotit=False, save_traj=False):
    # train with backprop the weights and/or coupling terms with the data in *batches*
    # optionally use MG-routing
    # stop when the testing loss goes up (indicating overfitting)
    losstrain, lossval, losstest, correct = [], [], [], []
    # evaluate
    if len(datatrain)>1:
        con_datatrain = np.concatenate(datatrain,axis=0)
        con_labeltrain = np.concatenate(labeltrain)
    else:
        con_datatrain = datatrain[0]
        con_labeltrain = labeltrain[0]
    _, lqq = test(con_datatrain, con_labeltrain, net, loss_fn, dtype=dtype,
                  mumean=Tpar['mmean'])
    losstrain.append(lqq)
    _, lqq = test(dataval, labelval, net, loss_fn, dtype=dtype, mumean=Tpar['mmean'])
    lossval.append(lqq)
    cqq, lqq = test(datatest, labeltest, net, loss_fn, dtype=dtype, mumean=Tpar['mmean'],)
    losstest.append(lqq)
    correct.append(cqq)
    if plotit: fig, ax = plt.subplots(1,4,figsize=(11,2))
    if Tpar['flex_epoch']:
        netbest = copy.deepcopy(net).to('cpu')
        Nupdates = 0
        bestind = 0
    else:
        bestind = Nupdates = Tpar['Emax']-1
    # current dataset
    for ee in range(Tpar['Emax']):
        for bb in range(len(datatrain)):
            x = torch.tensor(datatrain[bb], dtype=dtype, device=net.device)
            mod_bb = modtrain[bb].to(net.device)
            if Tpar['backprop']:
                y = torch.tensor(labeltrain[bb], dtype=torch.long, device=net.device)
                net.backward(x, mod_bb, y, loss_fn, optimizer, l1_lambda=Tpar['l1_lambda'])
            if Tpar['route']:
                net = trainroute(x, Tpar['samp_rate'], Tpar['mmsigma'], net)
            del x
        # evaluate
        _, lqq = test(con_datatrain, con_labeltrain, net, loss_fn, dtype=dtype,
                      mumean=Tpar['mmean'])
        losstrain.append(lqq)
        _, lqq = test(dataval, labelval, net, loss_fn, dtype=dtype, mumean=Tpar['mmean'])
        lossval.append(lqq)
        cqq, lqq = test(datatest, labeltest, net, loss_fn, dtype=dtype, mumean=Tpar['mmean'])
        losstest.append(lqq)
        correct.append(cqq)
        if Tpar['flex_epoch']:
            # is this model better than the best one so far?
            if lossval[-1] <= np.min(lossval):
                Nupdates +=1
                netbest = copy.deepcopy(net)
                bestind = ee+1
    if Tpar['flex_epoch']==False: netbest = copy.deepcopy(net)
    if plotit:
        ax[0].plot(losstrain, '-')
        ax[1].plot(lossval, '-')
        ax[2].plot(losstest, '-')
        ax[3].plot(correct, '-')
        ax[0].set_title('training loss (N='+np.str(len(datatrain)))
        ax[1].set_title('validation loss (N='+np.str(dataval.shape[0]))
        ax[2].set_title('testing loss (N='+np.str(datatest.shape[0]))
        ax[3].set_title('testing % correct')
        for aa in range(3):
            ax[aa].set_xlabel('# epochs')
            ax[aa].set_yscale('log')
        ax[3].set_xlabel('# epochs')
        ax[3].set_ylim(0,1)
        fig.tight_layout()
        print('# of model updates: ', Nupdates)
    if save_traj:
        traj = {'training': losstrain, 'testing': losstest, 'validation': lossval, 'correct':correct}
    else:
        traj = None
    return netbest, losstrain[bestind], losstest[bestind], correct[bestind], traj


def trainroute(x, samp_rate, musigma, net):
    xrout = torch.zeros([samp_rate,x.shape[1]], device=net.device)
    modvar = torch.randn(samp_rate).unsqueeze(1).to(net.device) * musigma
    net.learn_routing(xrout, modvar)
    return net


def test(data, label, net, loss_fn, dtype, mumean=0):
    if (net.device=='cpu')|(len(label)<=100):
        # testing
        x = torch.tensor(data, dtype=dtype, device=net.device)
        y = torch.tensor(label, dtype=torch.long, device=net.device)
        # forward
        y_pred = net.forward(x, mumean+torch.zeros(x.shape[0], device=net.device).unsqueeze(1))
        # evaluate
        loss_test = float(loss_fn(y_pred, y).detach())
        correct = float(torch.mean((y == torch.argmax(y_pred.detach(), axis=1)).type(torch.float)))
    else:
        # take batches of data
        loss_test, correct = [], []
        for bb in range(0, data.shape[0], 100):
            bound = np.min([bb+100, data.shape[0]])
            # testing
            x = torch.tensor(data[bb:bound], dtype=dtype, device=net.device)
            y = torch.tensor(label[bb:bound], dtype=torch.long, device=net.device)
            # forward
            y_pred = net.forward(x, mumean + torch.zeros(x.shape[0], device=net.device).unsqueeze(1))
            # evaluate
            loss_test.append(float(loss_fn(y_pred, y).detach()))
            correct.append(float(torch.mean((y == torch.argmax(y_pred.detach(), axis=1)).type(torch.float))))
        correct, loss_test = sum(correct) / len(correct), sum(loss_test) / len(loss_test)
    return correct, loss_test

####################### helper functions ############################

def vis_weights(datatask, F, scal, nettask, wold, a1old, a2old, net, mask, N):
    ### visualize network learning in weights
    fig2, ax2 = plt.subplots(1, 5, figsize=(20, 3))
    dtmp = np.mean(datatask, axis=0)
    im = ax2[0].imshow(dtmp.reshape(F * scal, F * scal), cmap='gray');
    fig2.colorbar(im, ax=ax2[0])

    wdiff = (nettask.modulate.weight.detach().numpy().ravel()) - wold
    enc = mask.numpy()  
    im = ax2[1].imshow((enc.dot(wdiff)).reshape(F * scal, F * scal), cmap='gray');
    fig2.colorbar(im, ax=ax2[1])

    a1diff = nettask.gain1.detach().numpy().ravel() - a1old
    im = ax2[2].imshow((enc.dot(a1diff)).reshape(F * scal, F * scal), cmap='gray');
    fig2.colorbar(im, ax=ax2[2])

    weightdiff = nettask.layer_enc.weight.data.detach().numpy().T - net.layer_enc.weight.data.detach().numpy().T
    im = ax2[4].imshow(np.mean((mask.numpy() * (weightdiff)), axis=1).reshape(F * scal, F * scal),
                       cmap='gray');
    fig2.colorbar(im, ax=ax2[4])

    map2 = (nettask.hidden1.weight.data.detach().numpy().dot(enc.T / N))
    a2diff = nettask.gain2.data.detach().numpy() - a2old

    im = ax2[3].imshow((a2diff.dot(map2)).reshape(F * scal, F * scal),
                       cmap='gray');
    fig2.colorbar(im, ax=ax2[3])
    ax2[0].set_title('task')
    ax2[1].set_title('w $\Delta$')
    ax2[2].set_title('gain $\Delta$ a1')
    ax2[4].set_title('ave $\Delta$')
    ax2[3].set_title('gain $\Delta$ a2')
    fig2.tight_layout()
    return fig2, ax2

def gabor(nsamp, Gpar):
    # along each spatial dimension there are Gpar['Npos'] positions
    t = np.linspace(-2*np.pi, 2*np.pi, nsamp)
    x, y = np.meshgrid(t, t)
    # define position grid
    pos = np.linspace(-2*np.pi, 2*np.pi, Gpar['Npos'])[Gpar['bounds']:(Gpar['Npos']-Gpar['bounds'])]
    posx, posy = np.meshgrid(pos,pos)
    posx, posy = posx.ravel(), posy.ravel()
    orien = np.arange(0, np.pi,np.pi/Gpar['Norien'])
    gab, prop = [], []
    for pp in range(len(posx)):
        mu = [posx[pp], posy[pp]]
        for oo in range(Gpar['Norien']):
            theta = orien[oo]
            # orientation
            xp = (x.ravel()-mu[0])*np.cos(theta)+(y.ravel()-mu[1])*np.sin(theta)
            yp = -(x.ravel()-mu[0])*np.sin(theta)+(y.ravel()-mu[1])*np.cos(theta)
            # gaussian times cosine
            gab.append(np.exp(-((xp)**2+(yp)**2)/(2*Gpar['sig']**2))*np.cos(2*np.pi*xp/Gpar['lam']))
            gab[-1][np.abs(gab[-1]) < .1] = 0
            prop.append([mu[0], mu[1], theta])
    return gab, prop


def vis_res(Npar, data, label, dtype,
            names, loss_train, loss_test, correct, mmean,Gpar_data_task,
            model1=None, model2=None, model3=None, model4=None, fig=None, ax=None, fig_label=None):
    if loss_train is not None:
        if fig is None: fig, ax = plt.subplots(1, 3, figsize=(9, 3))
        for ii in range(len(loss_train)):
            if len(loss_train[ii])>1:
                ax[0].plot(np.arange(Npar['Nsamples'], Npar['Nsamples'] * (len(loss_train[ii]) + 1), Npar['Nsamples']), loss_train[ii],
                           label=names[ii], color=dec_colors[names[ii]])
                ax[1].plot(np.arange(0, Npar['Nsamples'] * (len(loss_train[ii]) + 1), Npar['Nsamples']), loss_test[ii],
                           color=dec_colors[names[ii]])
                ax[2].plot(np.arange(0, Npar['Nsamples'] * (len(loss_train[ii]) + 1), Npar['Nsamples']), correct[ii]*100,
                           color=dec_colors[names[ii]], label=fig_label)
                fig_label = None
        if 'LOC' in Gpar_data_task:
            ax[0].set_title('LOC='+np.str(Gpar_data_task['LOC'][0])+'\n training loss')
        ax[1].set_title('testing loss')
        ax[2].set_title('testing % correct')
        ax[2].set_ylim(-1, 101)
        for aa in range(2):
            ax[aa].set_xlabel('# data points')
            ax[aa].set_yscale('log')
        ax[2].set_xlabel('# data points')
        fig.tight_layout()

    ################################################
    if data is not None:
        x = torch.tensor(data, dtype=dtype)
        y = torch.tensor(label, dtype=torch.long)
        if model1 is not None:
            mod = torch.zeros(x.shape[0]).unsqueeze(1) + mmean[0]
            y_pred = model1.forward(x, mod)
            y_pred = np.argmax(y_pred.detach().numpy(), axis=1)
            print(names[0]+': new data perf: ', np.round(100*np.mean(y.numpy() == y_pred)))
        if model2 is not None:
            mod = torch.zeros(x.shape[0]).unsqueeze(1) + mmean[1]
            y_pred = model2.forward(x, mod)
            y_pred = np.argmax(y_pred.detach().numpy(), axis=1)
            print(names[1]+': new data perf: ', np.round(100*np.mean(y.numpy() == y_pred)))
        if model3 is not None:
            mod = torch.zeros(x.shape[0]).unsqueeze(1) + mmean[2]
            y_pred = model3.forward(x, mod)
            y_pred = np.argmax(y_pred.detach().numpy(), axis=1)
            print(names[2]+': new data perf: ', np.round(100*np.mean(y.numpy() == y_pred)))
        if model4 is not None:
            mod = torch.zeros(x.shape[0]).unsqueeze(1) + mmean[3]
            y_pred = model4.forward(x, mod)
            y_pred = np.argmax(y_pred.detach().numpy(), axis=1)
            print(names[3]+': new data perf: ', np.round(100*np.mean(y.numpy() == y_pred)))
    return fig, ax

def save_state_dict(model, path):
    out = model.state_dict()
    out['gain'] = model.gain
    torch.save(out, path+'.pt')

def load_model(model, Gpar_data, Npar_pre, Gpar_net, path, mask):
    if mask is None:
        if ("masktype" in Npar_pre) == False:
            Npar_pre['masktype'] = 'full'
        mask = create_mask(Npar_pre, Gpar_data, Gpar_net, dtype=torch.float, device='cpu')
    model_out = model(Gpar_data['I'], Npar_pre['N'], C=Gpar_data['Norien'], Gpar=Gpar_net, mask=mask,
                            Nhidden=Npar_pre['Nhidden'], mod_loc=Npar_pre['mod_loc'])
    model_out.load_state_dict(torch.load(path +'.pt', map_location=torch.device('cpu')), strict=False)
    inp = torch.load(path +'.pt', map_location=torch.device('cpu'))
    if 'gain' in inp:
        print('setting gain')
        model_out.gain = inp['gain']
    return model_out


def unit_tests(model_pre, model_tun, model_tun_dec, model_tun_gain, model_tun_stoch, Npar_pre, eps=1e-5):
    test = []
    if model_pre is not None:
        print('pretrained model should have all gain terms set to 1: ')
        test.append(torch.mean(model_pre.gain)==1)
        print('\t', test[-1])
        print('pretrained model should have all coupling terms set to 1')
        test.append(torch.mean(model_pre.modulate.weight)==1)
        print('\t', test[-1])
    if model_tun is not None:
        print('tuned weight model should have all gain terms set to 1')
        test.append(torch.mean(model_tun.gain)==1)
        print('\t', test[-1])

        print('tuned weight model should have all coupling terms set to 1')
        test.append(torch.mean(model_tun.modulate.weight)==1)
        print('\t', test[-1])
    if model_tun_gain is not None:
        print('tuned gain model should have all gain terms set to 1')
        test.append(torch.mean(model_tun_gain.gain)==1)
        print('\t', test[-1])

        print('tuned gain model should no longer have all coupling terms set to 1')
        test.append(torch.mean((1*(model_tun_gain.modulate.weight==1)).type(torch.float))!=1)
        print('\t', test[-1])
    if model_tun_stoch is not None:
        print('tuned stochastic model should no longer have all gain terms set to 1')
        test.append(torch.mean((1*(model_tun_stoch.gain==1)).type(torch.float))!=1)
        print('\t', test[-1])

        print('tuned stochastic model should no longer have all coupling terms set to 1')
        test.append(torch.mean((1*(model_tun_stoch.modulate.weight==1)).type(torch.float))!=1)
        print('\t', test[-1])
    if (model_pre is not None)&(model_tun_gain is not None)&(model_tun_stoch is not None):
        print('pretrained model and gain model and stochastic gain model should all have the exact same weights')
        for nn in range(0,Npar_pre['Nhidden']*2,2):
            test.append((np.abs(torch.mean((model_pre.hidden[nn].weight==model_tun_gain.hidden[nn].weight).type(torch.float))-1)<eps)&
                  (np.abs(torch.mean((model_pre.hidden[nn].weight==model_tun_stoch.hidden[nn].weight).type(torch.float))-1)<eps))
            print('\t', test[-1])
        test.append((torch.mean((model_pre.layer_out.weight==model_tun_gain.layer_out.weight).type(torch.float))==1)&
             (torch.mean((model_pre.layer_out.weight==model_tun_stoch.layer_out.weight).type(torch.float))==1))
        print('\t', test[-1])

    if (model_pre is not None)&(model_tun is not None):
        print('tuned weight model and pretrained model should not have the same weights')
        for nn in range(0,Npar_pre['Nhidden']*2,2):
            test.append((torch.mean((model_pre.hidden[nn].weight==model_tun.hidden[nn].weight).type(torch.float))<1))
            print('\t', test[-1])
        test.append((torch.mean((model_pre.layer_out.weight==model_tun.layer_out.weight).type(torch.float))<1))
        print('\t', test[-1])
    if (model_pre is not None)&(model_tun_dec is not None):
        print('tuned decoder model and pretrained model should have the same hidden weights but not have the same weights in the last layer')
        for nn in range(0,Npar_pre['Nhidden']*2,2):
            test.append(np.abs(torch.mean((model_pre.hidden[nn].weight==model_tun_dec.hidden[nn].weight).type(torch.float))-1)<eps)
            print('\t', test[-1])
        test.append((torch.mean((model_pre.layer_out.weight==model_tun_dec.layer_out.weight).type(torch.float))<1))
        print('\t', test[-1])

    ### test for the mask
    print('are mask entries binary')
    for ll in range(1, Npar_pre['Nhidden'] + 1):
        test.append((torch.sum(model_pre.mask[ll]==1)+torch.sum(model_pre.mask[ll]==0))==\
                    (model_pre.mask[ll].shape[0]*model_pre.mask[ll].shape[1]))
        print('\t', test[-1])
    if Npar_pre['masktype']=='sparse':
        print('do all mask entries have the same number of non-zero elements')
        for ll in range(1, Npar_pre['Nhidden']+1):
            conn = np.sum(model_pre.mask[ll].numpy(),axis=0)[0]
            test.append(np.mean(np.sum(model_pre.mask[ll].numpy(),axis=0)==conn)==1)
            print('\t', test[-1])
        print('is orientation preserved in mask pooling (testing only a few random samples')
        test.append(np.sum([np.sum(np.diff(np.stack(model_pre.gab_prop)[model_pre.mask[1][:,tt].numpy() > 0, 2]) != 0)
                            for tt in np.random.choice(model_pre.mask[1].shape[1],10, replace=False)]) == 0)
        print('\t', test[-1])
        if Npar_pre['Nhidden']>1:
            test.append(np.sum([np.sum(np.diff(np.stack(model_pre.gab_prop)[model_pre.mask[1].numpy().dot(\
                                                                    model_pre.mask[2][:,tt].numpy()) > 0, 2]) != 0)
                                for tt in np.random.choice(model_pre.mask[2].shape[1],10, replace=False)]) == 0)
            print('\t', test[-1])
            if Npar_pre['Nhidden'] > 2:
                test.append(np.sum([np.sum(np.diff(np.stack(model_pre.gab_prop)[model_pre.mask[1].numpy().dot( \
                    model_pre.mask[2].numpy()).dot(model_pre.mask[3][:, tt]) > 0, 2]) != 0)
                                    for tt in np.random.choice(model_pre.mask[3].shape[1], 10, replace=False)]) == 0)
                print('\t', test[-1])

    if np.mean(test)==1:
        print('\u001b[32m'+'ALL TESTS PASSED'+'\033[0m')
    else:
        print('\u001b[31m'+'TEST DID NOT PASS'+'\033[0m')

def produce_dataset(Gpar_data, nsamplespre, nsamplestun, testtoo,
                    pretraining = True, task=None, Tpar_pre=None, Tpar=None,
                    path_mnist='', seed0=None):

    if task is not None: s1, s2, loc = Gpar_data['S1'][task], Gpar_data['S2'][task], Gpar_data['LOC'][task]
    else: datatask, datatasktest, labeltask, labeltasktest = None, None, None, None
    if seed0 is not None:
        np.random.seed((seed0))
    if Gpar_data['data_type']=='gabor':
        # example dataset
        images, labelall = gabor(np.sqrt(Gpar_data['I']), Gpar_data)
        images, labelall = np.array(images), np.array(labelall)
        position = labelall[:, :2].dot(np.array([1, 100]))
        position = np.digitize(position, np.unique(position)) - 1
        label0 = np.digitize(labelall[:, 2], np.unique(labelall[:, 2])) - 1
        label = label0.copy()
        if pretraining:
            # only use certain orientations
            if Gpar_data['pool_orien'] is not None:
                ind_orien = np.sum(np.array([label == dd for dd in Gpar_data['pool_orien']]), axis=0) > 0
                images, label = images[ind_orien], label[ind_orien]
            while images.shape[0]<nsamplespre:
                images, label = np.repeat(images, 2, axis=0), np.repeat(label,2)
            # add nois
            data = images+np.random.randn(len(label),Gpar_data['I'])*Gpar_data['nois']
            if testtoo:
                datatest, labeltest = images+np.random.randn(len(label), Gpar_data['I'])*Gpar_data['nois'], label.copy()
            print(len(data), ' data samples for pretraining')
            # standardize
            data = ((data.T-np.mean(data,axis=1))/np.sqrt(np.var(data,axis=1))).T
            if testtoo: datatest = ((datatest.T-np.mean(datatest,axis=1))/np.sqrt(np.var(datatest,axis=1))).T
            else: datatest, labeltest=None, None
        else:
            data, label, datatest, labeltest= None, None, None, None
        ### task ###
        if task is not None:
            # specify which position and which orientations involve the task
            ind = (position==loc)&((label0==s1)|(label0==s2))
            taskind, nottaskind = np.where(ind)[0], np.where(ind==False)[0]
            datatask, labeltask = images[taskind], np.digitize(labelall[taskind, 2], np.unique(labelall[:, 2])) - 1
            if testtoo:
                nn = 1
                datatasktest, labeltasktest = datatask.copy(), labeltask.copy()
                while datatasktest.shape[0] < nsamplespre:
                    datatasktest, labeltasktest = np.repeat(datatasktest, nn, axis=0), np.repeat(labeltasktest, nn, axis=0)
                    nn += 1
            print(len(datatask), ' data samples for task ', task)
            ### distractor ###
            pospos = np.sort(np.unique(position))
            # exclude the task position
            pospos = pospos[loc!=pospos]
            # sample distractor position
            #np.random.seed(seeds[2])
            distractors = np.random.choice(pospos, Gpar_data['Ndistractors'], replace=False)
            # sample distractor orientation
            distr_orien = np.array([s1, s2])
            if Gpar_data['Ndistr_cat']>0:
                orien = np.unique(label0[(label0!=s1)&(label0!=s2)])
                distr_orien = np.concatenate((distr_orien, np.random.choice(orien, Gpar_data['Ndistr_cat'], replace=False)))
            for ii in distractors:
                # sample random among orientations for distractors
                ind_ii = np.sum(np.array([label0 == dd for dd in distr_orien]), axis=0) > 0
                samp = np.random.choice(np.where((ii==position)&ind_ii)[0], len(datatask), replace=True)
                datatask += images[samp]
                if testtoo:
                    ind_ii = np.sum(np.array([label0==dd for dd in distr_orien]),axis=0)>0
                    samp = np.random.choice(np.where((ii == position)&ind_ii)[0],
                                            len(datatasktest), replace=True)
                    datatasktest += images[samp]
            # add noise
            datatask = datatask + np.random.randn(len(labeltask), Gpar_data['I']) * Gpar_data['nois']
            if testtoo:
                datatasktest = datatasktest + np.random.randn(len(labeltasktest), Gpar_data['I']) * Gpar_data['nois']
            # standardize
            datatask = ((datatask.T-np.mean(datatask,axis=1))/np.sqrt(np.var(datatask,axis=1))).T
            if testtoo: datatasktest = ((datatasktest.T-np.mean(datatasktest,axis=1))/np.sqrt(np.var(datatasktest,axis=1))).T
            else: datatasktest, labeltasktest = None, None
    else:
        seeds = np.random.choice(1000, 4, replace=False)
        # data
        # convert data to torch.FloatTensor
        transform = transforms.ToTensor()
        # choose the training and test datasets
        train_data = torchvision.datasets.MNIST(path_mnist, train=True,
                                                download=True, transform=transform)
        test_data = torchvision.datasets.MNIST(path_mnist, train=False,
                                               download=True, transform=transform)
        # prepare data loaders
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=1000,
                                                   num_workers=0)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=1000,
                                                  num_workers=0)
        # prepare data loaders
        images = []
        digits = []
        for batch_idx, (im, lab) in enumerate(train_loader):
            imres = im.reshape(im.numpy().shape[0], 1, Gpar_data['RFsize'], Gpar_data['downscale'],
                               Gpar_data['RFsize'], Gpar_data['downscale'])
            im = np.max(imres.detach().numpy(), axis=(3, 5))
            images.append(im[:, 0, :, :])
            digits.append(lab)
        images = np.concatenate(images)
        digits = np.concatenate(digits)

        if testtoo:
            imagestest = []
            digitstest = []
            for batch_idx, (im, lab) in enumerate(test_loader):
                imres = im.reshape(im.numpy().shape[0], 1, Gpar_data['RFsize'], Gpar_data['downscale'],
                                   Gpar_data['RFsize'], Gpar_data['downscale'])
                im = np.max(imres.detach().numpy(), axis=(3, 5))
                imagestest.append(im[:, 0, :, :])
                digitstest.append(lab)
            imagestest = np.concatenate(imagestest)
            digitstest = np.concatenate(digitstest)

        # add noise
        POS = np.meshgrid(np.arange(0, Gpar_data['RFsize'] * Gpar_data['scal'], Gpar_data['RFsize']),
                          np.arange(0, Gpar_data['RFsize'] * Gpar_data['scal'], Gpar_data['RFsize']))
        POS = np.array([POS[0].ravel(), POS[1].ravel()]).T
        # general data
        if pretraining:
            data, label, _ = get_data2(POS, images, digits, s1=None, s2=None, num=Tpar_pre['Nsamples'],
                                       F=Gpar_data['RFsize'], scal=Gpar_data['scal'],
                                       distract=False, nois=Gpar_data['nois'], task=None, seed=seeds[0])
            if testtoo:
                datatest, labeltest, _ = get_data2(POS, imagestest, digitstest, s1=None, s2=None,
                                               num=Tpar_pre['Nsamples'], F=Gpar_data['RFsize'], scal=Gpar_data['scal'],
                                               distract=False, nois=Gpar_data['nois'], task=None, seed=seeds[1])
            else:
                datatest, labeltest = None, None
            # size of input
            Gpar_data['I'] = data.shape[1]
        else:
            data, label, datatest, labeltest = None, None, None, None
        ###### task data #####
        if task is not None:
            print('TODO: at the moment the distractor digits are never the task-digit, need to make this a flexible par')
            datatask, labeltask, _ = get_data2(POS, images, digits, s1, s2, num=Tpar['Nsamples']*Tpar['Nsets'],
                                                   F=Gpar_data['RFsize'], scal=Gpar_data['scal'],
                                                   distract=True, nois=Gpar_data['nois'], task=loc, seed=seeds[2])
            if testtoo:
                datatasktest, labeltasktest, _ = get_data2(POS, imagestest, digitstest, s1, s2, num=Tpar_pre['Nsamples'],
                                                           F=Gpar_data['RFsize'], scal=Gpar_data['scal'],
                                                           distract=True, nois=Gpar_data['nois'], task=loc, seed=seeds[3])
            else:
                datatasktest, labeltasktest = None, None
    return data, datatest, label, labeltest, datatask, datatasktest, labeltask, labeltasktest

def load_results(NAME_PRE, NAME_TUN, path_save, model, plotit=True, runtests=True):

    model_pre, model_tun, model_tun_gain, model_tun_dec, model_tun_stoch, Tpar_tun, Tpar_pre, \
    Tpar_gain, Tpar_stoch, Npar_pre, Gpar_data, Gpar_data_task, Gpar_net, model_sw, \
    loss_train, loss_test, correct, dinf, names = None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None

    ##################### PRE TRAINING #####################
    MODEL_PRE = []
    GPAR_NET, GPAR_DATA, GPAR_DATA_TASK = [], [], []
    NPAR_PRE, TPAR_PRE = [], []
    count = 0
    for name_pre_mm in NAME_PRE:
        if ospath.exists(path_save + 'results/pretrained_model_' + name_pre_mm + '_Npar.npy'):
            Npar_pre = dict(enumerate(np.load(path_save + 'results/pretrained_model_' + name_pre_mm + '_Npar.npy',
                                              allow_pickle=True).flatten()))[0]
        if ospath.exists(path_save + 'results/pretrained_model_' + name_pre_mm + '_Tpar.npy'):
            Tpar_pre = dict(enumerate(np.load(path_save + 'results/pretrained_model_' + name_pre_mm + '_Tpar.npy',
                                              allow_pickle=True).flatten()))[0]

        # pretraining
        Gpar_data = dict(enumerate(
            np.load(path_save + 'results/' + name_pre_mm + '_Gpar_data.npy', allow_pickle=True).flatten()))[0]
        Gpar_net = dict(enumerate(
            np.load(path_save + 'results/' + name_pre_mm + '_Gpar_net.npy', allow_pickle=True).flatten()))[0]

        if ("masktype" in Npar_pre)==False:
            Npar_pre['masktype'] = 'full'
        if ('pool_fact_h2' in Npar_pre)==False:
            Npar_pre['pool_fact_h2'] = 3
        if ('coupl_redfact' in Gpar_net)==False:
            Gpar_net['coupl_redfact'] = 1  # should be a divisor of 'Npos'
            Gpar_net['coupl_offset'] = .5 # how coupling is centered relative to neurons PO (0=lies on it), in % of difference
            Gpar_net['coupl_sigma'] = [.3, .3, .2] # sigma of gaussian couplign filter
        mask = create_mask(Npar_pre, Gpar_data, Gpar_net=Gpar_net, dtype=torch.float, device='cpu')
        model_pre = load_model(model, Gpar_data, Npar_pre, Gpar_net,
                               path=path_save + 'pretrained_model_' + name_pre_mm, mask=mask)
        print(Npar_pre)
        ##################### TASK TRAINING #####################
        fig, ax = None, None
        LOSS_TRAIN, LOSS_TEST, CORRECT = [], [], []
        for name_tun_mm in NAME_TUN[count]:
            print('loading name_tun_mm')
            if ospath.exists(path_save + 'results/tune_weights_model_' + name_pre_mm + name_tun_mm + '_Npar.npy'):
                Npar_tun = dict(enumerate(
                    np.load(path_save + 'results/tune_weights_model_' + name_pre_mm + name_tun_mm + '_Npar.npy',
                            allow_pickle=True).flatten()))[0]
                print('Npar_tun: ', Npar_tun)
            if ospath.exists(path_save + 'results/tune_weights_model_' + name_pre_mm + name_tun_mm + '_Tpar.npy'):
                Tpar_tun = dict(enumerate(
                    np.load(path_save + 'results/tune_weights_model_' + name_pre_mm + name_tun_mm + '_Tpar.npy',
                            allow_pickle=True).flatten()))[0]
            if ospath.exists(path_save + 'results/tune_gain_model_' + name_pre_mm + name_tun_mm + '_Tpar.npy'):
                Tpar_gain = dict(enumerate(
                    np.load(path_save + 'results/tune_gain_model_' + name_pre_mm + name_tun_mm + '_Tpar.npy',
                            allow_pickle=True).flatten()))[0]
            if ospath.exists(path_save + 'results/tune_stoch_model_' + name_pre_mm + name_tun_mm + '_Tpar.npy'):
                Tpar_stoch = dict(enumerate(
                    np.load(path_save + 'results/tune_stoch_model_' + name_pre_mm + name_tun_mm + '_Tpar.npy',
                            allow_pickle=True).flatten()))[0]
            if ospath.exists(path_save + 'results/' + name_pre_mm + name_tun_mm + '_Gpar_data_task.npy'):
                Gpar_data_task = dict(enumerate(
                    np.load(path_save + 'results/' + name_pre_mm + name_tun_mm + '_Gpar_data_task.npy',
                            allow_pickle=True).flatten()))[0]
                print(Gpar_data_task)
            else:
                print('Gpar_data_task does not exist, use Gpar_data')
                Gpar_data_task = Gpar_data

            for tt in range(Gpar_data_task['Ttasks']):
                # informativeness
                if ospath.exists(path_save + 'results/' + name_pre_mm + name_tun_mm + '_task' + np.str(tt) + '_dinf.npy'):
                    dinf = np.load(path_save + 'results/' + name_pre_mm + name_tun_mm + '_task' + np.str(tt) + '_dinf.npy')
                ################# MODELS ######################
                if ospath.exists(path_save + 'pretrained_model_' + name_pre_mm + name_tun_mm + '_switched.pt'):
                    model_sw = load_model(model, Gpar_data, Npar_pre, Gpar_net,
                                          path=path_save + 'pretrained_model_' + name_pre_mm + name_tun_mm + '_switched', mask=mask)
                else:
                    print('no retrained model')
                    model_sw = None
                # tuned weights
                if ospath.exists(
                        path_save + 'tune_weights_model_' + name_pre_mm + name_tun_mm + '_task' + np.str(tt) + '.pt'):
                    model_tun = load_model(model, Gpar_data, Npar_pre, Gpar_net,
                                           path=path_save + 'tune_weights_model_' + name_pre_mm + name_tun_mm + '_task' + np.str(
                                               tt), mask=mask)
                else:
                    print('no model_tun')
                    model_tun = None
                # MG
                if ospath.exists(
                        path_save + 'tune_stoch_model_' + name_pre_mm + name_tun_mm + '_task' + np.str(tt) + '.pt'):
                    model_tun_stoch = load_model(model, Gpar_data, Npar_pre, Gpar_net,
                                                 path=path_save + 'tune_stoch_model_' + name_pre_mm + name_tun_mm + '_task' + np.str(
                                                     tt), mask=mask)
                else:
                    print('no model_tun_stoch')
                    model_tun_stoch = None
                # gain
                if ospath.exists(path_save + 'tune_gain_model_' + name_pre_mm + name_tun_mm + '_task' + np.str(tt) + '.pt'):
                    model_tun_gain = load_model(model, Gpar_data, Npar_pre, Gpar_net,
                                                path=path_save + 'tune_gain_model_' + name_pre_mm + name_tun_mm + '_task' + np.str(
                                                    tt), mask=mask)
                else:
                    print('no model_tun_gain')
                    model_tun_gain = None
                # decoder
                if ospath.exists(path_save + 'tune_dec_model_' + name_pre_mm + name_tun_mm + '_task' + np.str(tt) + '.pt'):
                    model_tun_dec = load_model(model, Gpar_data, Npar_pre, Gpar_net,
                                               path=path_save + 'tune_dec_model_' + name_pre_mm + name_tun_mm + '_task' + np.str(
                                                   tt), mask=mask)
                else:
                    print('no model_tun_dec')
                    model_tun_dec = None

                # running unit tests:
                if runtests:
                    if model_sw is None:
                        unit_tests(model_pre, model_tun, model_tun_dec, model_tun_gain, model_tun_stoch, Npar_pre)
                    else:
                        unit_tests(model_sw, model_tun, model_tun_dec, model_tun_gain, model_tun_stoch, Npar_pre)

                # training results
                loss_train, loss_test, correct = [], [], []
                names = ['weights', 'gain', 'stoch', 'dec']
                found = 0
                for name_nn in names:
                    if ospath.exists(path_save + 'results/' + 'tune_' + name_nn + '_model_' + name_pre_mm + name_tun_mm + \
                                     '_task' + np.str(tt) + '_loss_train.npy'):
                        loss_train.append(
                            np.load(path_save + 'results/' + 'tune_' + name_nn + '_model_' + name_pre_mm + name_tun_mm + \
                                    '_task' + np.str(tt) + '_loss_train.npy', allow_pickle=True))
                    else:
                        loss_train.append([None])
                    if ospath.exists(path_save + 'results/' + 'tune_' + name_nn + '_model_' + name_pre_mm + name_tun_mm + \
                                     '_task' + np.str(tt) + '_loss_test.npy'):
                        try:
                            loss_test.append(
                                np.load(path_save + 'results/' + 'tune_' + name_nn + '_model_' + name_pre_mm + name_tun_mm + \
                                        '_task' + np.str(tt) + '_loss_test.npy'))
                        except ValueError:
                            print('could not load test loss for ', name_nn)
                            loss_test.append([None])
                    else:
                        loss_test.append([None])
                    if ospath.exists(path_save + 'results/' + 'tune_' + name_nn + '_model_' + name_pre_mm + name_tun_mm + \
                                     '_task' + np.str(tt) + '_correct_test.npy'):
                        try:
                            correct.append(
                                np.load(path_save + 'results/' + 'tune_' + name_nn + '_model_' + name_pre_mm + name_tun_mm + \
                                        '_task' + np.str(tt) + '_correct_test.npy'))
                        except ValueError:
                            print('could not load correct for ', name_nn)
                            correct.append([None])
                        found += 1
                    else:
                        correct.append([None])

                if plotit:
                    fig, ax = vis_res(Npar=Tpar_tun, mmean=[0, 1, 1, 0], data=None, label=None,
                                        names=names, loss_train=loss_train, loss_test=loss_test,
                                        correct=correct, dtype=torch.float, fig=fig, ax=ax, Gpar_data_task=Gpar_data_task,
                                        fig_label = name_pre_mm + name_tun_mm + '_task' + np.str(tt))
                    #ax[-1].set_title(name_pre_mm + name_tun_mm + '_task' + np.str(tt))
                    ax[-1].legend()
                for jj in range(4):
                    LOSS_TRAIN.append(np.zeros([4, 100])*np.nan)
                    LOSS_TEST.append(np.zeros([4, 100])*np.nan)
                    CORRECT.append(np.zeros([4, 100])*np.nan)
                    if loss_train[jj] is not None:
                        LOSS_TRAIN[-1][jj, :len(loss_train[jj])] = loss_train[jj]
                        LOSS_TEST[-1][jj, :len(loss_test[jj])] = loss_test[jj]
                        CORRECT[-1][jj, :len(correct[jj])] = correct[jj]
        MODEL_PRE.append(model_pre)
        GPAR_NET.append(Gpar_net)
        GPAR_DATA.append(Gpar_data)
        GPAR_DATA_TASK.append(Gpar_data_task)
        NPAR_PRE.append(Npar_pre)
        TPAR_PRE.append(Tpar_pre)
        count +=1
        if ((len(NAME_TUN[count-1]) > 1)|(Gpar_data_task['Ttasks']>1))&(Tpar_tun is not None)&plotit:
            fig2, ax2 = plt.subplots(1, 3, figsize=(10, 3))

            for mm in range(4):
                loss_train = np.stack(LOSS_TRAIN)[:,mm,:(np.max(np.sum(np.isnan(np.stack(LOSS_TRAIN)[:,mm,:])==False,axis=1)))]
                loss_test = np.stack(LOSS_TEST)[:,mm,:(np.max(np.sum(np.isnan(np.stack(LOSS_TEST)[:,mm,:])==False,axis=1)))]
                correct = np.stack(CORRECT)[:,mm,:(np.max(np.sum(np.isnan(np.stack(CORRECT)[:,mm,:])==False,axis=1)))]*100
                if loss_train.shape[1]>0:
                    ax2[0].fill_between(np.arange(Tpar_tun['Nsamples'], Tpar_tun['Nsamples'] * (Tpar_tun['Nsets'] + 1), Tpar_tun['Nsamples']),
                                       np.nanmean(loss_train, axis=0) - np.sqrt(np.nanvar(loss_train, axis=0)),
                                       np.nanmean(loss_train, axis=0) + np.sqrt(np.nanvar(loss_train, axis=0)),
                                       alpha=.2, color=dec_colors[names[mm]]);
                if loss_test.shape[1]>0:
                    ax2[1].fill_between(np.arange(0, Tpar_tun['Nsamples'] * (Tpar_tun['Nsets'] + 1), Tpar_tun['Nsamples']),
                                   np.nanmean(loss_test, axis=0) - np.sqrt(np.nanvar(loss_test, axis=0)),
                                   np.nanmean(loss_test, axis=0) + np.sqrt(np.nanvar(loss_test, axis=0)),
                                   alpha=.2, color=dec_colors[names[mm]]);
                if correct.shape[1]>0:
                    ax2[2].fill_between(np.arange(0, Tpar_tun['Nsamples'] * (Tpar_tun['Nsets'] + 1), Tpar_tun['Nsamples']),
                                   np.nanmean(correct, axis=0) - np.sqrt(np.nanvar(correct, axis=0)),
                                   np.nanmean(correct, axis=0) + np.sqrt(np.nanvar(correct, axis=0)),
                                    alpha=.2, color=dec_colors[names[mm]]);
            for mm in range(4):
                loss_train = np.stack(LOSS_TRAIN)[:,mm,:(np.max(np.sum(np.isnan(np.stack(LOSS_TRAIN)[:,mm,:])==False,axis=1)))]
                loss_test = np.stack(LOSS_TEST)[:,mm,:(np.max(np.sum(np.isnan(np.stack(LOSS_TEST)[:,mm,:])==False,axis=1)))]
                correct = np.stack(CORRECT)[:,mm,:(np.max(np.sum(np.isnan(np.stack(CORRECT)[:,mm,:])==False,axis=1)))]*100
                if loss_train.shape[1]>0:
                    ax2[0].plot(np.arange(Tpar_tun['Nsamples'], Tpar_tun['Nsamples'] * (Tpar_tun['Nsets'] + 1), Tpar_tun['Nsamples']),
                           np.nanmean(loss_train, axis=0), color=dec_colors[names[mm]], linewidth=2);
                if loss_test.shape[1]>0:
                    ax2[1].plot(np.arange(0, Tpar_tun['Nsamples'] * (Tpar_tun['Nsets'] + 1), Tpar_tun['Nsamples']),
                           np.nanmean(loss_test, axis=0), color=dec_colors[names[mm]], linewidth=2);
                if correct.shape[1]>0:
                    ax2[2].plot(np.arange(0, Tpar_tun['Nsamples'] * (Tpar_tun['Nsets'] + 1), Tpar_tun['Nsamples']),
                           np.nanmean(correct, axis=0), color=dec_colors[names[mm]], linewidth=2);
            ax2[2].set_title('% correct')
            ax2[2].set_xlabel('# data points')
            ax2[2].set_ylim(-1,101)
            ax2[2].set_xlim(-1,301)

    return MODEL_PRE, model_tun, model_tun_gain, model_tun_dec, model_tun_stoch, Tpar_tun, TPAR_PRE, \
           Tpar_gain, Tpar_stoch, NPAR_PRE, GPAR_DATA, GPAR_DATA_TASK, GPAR_NET, model_sw, \
           LOSS_TRAIN, LOSS_TEST, CORRECT, dinf, names

def create_mask(Npar_pre, Gpar_data, Gpar_net, dtype, device): #, second_hidden = 'full'):
    mask = [torch.ones([Gpar_data['I'], Npar_pre['N']], dtype=dtype, device=device)]
    if Npar_pre['masktype']=='full':
        print('create alltoall first hidden layer')
        mask.append(torch.ones([Npar_pre['N'], Npar_pre['Nh']], dtype=dtype, device=device))
        for nn in range(1, Npar_pre['Nhidden']):
            mask.append(torch.ones([Npar_pre['Nh'], Npar_pre['Nh']], dtype=dtype, device=device))
    elif Npar_pre['masktype']=='sparse':
        print('create sparse first hidden layer')
        pos = np.arange(Gpar_net['Npos'])
        orien = np.arange(Gpar_net['Norien'])
        ### first hidden layer mask ###
        mask_h = []
        for yy in pos[:-1]:
            for xx in pos[:-1]:
                for oo in orien:
                    mask_tt = np.zeros([len(pos), len(pos), len(orien)])
                    mask_tt[xx:(xx + 2), yy, oo] = 1
                    mask_tt[xx:(xx + 2), yy + 1, oo] = 1
                    mask_h.append(mask_tt.ravel())

        mask.append(torch.tensor(np.repeat(np.stack(mask_h).T, Npar_pre['Hfact'], axis=1),
                                 dtype=dtype, device=device))  # .to_sparse()

        ### second hidden layer mask ###
        if (Npar_pre['Nhidden']>1): 
            print('create sparse second hidden layer')
            mask_h2 = []
            for yy in pos[:-(Npar_pre['pool_fact_h2']-1)]:
                for xx in pos[:-(Npar_pre['pool_fact_h2']-1)]:
                    for oo in orien:
                        mask_tt = np.zeros([len(pos) - 1, len(pos) - 1, len(orien)])
                        # pool over locations
                        for pp in range(Npar_pre['pool_fact_h2']-1):
                            mask_tt[xx:(xx + (Npar_pre['pool_fact_h2']-1)), yy+pp, oo] = 1
                        mask_h2.append(np.repeat(mask_tt.ravel(), Npar_pre['Hfact']))
            mask.append(torch.tensor(np.repeat(np.stack(mask_h2).T, Npar_pre['Hfact'], axis=1),
                                 dtype=dtype, device=device)) # .to_sparse()
            if Npar_pre['Nhidden'] > 2:
                print('create sparse third hidden layer')
                mask_h3 = []
                for yy in pos[:-Npar_pre['pool_fact_h2']]:
                    for xx in pos[:-Npar_pre['pool_fact_h2']]:
                        for oo in orien:
                            mask_tt = np.zeros([len(pos) - 2, len(pos) - 2, len(orien)])
                            # pool over locations
                            for pp in range(Npar_pre['pool_fact_h2'] - 1):
                                mask_tt[xx:(xx + (Npar_pre['pool_fact_h2'] - 1)), yy + pp, oo] = 1
                            mask_h3.append(np.repeat(mask_tt.ravel(), Npar_pre['Hfact']))
                mask.append(torch.tensor(np.repeat(np.stack(mask_h3).T, Npar_pre['Hfact'], axis=1),
                                         dtype=dtype, device=device))  # .to_sparse()
                if Npar_pre['Nhidden'] > 3:
                    print('create alltoall rest of hidden layers')
                    for nn in range(2, Npar_pre['Nhidden']):
                        mask.append(torch.ones([mask[-1].shape[1], Npar_pre['Nh']], dtype=dtype, device=device))

    return mask

