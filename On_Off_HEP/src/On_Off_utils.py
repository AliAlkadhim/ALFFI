#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np; import pandas as pd
import scipy as sp; import scipy.stats as st
import torch; import torch.nn as nn
#use numba's just-in-time compiler to speed things up
from numba import njit
from sklearn.preprocessing import StandardScaler; from sklearn.model_selection import train_test_split
import matplotlib as mp; import matplotlib.pyplot as plt; 
#reset matplotlib stle/parameters
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
plt.style.use('seaborn-deep')
mp.rcParams['agg.path.chunksize'] = 10000
font_legend = 15; font_axes=15
# %matplotlib inline
import copy; import sys; import os
try:
    import optuna
except Exception:
    print('optuna is only used for hyperparameter tuning, not critical!')
    pass
#sometimes jupyter doesnt initialize MathJax automatically for latex, so do this
#import ipywidgets as wid; wid.HTMLMath('$\LaTeX$')



def import_base_stack():
    import numpy as np; import pandas as pd
    import scipy as sp; from numba import njit

def DR(s, theta):
    return sp.special.gammainc(s, theta)


def DL(s, theta):
    return 1 - sp.special.gammainc(s+1, theta)


def L_prof(n,m,theta):
    k=1
    k1 = k+1
    k2 = 0.5/k1
    g = n+m - k1*theta
    nu_hat = k2* (g+ np.sqrt(g*g +4*k1*m*theta))
    p1 = st.poisson.pmf(n, mu = theta + nu_hat)
    p2 = st.poisson.pmf(m, mu = k * nu_hat)
    
    return p1*p2


def theta_hat(n,m, MLE):
    theta_hat = n-m
    
    if not MLE:
        theta_hat = theta_hat * (theta_hat > 0)
    return theta_hat

# @njit
def lambda_test(theta, n, m, MLE):
    Ln = L_prof(n,m,theta)
    Ld = L_prof(n,m, theta_hat(n,m, MLE))
    eps=1e-15
    Ld=Ld+eps
    lambda_  = -2*np.log(Ln/Ld)
    return np.array(lambda_)


chi2_exp_size=40000

def run_sim(theta, nu, MLE, lambda_size):
    """Sample n ~ Pois(theta+nu), 
              m ~ Pois(nu), 
    and compute 
              lambda(theta, n, m)
              
    return: (n, m, lambda_), where each are np arrays of length lambda_size
    """
    n = st.poisson.rvs(theta+nu, size=lambda_size)
    m = st.poisson.rvs(nu, size=lambda_size)
    lambda_ = lambda_test(theta, n, m, MLE=MLE)
    return (n, m, lambda_)

def run_sims(points, MLE):
    """
    Run an entire simulation, that is, generate n and m from 
    run_sim above, and calculate lambda, for
    
    input: a tuple of (theta, nu) scalars
    
    Reurns:df, lambda_results
    
    where lambda_results is a list of tuples 
        (n, m, lambda_, theta, nu)
    and df is just a dataframe of [n,m,lambda,theta,nu]

    """
    lambda_results=[]
    df=pd.DataFrame()
    for p in points:
        theta, nu = p
        df['theta']=theta
        df['nu']=nu
        n, m, lambda_ = run_sim(theta, nu, MLE, lambda_size =chi2_exp_size)
        df['n'] = n
        df['m'] = m
        df['lambda']=lambda_
        lambda_results.append((n, m, lambda_, theta, nu))
    
        print( '\n \n (theta, nu) =  (%.f, %.f) \n ' % (theta, nu) )
        print(f'\t \t with associated n =  {n}, \n \n \t \t m = {m}, \n \n \t \t lambda = {lambda_}'  )
    return df, lambda_results

def plot_one(lambda_, theta, nu, ax):
    """Histogram the CDF of  lambda_t = -2log(Lp(theta)/Lp(theta_hat)), 
    for a given (fixed) theta and nu.
    Also, plot the actual CDF of a chi^2 distribution with 1 free parameter 
    (since only theta is left after we profile nu) """
    ftsize = 16; xmin= 0; xmax= 10
    ymin= 0; ymax= 1
    x_range = (xmin, xmax)
    y_range = (ymin, ymax)
    ax.set_xlim(x_range); ax.set_ylim(y_range)
    ax.set_xlabel(r'$\lambda \left(\theta,\hat{\nu}(\theta) \mid n, m \right)$',fontsize=ftsize)
    ax.set_ylabel(r'cdf$(\lambda)$', fontsize=ftsize)
    ##########HISTOGRAM CDF OF LAMBDA####################
    ax.hist(lambda_, bins=5*xmax, range=x_range,
    color=(0.8,0.8,0.9),
    density=True, cumulative=True,
    histtype='stepfilled', edgecolor='black', label=r'CDF$(\lambda)$')
    ############################################################
    ########### HISTOGRAM CDF OF THE CHI2 OF OF X WITH 1 DOF
    #x is not theta, that's the whole point of Wilks thm, x is an arbitrary RV
    x = np.arange(0, xmax, 0.2)
    y = st.chi2.cdf(x, 1)
    ax.plot(x, y, color='blue',
    linewidth=2, label=r'CDF$(\chi^2_1)$')
    # annotate
    xwid = (xmax-xmin)/12
    ywid = (ymax-ymin)/12
    xpos = xmin + xwid/2
    ypos = ymin + ywid*2
    ax.text(xpos, ypos,
    r'$ \theta = %d, \nu = %d$' % (theta, nu),
    fontsize=ftsize)
    ax.legend(loc='upper right',fontsize=15)
    

def observe_test_statistic_pivotality(points, lambda_size, savefig=False):
    """Histogram the CDF of  lambda_t = -2log(Lp(theta)/Lp(theta_hat)), 
    for a given (fixed) theta and nu.
    Also, plot the actual CDF of a chi^2 distribution with 1 free parameter 
    (since only theta is left after we profile nu) """
    fig, ax = plt.subplots(1,2, figsize=(10,5), sharey=True)
    plt.subplots_adjust(wspace=0.3)
    title_size=15
    for point in points:
        _, _, lambda_MLE = run_sim(theta=point[0], nu=point[1], MLE=True, lambda_size=chi2_exp_size)


        ftsize = 16; xmin= 0; xmax= 10
        ymin= 0; ymax= 1
        x_range = (xmin, xmax)
        y_range = (ymin, ymax)
        
        ax[0].hist(lambda_MLE, bins=5*xmax, range=x_range,
        # color=(0.8,0.8,0.9),
                   alpha=0.3,
        density=True, cumulative=True,
        histtype='stepfilled', edgecolor='black', 
        label=r'CDF$\left(\lambda_{NP}(\theta=%s,\nu=%s ) \right)$' % (str(point[0]), str(point[1])) )
        
        ############################################################
        ########### HISTOGRAM CDF OF THE CHI2 OF OF X WITH 1 DOF
        #x is not theta, that's the whole point of Wilks thm, x is an arbitrary RV
        x = np.arange(0, xmax, 0.2)
        y = st.chi2.cdf(x, 1)
        ax[0].plot(x, y, 
                   # color='blue',
                    linewidth=1, 
                   label=r'CDF$ \left(\chi^2_1(\theta=%s) \right)$' % str(point[0]))
        
        ax[0].set_title('MLE=True',fontsize=title_size)
        #####################Do the same for non-MLE
        
            
    for point in points:
        _, _, lambda_nonMLE = run_sim(theta=point[0], nu=point[1], MLE=False, lambda_size=chi2_exp_size)


        ftsize = 16; xmin= 0; xmax= 10
        ymin= 0; ymax= 1
        x_range = (xmin, xmax)
        y_range = (ymin, ymax)
        
        ax[1].hist(lambda_nonMLE, bins=5*xmax, range=x_range,
        # color=(0.8,0.8,0.9),
                   alpha=0.3,
        density=True, cumulative=True,
        histtype='stepfilled', edgecolor='black', 
        label=r'CDF$\left(\lambda_{NP}(\theta=%s,\nu=%s ) \right)$' % (str(point[0]), str(point[1])) )
        
        ############################################################
        ########### HISTOGRAM CDF OF THE CHI2 OF OF X WITH 1 DOF
        #x is not theta, that's the whole point of Wilks thm, x is an arbitrary RV
        x = np.arange(0, xmax, 0.2)
        y = st.chi2.cdf(x, 1)
        ax[1].plot(x, y, 
                   # color='blue',
                    linewidth=1, 
                   label=r'CDF$ \left(\chi^2_1(\theta=%s) \right)$' % str(point[0]))
        
        
        ax[1].set_title('MLE=False',fontsize=title_size)
        
        # annotate
        xwid = (xmax-xmin)/12
        ywid = (ymax-ymin)/12
        xpos = xmin + xwid/2
        ypos = ymin + ywid*2
        
        
        
        
        
    for i in range(2):
        ax[i].set_xlabel(r'$\lambda_{NP} \left(\theta,\hat{\nu}(\theta) \mid n, m \right)$',fontsize=ftsize)
        ax[i].set_ylabel(r'cdf$(\lambda_{NP})$', fontsize=ftsize)
        ax[i].legend(loc='lower right',fontsize=9)
        ax[i].set_xlim(x_range)
        ax[i].set_ylim(y_range)
        
        
    if savefig:
        plt.savefig(LFI_BASE+'images/lambda_NP_observe_pivotality.png')
        
    plt.show()
    
    


def generate_training_data(Bprime, MLE, save_data=False):
    """Generate the training data, that is, features=[theta, nu, N, M], targets=Z"""
    #sample theta and nu from uniform(0,20)
    theta = st.uniform.rvs(thetaMin, thetaMax, size=Bprime)
    # nu = st.uniform.rvs(nuMin, nuMax, size=Bprime)
    nu= st.uniform.rvs(numin, numax, size=Bprime)
    #n,m ~ F_{\theta,\nu}, ie our simulator. sample n from a Poisson with mean theta+nu 
    n = st.poisson.rvs(theta+ nu, size=Bprime)
    #sample m from a poisson with mean nu
    m = st.poisson.rvs(nu, size=Bprime)
    #sample our observed counts (N,M), which take the place of D
    N = np.random.randint(Nmin, Nmax, size=Bprime)
    M = np.random.randint(Mmin, Mmax, size=Bprime)
    theta_hat_ = theta_hat(N,M, MLE)
    SUBSAMPLE=10
    print('n=', n[:SUBSAMPLE])
    print('m=', m[:SUBSAMPLE])
    print('N=', N[:SUBSAMPLE])
    print('M=', M[:SUBSAMPLE])
    lambda_gen = lambda_test(theta, n, m, MLE)
    print('lambda_gen= ', lambda_gen[:SUBSAMPLE])
    lambda_D = lambda_test(theta, N, M, MLE)
    print('lambda_D= ', lambda_D[:SUBSAMPLE])
    #if lambda_gen <= lambda_D: Z=1, else Z=0
    Z = (lambda_gen <= lambda_D).astype(np.int32)
    
    data_2_param = {'Z' : Z, 'theta' : theta, 'nu': nu, 'theta_hat': theta_hat_, 'N':N, 'M':M}

    data_2_param = pd.DataFrame.from_dict(data_2_param)
    if save_data:
        data_2_param.to_csv(LFI_BASE+'data/two_parameters_theta_%s_%s_%sk_Examples_MLE_%s.csv' %                            (str(thetaMin), str(thetaMax), str(int(Bprime/1000)), str(MLE)) )

    print('\n')
    print(data_2_param.describe())
    return data_2_param


def make_hist_data(Bprime,
              thetamin, thetamax,
              nu, N, M,
                nbins,
             MLE=True):

    theta = st.uniform.rvs(thetamin, thetamax, size=Bprime)
    n = st.poisson.rvs(theta + nu, size=Bprime)
    m = st.poisson.rvs(nu, size=Bprime)
    
    Z = (lambda_test(theta, n, m, MLE=MLE) < 
         lambda_test(theta, N, M, MLE=MLE)).astype(np.int32)

    thetarange = (thetamin, thetamax)
    # bins = binsize(Bprime)

    # weighted histogram   (count the number of ones per bin)
    y1, bb = np.histogram(theta, 
                          bins=nbins, 
                          range=thetarange, 
                          weights=Z)
    
    # unweighted histogram (count number of ones and zeros per bin)
    yt, _ = np.histogram(theta, 
                         bins=nbins, 
                         range=thetarange)

    y =  y1 / yt    
    
    return y, bb


def plot_one_hist(Bprime, thetamin, thetamax, nu, N, M, MLE, nbins, ax):
    counts, bins= make_hist_data(Bprime,
              thetamin, thetamax,
              nu, N, M,
            nbins,
             MLE)
    bin_centers = (bins[1:]+bins[:-1])/2
    ax.plot(bin_centers, counts, label= r'$\mathbf{h}$ Example', lw=3)
    ax.set_xlabel(r'$\theta$',fontsize=font_axes)
    ax.set_ylabel(r'$E(Z|\theta,\nu)$',fontsize=font_axes)
    ax.legend(loc='center right',fontsize=font_legend)
    
    
def plot_data_one_nu(Bprime, thetamin, thetamax, nu, D, MLE, 
              FONTSIZE=15,
              func=None,
              fgsize=(10, 6)):
    
    # make room for 6 sub-plots
    fig, ax = plt.subplots(nrows=2, 
                           ncols=3, 
                           figsize=fgsize)
    
    # padding
    plt.subplots_adjust(hspace=0.01)
    plt.subplots_adjust(wspace=0.20)
    
    # use flatten() to convert a numpy array of 
    # shape (nrows, ncols) to a 1-d array. 
    ax = ax.flatten()
    
    for j, (N, M) in enumerate(D):
        
        y, bb = make_hist_data(Bprime,
                              thetamin, thetamax,
                              nu, N, M,
                              nbins=200,
                              MLE=True)
    
        ax[j].set_xlim(thetamin, thetamax-5)
        ax[j].set_ylim(0, 1)
        ax[j].set_xlabel(r'$\theta$', fontsize=FONTSIZE)
        ax[j].set_ylabel(r'$E(Z|\theta, \nu)$', fontsize=FONTSIZE)
        
        x = (bb[1:]+bb[:-1])/2
        ax[j].plot(x, y, 'b', lw=2, label='$\mathbf{h}$, MLE', alpha=0.3)
        #h is histogram approximation

        y_nonMLE, bb_nonMLE = make_hist_data(Bprime,
                              thetamin, thetamax,
                              nu, N, M,
                              nbins=200,
                              MLE=False)
        
        
        x_nonMLE = (bb_nonMLE[1:]+bb_nonMLE[:-1])/2
        ax[j].plot(x_nonMLE, y_nonMLE, 'r', lw=2, label='$\mathbf{h}$, non-MLE',alpha=0.3)
        
        
        if func:
            p, _ = func(nu, N, M)
            ax[j].plot(x, p, 'r', lw=2, label='f')
            #f is model approximation
        
        ax[j].grid(True, which="both", linestyle='-')
        ax[j].text(5.1, 0.42, r'$N, M = %d, %d$' % (N, M), fontsize=font_legend-3
                   # fontsize=FONTSIZE
                  ) 

        ax[j].text(5.1, 0.30, r'$\nu = %5.1f$' % nu, fontsize=font_legend-3
                   # fontsize=FONTSIZE
                  ) 

        ax[j].legend(loc='upper right',fontsize=font_legend-3)
        
    # hide unused sub-plots
    for k in range(j+1, len(ax)):
        ax[k].set_visible(False)
    
    plt.tight_layout()
    plt.show()

def getwholedata(MLE_or_nonMLE, valid=False):
    if MLE:
        data = pd.read_csv(LFI_BASE+'data/two_parameters_theta_0_20_1000k_Examples_MLE_True.csv', 
                     # nrows=SUBSAMPLE,
                     usecols=['theta', 'nu', 'theta_hat', 'N', 'M']
                    )
        
    else:
        data = pd.read_csv(LFI_BASE+'data/two_parameters_theta_0_20_1000k_Examples_MLE_False.csv', 
             # nrows=SUBSAMPLE,
             usecols=['theta', 'nu', 'theta_hat', 'N', 'M']
            )
    train_data, test_data = train_test_split(data, test_size=0.2)
    #split the train data (0.8 of whole set) again into 0.8*0.8=0.64 of whole set
    

    train_data = train_data.reset_index(drop=True)
    test_data  = test_data.reset_index(drop=True)

    target='Z'
    source = ['theta','nu','theta_hat','N','M']

    train_t, train_x = split_t_x(train_data, target=target, source=source)
    test_t,  test_x  = split_t_x(test_data,  target=target, source=source)
    print('train_t shape = ', train_t.shape, '\n')
    print('train_x shape = ', train_x.shape, '\n')
    
    if valid:
        #if you want to also make a validation data set
        train_data, valid_data = train_test_split(train_data, test_size=0.2)
        valid_data = valid_data.reset_index(drop=True)
        valid_t, valid_x = split_t_x(valid_data, target=target, source=source)

        
    return train_t, train_x, test_t,  test_x



class Model(nn.Module):
    
    def __init__(self, n_inputs=4, n_nodes=20, n_layers=5):

        # call constructor of base (or super, or parent) class
        super(Model, self).__init__()

        self.layers = []
        
        # create input layer
        self.layer0 = nn.Linear(n_inputs, n_nodes)
        self.layers.append(self.layer0)

        # create "hidden" layers
        for l in range(1, n_layers):
            cmd = 'self.layer%d = nn.Linear(%d, %d)' %             (l, n_nodes, n_nodes)
            exec(cmd)
            cmd = 'self.layers.append(self.layer%d)' % l
            exec(cmd)
          
        # create output layer
        cmd = 'self.layer%d = nn.Linear(%d, 1)' % (n_layers, n_nodes)
        exec(cmd)
        cmd = 'self.layers.append(self.layer%d)' % n_layers
        exec(cmd)

    # define (required) method to compute output of network
    def forward(self, x):
        y = x
        for layer in self.layers[:-1]:
            y = layer(y)
            y = torch.relu(y)
        y = self.layers[-1](y)
        y = torch.sigmoid(y)
        return y

def average_quadratic_loss(f, t, x):
    # f and t must be of the same shape
    return  torch.mean((f - t)**2)
def average_loss(f, t):
    # f and t must be of the same shape
    return  torch.mean((f - t)**2)

def validate(model, avloss, inputs, targets):
    # make sure we set evaluation mode so that any training specific
    # operations are disabled.
    model.eval() # evaluation mode
    
    with torch.no_grad(): # no need to compute gradients wrt. x and t
        x = torch.from_numpy(inputs).float()
        t = torch.from_numpy(targets).float()
        # remember to reshape!
        o = model(x).reshape(t.shape)
    return avloss(o, t, x)


def get_features_training_batch(x, t, batch_size):
    # the numpy function choice(length, number)
    # selects at random "batch_size" integers from 
    # the range [0, length-1] corresponding to the
    # row indices.
    rows    = np.random.choice(len(x), batch_size)
    batch_x = x[rows]
    batch_t = t[rows]
    # batch_x.T[-1] = np.random.uniform(0, 1, batch_size)
    return (batch_x, batch_t)


def train(model, optimizer, avloss,
          batch_size, 
          n_iterations, traces, 
          step, window, MLE):
    
    # to keep track of average losses
    xx, yy_t, yy_v, yy_v_avg = traces
    

    
    if MLE:
        train_t, train_x, test_t,  test_x = getwholedata(MLE_or_nonMLE=True, valid=False)
    else:
        train_t, train_x, test_t,  test_x = getwholedata(MLE_or_nonMLE=False, valid=False)
        
    n = len(test_x)
    print('Iteration vs average loss')
    print("%10s\t%10s\t%10s" %           ('iteration', 'train-set', 'valid-set'))
    
    # training_set_features, training_set_targets, evaluation_set_features, evaluation_set_targets = get_data_sets(simulate_data=False, batchsize=batch_size)
    
    for ii in range(n_iterations):

        # set mode to training so that training specific 
        # operations such as dropout are enabled.

        
        model.train()
        
        # get a random sample (a batch) of data (as numpy arrays)
        
        #Harrison-like Loader
        batch_x, batch_t = get_features_training_batch(train_x, train_t, batch_size)
        
        #Or Ali's Loader
        # batch_x, batch_t = next(training_set_features()), next(training_set_targets())
        # batch_x_eval, batch_t_eval = next(evaluation_set_features()), next(evaluation_set_targets())

        with torch.no_grad(): # no need to compute gradients 
            # wrt. x and t
            x = torch.from_numpy(batch_x).float()
            t = torch.from_numpy(batch_t).float()      


        outputs = model(x).reshape(t.shape)
   
        # compute a noisy approximation to the average loss
        empirical_risk = avloss(outputs, t, x)
        
        # use automatic differentiation to compute a 
        # noisy approximation of the local gradient
        optimizer.zero_grad()       # clear previous gradients
        empirical_risk.backward()   # compute gradients
        
        # finally, advance one step in the direction of steepest 
        # descent, using the noisy local gradient. 
        optimizer.step()            # move one step
        
        if ii % step == 0:
            
            
            #using Harrison-like loader
            acc_t = validate(model, avloss, train_x[:n], train_t[:n]) 
            acc_v = validate(model, avloss, test_x[:n], test_t[:n])
            
            #using Ali's loader
            # acc_t = validate(model, avloss, batch_x, batch_t) 
            # acc_v = validate(model, avloss, batch_x_eval, batch_t_eval)
            

            yy_t.append(acc_t)
            yy_v.append(acc_v)
            
            # compute running average for validation data
            len_yy_v = len(yy_v)
            if   len_yy_v < window:
                yy_v_avg.append( yy_v[-1] )
            elif len_yy_v == window:
                yy_v_avg.append( sum(yy_v) / window )
            else:
                acc_v_avg  = yy_v_avg[-1] * window
                acc_v_avg += yy_v[-1] - yy_v[-window-1]
                yy_v_avg.append(acc_v_avg / window)
                        
            if len(xx) < 1:
                xx.append(0)
                print("%10d\t%10.6f\t%10.6f" %                       (xx[-1], yy_t[-1], yy_v[-1]))
            else:
                xx.append(xx[-1] + step)
                    
                print("\r%10d\t%10.6f\t%10.6f\t%10.6f" %                           (xx[-1], yy_t[-1], yy_v[-1], yy_v_avg[-1]), 
                      end='')
            
    print()      
    return (xx, yy_t, yy_v, yy_v_avg)

def plot_average_loss(traces, ftsize=18,save_loss_plots=False):
    
    xx, yy_t, yy_v, yy_v_avg = traces
    
    # create an empty figure
    fig = plt.figure(figsize=(6, 4.5))
    fig.tight_layout()
    
    # add a subplot to it
    nrows, ncols, index = 1,1,1
    ax  = fig.add_subplot(nrows,ncols,index)

    ax.set_title("Average loss")
    
    ax.plot(xx, yy_t, 'b', lw=2, label='Training')
    ax.plot(xx, yy_v, 'r', lw=2, label='Validation')
    #ax.plot(xx, yy_v_avg, 'g', lw=2, label='Running average')

    ax.set_xlabel('Iterations', fontsize=ftsize)
    ax.set_ylabel('average loss', fontsize=ftsize)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, which="both", linestyle='-')
    ax.legend(loc='upper right')
    if save_loss_plots:
        plt.savefig(LFI_BASE+'images/loss_curves/IQN_'+N+T+'_Consecutive_2.png')
        print('\nloss curve saved in images/loss_curves/IQN_'+N+target+'_Consecutive.png')
    # if show_loss_plots:
    plt.show()

class RegularizedRegressionModel(nn.Module):
    #inherit from the super class
    def __init__(self, nfeatures, ntargets, nlayers, hidden_size, dropout):
        super().__init__()
        layers = []
        for _ in range(nlayers):
            if len(layers) ==0:
                #inital layer has to have size of input features as its input layer
                #its output layer can have any size but it must match the size of the input layer of the next linear layer
                #here we choose its output layer as the hidden size (fully connected)
                layers.append(nn.Linear(nfeatures, hidden_size))
                #batch normalization
                # layers.append(nn.BatchNorm1d(hidden_size))
                #Dropout seems to worsen model performance
                layers.append(nn.Dropout(dropout))
                #ReLU activation 
                layers.append(nn.ReLU())
            else:
                #if this is not the first layer (we dont have layers)
                layers.append(nn.Linear(hidden_size, hidden_size))
                # layers.append(nn.BatchNorm1d(hidden_size))
                #Dropout seems to worsen model performance
                layers.append(nn.Dropout(dropout))
                layers.append(nn.ReLU())
                #output layer:
        layers.append(nn.Linear(hidden_size, ntargets)) 

        # ONLY IF ITS A CLASSIFICATION, ADD SIGMOID
        layers.append(nn.Sigmoid())
            #we have defined sequential model using the layers in oulist 
        self.model = nn.Sequential(*layers)

    
    def forward(self, x):
        return self.model(x)
    
class Engine:
    """loss, training and evaluation"""
    def __init__(self, model, optimizer, batch_size):
                 #, device):
        self.model = model
        #self.device= device
        self.optimizer = optimizer
        self.batch_size=batch_size
        
    #the loss function returns the loss function. It is a static method so it doesn't need self
    # @staticmethod
    # def loss_fun(targets, outputs):
    #   tau = torch.rand(outputs.shape)
    #   return torch.mean(torch.where(targets >= outputs, 
    #                                   tau * (targets - outputs), 
    #                                   (1 - tau)*(outputs - targets)))

#     This loss combines a Sigmoid layer and the BCELoss in one single class. This version is more numerically stable than using a plain Sigmoid followed by a BCELoss as, 
#     by combining the operations into one layer

    def train(self, x, t):
        """the training function: takes the training dataloader"""
        self.model.train()
        final_loss = 0
        for iteration in range(n_iterations):
            self.optimizer.zero_grad()
            batch_x, batch_t = get_features_training_batch(x, t,  self.batch_size)#x and t are train_x and train_t

            # with torch.no_grad():
            inputs=torch.from_numpy(batch_x).float()
            targets=torch.from_numpy(batch_t).float()
            outputs = self.model(inputs)
            loss = average_quadratic_loss(outputs, targets, inputs)
            loss.backward()
            self.optimizer.step()
            final_loss += loss.item()

        return final_loss / self.batch_size
    
    def evaluate(self, x, t):
        """the training function: takes the training dataloader"""
        self.model.eval()
        final_loss = 0
        for iteration in range(n_iterations):
            batch_x, batch_t = get_features_training_batch(x, t,  self.batch_size)#x and t are train_x and train_t

            # with torch.no_grad():            
            inputs=torch.from_numpy(batch_x).float()
            targets=torch.from_numpy(batch_t).float()
            outputs = self.model(inputs)
            loss =average_quadratic_loss(outputs, targets, inputs)
            final_loss += loss.item()
        return final_loss / self.batch_size



EPOCHS=1
def run_train(params, save_model=False):
    """For tuning the parameters"""

    model =  RegularizedRegressionModel(
              nfeatures=sample_x.shape[1], 
                ntargets=1,
                nlayers=params["nlayers"], 
                hidden_size=params["hidden_size"],
                dropout=params["dropout"]
                )
    # print(model)
    

    learning_rate= params["learning_rate"]
    optimizer_name = params["optimizer_name"]
    
    # optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"]) 
    
    optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), lr=learning_rate)
    
    eng=Engine(model, optimizer, batch_size=params["batch_size"])
    best_loss = np.inf
    early_stopping_iter=10
    early_stopping_coutner=0

    for epoch in range(EPOCHS):
        train_loss = eng.train(train_x, train_t)
        valid_loss=eng.evaluate(test_x, test_t)

        print(f"{epoch} \t {train_loss} \t {valid_loss}")
        if valid_loss<best_loss:
            best_loss=valid_loss
            if save_model:
                model.save(model.state_dict(), "model_m.bin")
        else:
            early_stopping_coutner+=1
        if early_stopping_coutner > early_stopping_iter:
            break
    return best_loss

# run_train()

def objective(trial):
    params = {
      "nlayers": trial.suggest_int("nlayers",1,13),      
      "hidden_size": trial.suggest_int("hidden_size", 2, 130),
      "dropout": trial.suggest_float("dropout", 0.1,0.5),
      "optimizer_name" : trial.suggest_categorical("optimizer_name", ["Adam", "RMSprop"]),
      "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-2),
      "batch_size": trial.suggest_int("batch_size", 1000, 10000)

    }
    # all_losses=[]

    temp_loss = run_train(params,save_model=False)
    # all_losses.append(temp_loss)
    return temp_loss

def tune_hyperparameters():
    print('Getting best hyperparameters')
    study=optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=10)
    best_trial = study.best_trial
    print('best model parameters', best_trial.params)

    best_params=best_trial.params#this is a dictionary
    filename=LFI_BASE+'best_params/best_params_Test_Trials.csv'
    param_df=pd.DataFrame({
                            'n_layers':best_params["nlayers"], 
                            'hidden_size':best_params["hidden_size"], 
                            'dropout':best_params["dropout"],
                            'optimizer_name':best_params["optimizer_name"],
                            'learning_rate': best_params["learning_rate"], 
                            'batch_size':best_params["batch_size"] },
                                    index=[0]
    )

    param_df.to_csv(filename)   
    
    
# BEST_PARAMS = pd.read_csv(LFI_BASE+'best_params/best_params_Test_Trials.csv')
# print(BEST_PARAMS)

# n_layers = int(BEST_PARAMS["n_layers"]) 
# hidden_size = int(BEST_PARAMS["hidden_size"])
# dropout = float(BEST_PARAMS["dropout"])
# optimizer_name = BEST_PARAMS["optimizer_name"].to_string().split()[1]
# learning_rate =  float(BEST_PARAMS["learning_rate"])
# batch_size = int(BEST_PARAMS["batch_size"])

def initiate_whose_model(Ali_or_Harrison, MLE):
    whose_model='Ali'

    if whose_model=='Harrison':
        n_layers=5
        hidden_size=5
        dropout=0
        learning_rate=int(1e-3)
        batch_size=64
        optimizer     = torch.optim.Adam(model.parameters(), lr=int(1e-3)) 
        model=Model()
    elif whose_model=='Ali':
        n_layers = int(BEST_PARAMS["n_layers"]) 
        hidden_size = int(BEST_PARAMS["hidden_size"])
        dropout = float(BEST_PARAMS["dropout"])
        optimizer_name = BEST_PARAMS["optimizer_name"].to_string().split()[1]
        learning_rate =  float(BEST_PARAMS["learning_rate"])
        batch_size = int(BEST_PARAMS["batch_size"])
        model =  RegularizedRegressionModel(
            nfeatures=sample_x.shape[1], 
            ntargets=1,
            nlayers=n_layers, 
            hidden_size=hidden_size, 
            dropout=dropout
            )
        optimizer = getattr(torch.optim, str(optimizer_name) )(model.parameters(), lr=learning_rate)
        
    return n_layers, hidden_size, dropout, optimizer_name, learning_rate, batch_size, model, optimizer


def make_eval_data(Bprime, train_df, nu, N, M, nbins):
    #if MLE true, load the model that was trained on MLE data and vice versa
    # N, M = D
    # nbins=NBINS
    # thetamin,thetamax=0,20
    thetamin=train_df['theta'].min()
    thetamax=train_df['theta'].max()
    thetastep = (thetamax-thetamin) / nbins
    bb    = np.arange(thetamin, thetamax+thetastep, thetastep)#this is just making a vector of thetas
    X     = (bb[1:] + bb[:-1])/2
    tensor = torch.Tensor([[x, nu, theta_hat(N, M, MLE=True), N, M] for x in X])
    return tensor, X.ravel()

def usemodel(Bprime, train_df, nu, N,M, MLE, nbins):
    
    #Generate evaluation data at those fixed nu, N, M values
    eval_data, eval_bins =make_eval_data(Bprime,train_df,nu, N,M, nbins)#eval data is indipendent of MLE, since its just constants witha theta variable

    # if MLE==True:
    #     model=model
    #else load the model trained on non-MLE data
    # PATH='models/MLE_TRUE_Regressor_200.0K_training_iter.pt'
    
    #LOAD TRAINED MODEL
    with_theta_hat=False
    if MLE:
        
        PATH= 'models/MLE_TRUE_Regressor_200.0K_training_iter.pt'
        PATH= 'models/MLE_True_Regressor_100.0K_training_iter_with_theta_hat.pt'
        
    else:
        PATH= 'models/MLE_False_Regressor_200.0K_training_iter.pt'
        PATH= 'models/MLE_False_Regressor_100.0K_training_iter_with_theta_hat.pt'
    n_layers = int(BEST_PARAMS["n_layers"]) 
    hidden_size = int(BEST_PARAMS["hidden_size"])
    dropout = float(BEST_PARAMS["dropout"])
    optimizer_name = BEST_PARAMS["optimizer_name"].to_string().split()[1]
    learning_rate =  float(BEST_PARAMS["learning_rate"])
    batch_size = int(BEST_PARAMS["batch_size"])
    model =  RegularizedRegressionModel(
        nfeatures=sample_x.shape[1], 
        ntargets=1,
        nlayers=n_layers, 
        hidden_size=hidden_size, 
        dropout=dropout
        )
    #EVALUATE AT AT EVAL_DATA
    model.load_state_dict(torch.load(PATH) )
    model.eval()
    return model(eval_data).detach().numpy(), eval_bins

# sample_x=train_df_MLE
def plot_data_one_nu_with_model(Bprime, thetamin, thetamax, nu, D, MLE, 
                     NBINS,
              FONTSIZE=15,
              func=None,
              fgsize=(10, 6), save_image=False):
    
    # make room for 6 sub-plots
    fig, ax = plt.subplots(nrows=2, 
                           ncols=3, 
                           figsize=fgsize)
    
    # padding
    plt.subplots_adjust(hspace=0.01)
    plt.subplots_adjust(wspace=0.20)
    
    # use flatten() to convert a numpy array of 
    # shape (nrows, ncols) to a 1-d array. 
    ax = ax.flatten()
    
    for j, (N, M) in enumerate(D):
        
        y, bb = make_hist_data(Bprime,
                              thetamin, thetamax,
                              nu, N, M,
                              nbins=NBINS,
                              MLE=True)
    
        ax[j].set_xlim(thetamin-0.5, thetamax-5)
        ax[j].set_ylim(0, 1.03)
        ax[j].set_xlabel(r'$\mathbf{\theta}$', fontsize=FONTSIZE-3)
        ax[j].set_ylabel(r'$\mathbf{E(Z|\theta, \nu)}$', fontsize=FONTSIZE-3)
        
        x = (bb[1:]+bb[:-1])/2
        ax[j].plot(x, y, 'b', lw=2, label='$\mathbf{h}$ MLE', alpha=0.4)
        #h is histogram approximation

        y_nonMLE, bb_nonMLE = make_hist_data(Bprime,
                              thetamin, thetamax,
                              nu, N, M,
                              nbins=NBINS,
                              MLE=False)
        
        
        x_nonMLE = (bb_nonMLE[1:]+bb_nonMLE[:-1])/2
        ax[j].plot(x_nonMLE, y_nonMLE, 'r', lw=2, label='$\mathbf{h}$ non-MLE',alpha=0.4)
        
        
        if func:
            train_df_MLE = load_train_df(MLE=True)
            train_df_nonMLE = load_train_df(MLE=False)
            
            f_MLE, f_bins_MLE = func(Bprime, train_df_MLE, nu, N, M, MLE=True, nbins=NBINS)
            ax[j].plot(f_bins_MLE, f_MLE, color='g', lw=2, label='$\mathbf{f}$ MLE', alpha=0.4)
            #f is model approximation
            
            f_nonMLE, f_bins_nonMLE = func(Bprime, train_df_nonMLE, nu, N, M, MLE=False, nbins=NBINS)
            ax[j].plot(f_bins_nonMLE, f_nonMLE, color='c', lw=2, label='$\mathbf{f}$ non-MLE', alpha=0.4)
            
        ax[j].grid(True, which="both", linestyle='-')
        ax[j].text(3.1, 0.42, r'$N, M = %d, %d$' % (N, M), fontsize=font_legend-3
                   # fontsize=FONTSIZE
                  ) 

        ax[j].text(3.1, 0.30, r'$\nu = %5.1f$' % nu, fontsize=font_legend-3
                   # fontsize=FONTSIZE
                  ) 

        ax[j].legend(loc='upper right',fontsize=font_legend-3)
        
    # hide unused sub-plots
    for k in range(j+1, len(ax)):
        ax[k].set_visible(False)
    
    plt.tight_layout()
    if save_image:
        plt.savefig('images/h_MLE_nonMLE_f_MLE_f_nonMLE_one_nu%s.png' % str(nu))
    plt.show()

def plot_data_many_nus_with_model(Bprime, thetamin, thetamax, nu_list, D,
                     NBINS,
              FONTSIZE=15,
              func=None,
              fgsize=(10, 6), save_image=False):
    
    # make room for 6 sub-plots
    fig, ax = plt.subplots(nrows=2, 
                           ncols=2, 
                           figsize=fgsize)
    
    outside=''
    ALPHA=0.8
    TITLE_SIZE=font_legend+1
    
    # padding
    plt.subplots_adjust(hspace=3)
    plt.subplots_adjust(wspace=1)#horizontal distance
    
    # use flatten() to convert a numpy array of 
    # shape (nrows, ncols) to a 1-d array. 
    
    for nu in nu_list:
        
        N, M = D
        y, bb = make_hist_data(Bprime,
                              thetamin, thetamax,
                              nu, N, M,
                              nbins=NBINS,
                              MLE=True)
    

        if nu > 20:
            outside = outside + r' ($>$ train data)'
        
        x = (bb[1:]+bb[:-1])/2
        ax[0,0].plot(x, y, lw=2, label=r'$\nu= %s$ %s' % (str(nu), outside), alpha=ALPHA)
        ax[0,0].set_title(r'$\mathbf{h}$ MLE', fontsize=TITLE_SIZE)
        #h is histogram approximation

        y_nonMLE, bb_nonMLE = make_hist_data(Bprime,
                              thetamin, thetamax,
                              nu, N, M,
                              nbins=NBINS,
                              MLE=False)
        
        
        x_nonMLE = (bb_nonMLE[1:]+bb_nonMLE[:-1])/2

        ax[1,0].plot(x_nonMLE, y_nonMLE, lw=2, label=r'$\nu= %s$ %s' % (str(nu), outside) ,alpha=ALPHA)
        ax[1,0].set_title(r'$\mathbf{h}$ non-MLE',fontsize=TITLE_SIZE)
        
        if func:
            #load the correct dataframe
            train_df_MLE = load_train_df(MLE=True)
            train_df_nonMLE = load_train_df(MLE=False)
            
            f_MLE, f_bins_MLE = func(Bprime, train_df_MLE, nu, N, M, MLE=True, nbins=NBINS)
            ax[0,1].plot(x, f_MLE, lw=2, label=r'$\nu= %s$ %s' % (str(nu), outside), alpha=ALPHA)
            ax[0,1].set_title(r'$\mathbf{f}$ MLE',fontsize=TITLE_SIZE)
            #f is model approximation
            
            f_nonMLE, f_bins_nonMLE = func(Bprime, train_df_nonMLE, nu, N, M, MLE=False, nbins=NBINS)
            ax[1,1].plot(f_bins_nonMLE, f_nonMLE, lw=2, label=r'$\nu= %s$ %s' % (str(nu), outside), alpha=ALPHA)
            ax[1,1].set_title(r'$\mathbf{f}$ non-MLE',fontsize=TITLE_SIZE)
        
        
        for i in range(ax.shape[0]):
            for j in range(ax.shape[1]):
                ax[i,j].set_xlim(thetamin-0.5, thetamax-7)
                ax[i,j].set_ylim(0.2, 1.03)
                ax[i,j].set_xlabel(r'$\mathbf{\theta}$', fontsize=FONTSIZE-2)
                ax[i,j].set_ylabel(r'$\mathbf{E(Z|\theta, \nu)}$', fontsize=FONTSIZE-2)
                ax[i,j].text(2, 0.5, r'$N, M = %d, %d$' % (N, M), fontsize=font_legend-3
                           # fontsize=FONTSIZE
                          ) 
                
                ax[i,j].grid(True, which="both", linestyle='-')

                ax[i,j].legend(loc='center right',fontsize=font_legend-3)
                ax[i,j].patch.set_edgecolor('black')  

                ax[i,j].patch.set_linewidth('1')  
        # ax[j].text(3.1, 0.30, r'$\nu = %5.1f$' % nu, fontsize=font_legend-3
        #            # fontsize=FONTSIZE
        #           ) 

        
        
    # hide unused sub-plots
#     for k in range(j+1, len(ax)):
#         ax[k].set_visible(False)
    
    plt.tight_layout()
    if save_image:
        plt.savefig('images/h_MLE_nonMLE_f_MLE_f_nonMLE_many_nus.png')
    plt.show()
  

def get_one_batch(x,  batch_size):
    rows    = np.random.choice(len(x), batch_size)
    batch_x = x[rows]

    # batch_x.T[-1] = np.random.uniform(0, 1, batch_size)
    return batch_x
def getwholedata_f_and_lambda(MLE_or_nonMLE, valid=False):
    feature_cols = ['theta', 'nu', 'theta_hat', 'N', 'M']
    if MLE:
        data = pd.read_csv('data/two_parameters_theta_0_20_1000k_Examples_MLE_True.csv', 
                     # nrows=SUBSAMPLE,
                     usecols=feature_cols
                    )
        
    else:
        data = pd.read_csv('data/two_parameters_theta_0_20_1000k_Examples_MLE_False.csv', 
             # nrows=SUBSAMPLE,
             usecols=feature_cols
            )
        
    train_data, test_data = train_test_split(data, test_size=0.2)
    #split the train data (0.8 of whole set) again into 0.8*0.8=0.64 of whole set

    train_data = train_data.reset_index(drop=True)
    test_data  = test_data.reset_index(drop=True)

    target='Z'; source = ['theta','nu','theta_hat','N','M']# these are not needed here
    train_x = train_data.to_numpy()
    test_x = test_data.to_numpy()
    #dont return the torch tensors! we want to do operations on them while training
        
    return train_x, test_x

def get_one_batch(x,  batch_size):
    rows    = np.random.choice(len(x), batch_size)
    batch_x = x[rows]
    # batch_x.T[-1] = np.random.uniform(0, 1, batch_size)
    return batch_x

def RMS(v):
    return (torch.mean(v**2))**0.5
    
def average_quadratic_loss_f_pivot(f, t, df_dnu):
    kappa=1.5
    #here t will be Z_tilde, f is model_f(x)
    return  torch.mean((f - t)**2) - kappa/2 * RMS(df_dnu)


def average_quadratic_loss_tildelambda_pivot(lambdatilde, t, dlmbdatilde_dnu):
    psi=2
    #here t will be lambda_D(N,M), lambda_tilde is model_lambda(x)
    return  torch.mean((lambdatilde - t)**2) - psi/2 * RMS(dlambdatilde_dnu)


def validate_f(model, avloss, inputs, targets, df_dnu):
    # make sure we set evaluation mode so that any training specific
    # operations are disabled.
    model.eval() # evaluation mode
    with torch.no_grad(): # no need to compute gradients wrt. x and t
        x = torch.from_numpy(inputs).float()
        t = torch.from_numpy(targets).float()
        # remember to reshape!
        o = model(x).reshape(t.shape)
        #avloss has signature average_quadratic_loss_f_pivot(f, t, df_dnu)
    return avloss(o, t, df_dnu)

def validate_lambda_tilde(model, avloss, inputs, targets, dlambdatilde_dnu):
    # make sure we set evaluation mode so that any training specific
    # operations are disabled.
    model.eval() # evaluation mode
    with torch.no_grad(): # no need to compute gradients wrt. x and t
        x = torch.from_numpy(inputs).float()
        t = torch.from_numpy(targets).float()
        # remember to reshape!
        o = model(x).reshape(t.shape)#this is lambda tilde
        #avloss has signaure average_quadratic_loss_tildelambda_pivot(lambdatilde, t, dlmbdatilde_dnu)
    return avloss(o, t, dlambdatilde_dnu)

def train_pivotal_model_and_lambda(model_f, model_lambda, 
                                   optimizer_f, optimizer_lambda, 
                                   avloss_f, avloss_lambda,
                                    batch_size, n_iterations, 
                                   traces_f, traces_lambda, 
                                      step, window, MLE):
    
    # to keep track of average losses
    xx_F, yy_t_F, yy_v_F, yy_v_avg_F = traces_f
    xx_lambda, yy_t_lambda, yy_v_lambda, yy_v_avg_lambda = traces_lambda
    
    
    if MLE:
        train_x, test_x = getwholedata_f_and_lambda(MLE_or_nonMLE=True, valid=False)
    else:
        train_x, test_x = getwholedata_f_and_lambda(MLE_or_nonMLE=False, valid=False)
    
    #Remember train_x will have columns ['theta','nu','theta_hat','N','M']
    n = len(test_x)
    print('Iteration vs average loss')
    print("%10s\t%10s\t%10s" %           ('iteration', 'train-set', 'valid-set'))    
    for ii in range(n_iterations):
        # model.eval()
        #Harrison-like Loader
        batch_x_train = get_one_batch(train_x,  batch_size)
        
        #Or Ali's Loader
        # batch_x, batch_t = next(training_set_features()), next(training_set_targets())
        # batch_x_eval, batch_t_eval = next(evaluation_set_features()), next(evaluation_set_targets())
        # x = torch.from_numpy(batch_x).float()
        x = batch_x_train
        # print('x is leaf: ', x.is_leaf)
        # x.retain_grad()
        # print('x is leaf after retain: ', x.is_leaf)
        # x.requires_grad_(True)
        # x.retain_grad()
        theta = x[:,0]
        nu = x[:,1]
        
        n = st.poisson.rvs(theta+ nu, size=batch_size)
        m = st.poisson.rvs(nu, size=batch_size)
        
        lambda_gen = lambda_test(theta, n, m, MLE)
        
        N = x[:,3]
        M = x[:,4]
        
        lambda_D = lambda_test(theta, N, M, MLE)
        Z_tilde = (lambda_gen < lambda_D).astype(np.int32)
        
        
        
        ######## take grade of model_f wrt nu
        x = torch.tensor(x).float()
        x.requires_grad_(True)
        f = model_f(x)
        f = f.view(-1)
        #multiply the model by its ransverse, remember we can only take gradients of scalars
        #and f will be a vector before this
        f = f @ f.t()
        # f = torch.tensor(f, requires_grad=True)
        # print('f shape: ', f.shape)
        # print('f is leaf: ', f.is_leaf)
        
        ###################### Get lambda tilde
        lambda_tilde = model_lambda(x) * torch.tensor(lambda_D).float()
        # take amplitude of lambda_tilde
        
        lambda_tilde = lambda_tilde.view(-1)
        lambda_tilde = lambda_tilde @ lambda_tilde.t()
        # f_2 = f**2
        # print('f2 shape', f_2.shape)
        # nu = torch.autograd.Variable( x[:,1], requires_grad=True)
        
        # nu=torch.autograd.Variable(x[:,1], requires_grad=True)
        # nu=torch.tensor(x[:,1], requires_grad=True)
        # print(type(nu))
        # nu.retain_grad()
        # print('nu shape: ', nu.shape)
        # print('nu is leaf: ', nu.is_leaf)
        # print('nu type', type(nu))
        
        
        # WE NEED TO RETAIN_GRAD ON NON-LEAF NODES 
        f.retain_grad()
        f.backward(gradient=torch.ones_like(f), retain_graph=True)
        df_dx = x.grad
        # print('df_dnu =', df_dnu)
        # print('df_dx =', df_dx)
        # print('df_dx shape :', df_dx.shape)
        df_dnu = df_dx[:,1]
        # x.grad.zero_()
        # print('df_dnu shape: ', df_dnu.shape)
        #################### Lambda_tilde gradient ##############################
        x.requires_grad_(True)
        lambda_tilde.retain_grad()
        lambda_tilde.backward(gradient=torch.ones_like(lambda_tilde), retain_graph=True)
        dlambda_tilde_dx = x.grad
        dlambda_tilde_dnu = dlambda_tilde_dx[:,1]
        
        #clear the gradient after you take it
        x.grad.zero_()
        # break        
        # with torch.no_grad():
        #     x = torch.from_numpy(batch_x).float()
        #     t = torch.from_numpy(batch_t).float()   
        
        ################################################################################
        lambda_D = torch.tensor(lambda_D).float()
        #lambda_D will be the target for model_lambda
        Z_tilde = torch.tensor(Z_tilde).float()
        #Z_tilde will be the target for model_fget for model_f
        
        #target for f
        t_f = Z_tilde
        t_lambda_tilde = lambda_D
        
        model_f.train()
        outputs_f = model_f(x).reshape(t_f.shape)
        # compute a noisy approximation to the average loss
        empirical_risk_f = avloss_f(outputs_f, t_f, df_dnu)
        
        # use automatic differentiation to compute a 
        # noisy approximation of the local gradient
        optimizer_f.zero_grad()       # clear previous gradients
        empirical_risk_f.backward()   # compute gradients
        # finally, advance one step in the direction of steepest 
        # descent, using the noisy local gradient. 
        optimizer_f.step()            # move one step
        
        
        model_lambda.train()
        outputs_lambda_tilde = model_lambda(x).reshape(t_lambda_tilde.shape)
        # compute a noisy approximation to the average loss
        empirical_risk_lambda_tilde = avloss_f(outputs_lambda_tilde, t_lambda_tilde, dlambda_tilde_dnu)
        # use automatic differentiation to compute a 
        # noisy approximation of the local gradient
        optimizer_lambda.zero_grad()       # clear previous gradients
        empirical_risk_lambda_tilde.backward()   # compute gradients
        # finally, advance one step in the direction of steepest 
        # descent, using the noisy local gradient. 
        optimizer_lambda.step()            # move one step
        
        
#         if ii % step == 0:
            
#             # this is an example of an x tensor
#             # [17.3352, 10.7722,  6.0000,  8.0000],
#             #[16.7822, 13.3260,  8.0000,  4.0000],
#             #using Harrison-like loader
#             batch_x_test = get_one_batch(test_x,  batch_size)
#             #validate_f has signature validate_f(model, avloss, inputs, targets, df_dnu)
#             acc_t_f = validate_f(model_f, avloss_f, train_x[:n], train_t[:n], df_dnu)
#             acc_v = validate(model, avloss, test_x[:n], test_t[:n], df_dnu)
            #using Ali's loader
            # acc_t = validate(model, avloss, batch_x, batch_t) 
            # acc_v = validate(model, avloss, batch_x_eval, batch_t_eval)
            
#             yy_t_F.append(acc_t)
#             yy_v_F.append(acc_v)
            
#             # compute running average for validation data
#             len_yy_v_F = len(yy_v_F)
#             if   len_yy_v_F < window:
#                 yy_v_avg_F.append( yy_v_F[-1] )
#             elif len_yy_v_F == window:
#                 yy_v_avg_F.append( sum(yy_v_F) / window )
#             else:
#                 acc_v_avg  = yy_v_avg_F[-1] * window
#                 acc_v_avg += yy_v_F[-1] - yy_v_F[-window-1]
#                 yy_v_avg_F.append(acc_v_avg / window)
                        
#             if len(xx_F) < 1:
#                 xx_F.append(0)
#                 print("%10d\t%10.6f\t%10.6f" % \
#                       (xx_F[-1], yy_t_F[-1], yy_v_F[-1]))
#             else:
#                 xx_F.append(xx_F[-1] + step)
                    
#                 print("\r%10d\t%10.6f\t%10.6f\t%10.6f" % \
#                           (xx_F[-1], yy_t_F[-1], yy_v_F[-1], yy_v_avg_F[-1]), 
#                       end='')
            
    print()      
    return (xx_F, yy_t_F, yy_v_F, yy_v_avg_F), (xx_lambda, yy_t_lambda, yy_v_lambda, yy_v_avg_lambda)

def make_D(train_df):
    Nmin = train_df['N'].min()
    Nmax = train_df['N'].max()
    Mmin = train_df['M'].min()
    Mmax = train_df['M'].max()
    D = [ (N, M) for N in range(Nmin, Nmax) for M in range(Mmin, Mmax)]
    return np.array(D)[[0, 10, 15, 20, 40]]


def save_model(MLE):
    if MLE:
        model = model_MLE
    else:
        model = model_nonMLE
    PATH='models/MLE_%s_Regressor_%sK_training_iter.pt' % ( str(MLE), str(n_iterations/1000) )
    with_theta_hat=True
    if with_theta_hat:
        PATH='models/MLE_%s_Regressor_%sK_training_iter_with_theta_hat.pt' % ( str(MLE), str(n_iterations/1000) )
    
    torch.save(model.state_dict(),  PATH)
    


def load_train_df(MLE):
    """ returns the dataframe, can be used if the dataframe is saved in csv format
    of if it is already in dataframe format (e.g. generated in this notebook). """
    # SUBSAMPLE=int(1e5)
    # if isinstance(df_name,str):
    if MLE:
        train_df = pd.read_csv('data/two_parameters_theta_0_20_1000k_Examples_MLE_True.csv', 
                         # nrows=SUBSAMPLE,
                         usecols=['Z','theta', 'nu', 'theta_hat', 'N', 'M']
                        )
    else:
        train_df = pd.read_csv('data/two_parameters_theta_0_20_1000k_Examples_MLE_False.csv', 
                 # nrows=SUBSAMPLE,
                 usecols=['Z','theta', 'nu', 'theta_hat', 'N', 'M']
                )
    return train_df


def mkdir(dir_):
    """make a directory without overwriting what's in it if it exists"""
    # assert isinstance(dir_, str)
    try:
        os.system('mkdir -p %s' % str(dir_) )
    except Exception:
        pass
    
    
def import_base_stack():
    import numpy as np; import pandas as pd
    import scipy as sp; from numba import njit
    
def binsize(data_df_or_Bprime):
    """ decide on the number of bins in a histogram according to Struge's rule"""
    if isinstance(data_df_or_Bprime,pd.core.frame.DataFrame):
        bin_count = int(np.ceil(np.log2(data_df_or_Bprime.shape[0])) + 1)
    elif isinstance(data_df_or_Bprime, int):
        bin_count = int(np.ceil(np.log2(data_df_or_Bprime)) + 1)
    return bin_count
