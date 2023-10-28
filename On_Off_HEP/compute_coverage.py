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
from joblib import  Memory

import copy; import sys; import os
from IPython.display import Image, display
from importlib import import_module
from model import RegularizedRegressionModel



try:
    import optuna
except Exception:
    print('optuna is only used for hyperparameter tuning, not critical!')
    pass
# import sympy as sy
#sometimes jupyter doesnt initialize MathJax automatically for latex, so do this

try:
    LFI_PIVOT_BASE = os.environ['LFI_PIVOT_BASE']
    print('BASE directoy properly set = ', LFI_PIVOT_BASE)
    utils_dir = os.path.join(LFI_PIVOT_BASE, 'utils')
    sys.path.append(utils_dir)
    import utils
    #usually its not recommended to import everything from a module, but we know
    #whats in it so its fine
    from utils import *
except Exception:
    print("""BASE directory not properly set. Read repo README.\
    If you need a function from utils, use the decorator below, or add utils to sys.path""")
    pass


#Harrison fonts
FONTSIZE=18
font = {'family': 'serif', 'weight':'normal', 'size':FONTSIZE}
mp.rc('font', **font)
mp.rc('text',usetex=True)
DATA_DIR = os.path.join(LFI_PIVOT_BASE,'data')
memory = Memory(DATA_DIR)

def debug(func):
    """Print the function signature and return value"""
    import functools

    @functools.wraps(func)
    def wrapper_debug(*args, **kwargs):
        args_repr = [repr(a) for a in args]
        kwargs_repr = [f"{k}={v!r}" for k, v in kwargs.items()]
        signature = ", ".join(args_repr + kwargs_repr)
        print(f"Calling {func.__name__}({signature})")
        values = func(*args, **kwargs)
        print(f"{func.__name__!r} returned {values!r}")
        return values

    return wrapper_debug

def theta_hat_func(n,m, MLE):
       #n,m are integer arrays
    if MLE==True:
        theta_hat = n-m
    else:
        # non-MLE
        # theta_hat = n-m
        # theta_hat = (theta_hat) * (theta_hat > 0)
        theta_hat = np.where(n>m, n-m, 0)
         
    return theta_hat

def L_prof_global(n,m, MLE):
    #n,m integer arrays
    # nu_hat = m, if theta_hat = theta_hat_MLE
    # nu_hat  =  (m+n)/2 if theta_hat = n-m
    # nu_hat = 0  if theta_hat != n-m
    theta_hat=theta_hat_func(n,m,MLE)
    # print('n-m ',  n-m)
    if MLE==True:
        # i.e. if theta_hat = n-m
        assert theta_hat==n-m
        nu_hat = m
    else:
        nu_hat =(m+n)/2
        # if theta_hat== n-m:
        #     nu_hat = (m+n)/2
        # else:
        #     nu_hat = 0
        # nu_hat = np.where(theta_hat==n-m,
        #                   (m+n)/2, 
        #                   0)
            
        
    p1=st.poisson.pmf(n, theta_hat+nu_hat)
    p2 = st.poisson.pmf(m, nu_hat)
    return p1*p2

def L_theta_nu(n,m,theta,nu):
    p1 = st.poisson.pmf(n, theta+nu)
    p2 = st.poisson.pmf(m, nu)
    return p1*p2
def lambda_test_2d(n,m, theta, nu, MLE):
    Ln= L_theta_nu(n,m,theta,nu)
    
    Ld= L_prof_global(n,m, MLE)
    eps=1e-20
    Ld=Ld+eps
    lambda_  = -2*np.log(Ln/Ld)
    return np.array(lambda_)

chi2_exp_size=int(1e5)

def run_sim_2d(theta, nu, MLE, lambda_size):
    """Sample n ~ Pois(theta+nu), 
              m ~ Pois(nu), 
    and compute 
              lambda(theta, n, m)
              
    return: (n, m, lambda_), where each are np arrays of length lambda_size
    """
    n = st.poisson.rvs(theta+nu, size=lambda_size)
    m = st.poisson.rvs(nu, size=lambda_size)
    lambda_ = lambda_test_2d(n, m, theta, nu, MLE)
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
        n, m, lambda_ = run_sim_2d(theta, nu, MLE, lambda_size =chi2_exp_size)
        df['n'] = n
        df['m'] = m
        df['lambda']=lambda_
        lambda_results.append((n, m, lambda_, theta, nu))
    
        print( '\n \n (theta, nu) =  (%.f, %.f) \n ' % (theta, nu) )
        print(f'\t \t with associated n =  {n}, \n \n \t \t m = {m}, \n \n \t \t lambda = {lambda_}'  )
    return df, lambda_results

def make_hist_2d_data_2d_inference(Bprime,
              thetamin, thetamax,
              numin, numax, 
                      N, M,
               nbinstheta, nbinsnu,                    
                      MLE):

    theta = st.uniform.rvs(thetamin, thetamax, size=Bprime)
    nu = st.uniform.rvs(numin, numax, size=Bprime)
    n = st.poisson.rvs(theta + nu, size=Bprime)
    m = st.poisson.rvs(nu, size=Bprime)
    #lambda_test_2d(n,m, theta, nu)
    Z = (lambda_test_2d(n,m, theta, nu, MLE)< 
         lambda_test_2d(N,M, theta, nu, MLE)).astype(np.int32)

    thetarange = (thetamin, thetamax)
    nurange = (numin, numax)
    # bins = binsize(Bprime)

    # Z-weighted histogram   (count the number of ones per bin)
    #theta will be on axis and nu on y axis
    y_theta_nu_w, bb_theta_edges, bb_nu_edges = np.histogram2d(theta, nu,
                          bins=(nbinstheta, nbinsnu), 
                          range=(thetarange, nurange), 
                          weights=Z)
    
    # unweighted histogram (count number of ones and zeros per bin)
    y_theta_nu_uw, bb_theta_edges, bb_nu_edges = np.histogram2d(theta, nu,
                          bins=(nbinstheta, nbinsnu), 
                          range=(thetarange, nurange))
    eps=1e-15
    P_theta_nu =  y_theta_nu_w / (y_theta_nu_uw + eps)    
    #P_theta_nu approximates E[Z]
    return P_theta_nu, bb_theta_edges, bb_nu_edges

thetaMin, thetaMax =  0, 20
numin, numax = 0, 20
Nmin, Nmax =  1,10
Mmin, Mmax =  1 , 10

def generate_training_data_2d(Bprime, MLE, save_data=False):
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
    lambda_gen = lambda_test_2d(n, m, theta, nu, MLE)
    print('lambda_gen= ', lambda_gen[:SUBSAMPLE])
    lambda_D = lambda_test_2d(N, M, theta, nu, MLE)
    
    print('lambda_D= ', lambda_D[:SUBSAMPLE])
    #if lambda_gen <= lambda_D: Z=1, else Z=0
    Z = (lambda_gen < lambda_D).astype(np.int32)
    
    data_2_param = {'Z' : Z, 'theta' : theta, 'nu': nu, 'theta_hat': theta_hat_, 'N':N, 'M':M}

    data_2_param = pd.DataFrame.from_dict(data_2_param)
    PATH = os.path.join(LFI_PIVOT_BASE, 
                        'data',
                        'TWO_PARAMETERS_theta_%s_%s_%sk_Examples_MLE_%s.csv' % (str(thetaMin), str(thetaMax), str(int(Bprime/1000)), str(MLE)) )
    if save_data:
        data_2_param.to_csv(PATH)

    print('\n')
    print(data_2_param.describe())
    return data_2_param

def split_t_x(df, target, source):
    # change from pandas dataframe format to a numpy 
    # array of the specified types
    t = np.array(df[target])
    x = np.array(df[source])
    return t, x

def get_batch(x, batch_size):
    # the numpy function choice(length, number)
    # selects at random "batch_size" integers from 
    # the range [0, length-1] corresponding to the
    # row indices.
    rows    = np.random.choice(len(x), batch_size)
    batch_x = x[rows]
    # batch_x.T[-1] = np.random.uniform(0, 1, batch_size)
    return batch_x


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

@memory.cache
def load_2d_train_df(MLE, with_lambda_D, small_df=True): 
    """ returns the dataframe, can be used if the dataframe is saved in csv format
    of if it is already in dataframe format (e.g. generated in this notebook). 
    small_df: return a dataframe a fraction of the size"""
    # SUBSAMPLE=int(1e5)
    # if isinstance(df_name,str):
    if with_lambda_D==True:
        # USECOLS=['Z','theta', 'nu', 'lambda_D']
        USECOLS=['Z','theta', 'nu', 'lambda_D', 'N', 'M']
        if MLE==True:
            DF_NAME= 'TWO_PARAMETERS_WITH_LAMBDA_D_theta_0_20_1000k_Examples_MLE_True.csv'
        else:
            DF_NAME='TWO_PARAMETERS_WITH_LAMBDA_D_theta_0_20_30000k_Examples_MLE_False.csv'
            
    else:
        USECOLS=['theta', 'nu', 'N', 'M']
        DF_NAME='TWO_PARAMETERS_WITH_LAMBDA_D_theta_0_20_30000k_Examples_MLE_False.csv'
        print('are you sure you want to use NM dataframe?')
    data_path=os.path.join(LFI_PIVOT_BASE, 
                    'data', DF_NAME)
        
    train_df = pd.read_csv(data_path, 
                    # nrows=SUBSAMPLE,
                    usecols=USECOLS
                )
    if small_df==True:
        sdf=train_df.iloc[:train_df.shape[0]//2]
        train_df=sdf
                              
    print(f'loading dataframe with name {DF_NAME}')
    print(train_df.describe())
    return train_df

@memory.cache
def getwholedata_2d(MLE, valid, with_lambda_D):
    """ Get train test split arrays"""
    
    data = load_2d_train_df(MLE=MLE, with_lambda_D=with_lambda_D)
        
    train_data, test_data = train_test_split(data, test_size=0.2)
    #split the train data (0.8 of whole set) again into 0.8*0.8=0.64 of whole set
    

    train_data = train_data.reset_index(drop=True)
    test_data  = test_data.reset_index(drop=True)

    target='Z'
    # source = ['theta','nu','theta_hat','N','M']
    USECOLS=list(data.columns)
    USECOLS.pop(0)
    source = USECOLS

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
          step, window, MLE, with_lambda_D):
    
    # to keep track of average losses
    xx, yy_t, yy_v, yy_v_avg = traces
    

    
    if MLE==True:
        train_t, train_x, test_t,  test_x = getwholedata_2d(MLE=True, valid=False, with_lambda_D=with_lambda_D)
    else:
        train_t, train_x, test_t,  test_x = getwholedata_2d(MLE=False, valid=False, with_lambda_D=with_lambda_D)
        
    n = len(test_x)
    print('Iteration vs average loss')
    print("%10s\t%10s\t%10s" % \
          ('iteration', 'train-set', 'valid-set'))
    
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
                print("%10d\t%10.6f\t%10.6f" % \
                      (xx[-1], yy_t[-1], yy_v[-1]))
            else:
                xx.append(xx[-1] + step)
                    
                print("\r%10d\t%10.6f\t%10.6f\t%10.6f" % \
                          (xx[-1], yy_t[-1], yy_v[-1], yy_v_avg[-1]), 
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
        plt.savefig('images/loss_curves/IQN_'+N+T+'_Consecutive_2.png')
        print('\nloss curve saved in images/loss_curves/IQN_'+N+target+'_Consecutive.png')
    # if show_loss_plots:
    plt.show()
    

########################333


def fig_ax():
    
    figwidth_by_height_ratio=1.33
    height=6
    width=figwidth_by_height_ratio*height
    width=6
    fig,ax = plt.subplots(1,1,
                          figsize=(9,4),
                         )
    return fig, ax

@debug
def load_untrained_model(PARAMS):
    """Load an untrained model (with weights initiatted) according to model paramateters in the 
    PARAMS dictionary

    Args:
        PARAMS (dict): dictionary of model/training parameters: i.e. hyperparameters and training parameters.

    Returns:
        utils.RegularizedRegressionModel object
    """
    model = RegularizedRegressionModel(
        nfeatures=PARAMS['NFEATURES'],
        ntargets=1,
        nlayers=PARAMS["n_layers"],
        hidden_size=PARAMS["hidden_size"],
        dropout=PARAMS["dropout"],
        activation=PARAMS["activation"],
    )
    # model.apply(initialize_weights)
    print('INITIATED UNTRAINED MODEL:', model)
    print(model)
    return model
    
@debug
def load_trained_model(PATH, PARAMS):
    model = RegularizedRegressionModel(
        nfeatures=PARAMS['NFEATURES'],
        ntargets=1,
        nlayers=PARAMS["n_layers"],
        hidden_size=PARAMS["hidden_size"],
        dropout=PARAMS["dropout"],
        activation=PARAMS["activation"],
    )
    model.load_state_dict(torch.load(PATH))
    print(model)
    model.train()
    return model


def save_model(model, PARAMS, pth_string):
    """pth string is the name of the pth file which is a dictionary of dictionaries"""
    models_path = os.path.join(LFI_PIVOT_BASE, 'models')
    PATH=os.path.join(models_path, pth_string)
    print(f'saving model with th string : {pth_string}\n')
    torch.save({'PARAMS': PARAMS,
                'model_state_dict': model.state_dict()},
                PATH)
    print(model)
    

def load_model(model, PARAMS, pth_string):
    models_path = os.path.join(LFI_PIVOT_BASE, 'models')
    PATH=os.path.join(models_path, pth_string)
    model = RegularizedRegressionModel(
        nfeatures=PARAMS['NFEATURES'],
        ntargets=1,
        nlayers=PARAMS["n_layers"],
        hidden_size=PARAMS["hidden_size"],
        dropout=PARAMS["dropout"],
        activation=PARAMS["activation"],
    )
    checkpoint = torch.load(PATH)
    print('INITIATED MODEL:',  model)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f'loading model with th string : {pth_string}\n')    
    print(model)
    
    return model
    
def evaluate_2d_model_on_df( with_lambda_D, PARAMS):
    """ returns the dataframe, can be used if the dataframe is saved in csv format
    of if it is already in dataframe format (e.g. generated in this notebook). """
    # SUBSAMPLE=int(1e5)
    #LOAD DATAFRAME
    MLE=PARAMS['MLE']
    whole_df = load_2d_train_df(MLE, with_lambda_D)
    # eval_df = eval_df
    #Convert DF to tensor

    if with_lambda_D==True:
        #Train on {theta, nu, lambda_D} as features and Z as target
        EVAL_COLUMNS=['Z','theta', 'nu', 'lambda_D']
        NFEATURES=3
    else:
        EVAL_COLUMNS=['Z','theta', 'nu', 'theta_hat', 'N', 'M']
        NFEATURES=5
    
    #MAKE EVAL TENSOR
    #remove Z
    EVAL_COLUMNS.pop(0)
    print(f'coumns used: {EVAL_COLUMNS}')
    eval_df=whole_df[EVAL_COLUMNS]
    eval_tensor= torch.from_numpy(eval_df.to_numpy())
    eval_tensor=eval_tensor.float()
    print(eval_tensor[:5])
    
    #LOAD MODEL
    model= load_untrained_model(PARAMS)
    trained_model= load_model(model=model, 
                                    PARAMS=PARAMS, 
                                    pth_string=PARAMS['pth_string']
                                    )
    trained_model.eval()

    print(trained_model)
    cdfhat = trained_model(eval_tensor).view(-1,).detach().numpy()
    # eval_df['cdfhat']=cdfhat
    # cdf_hat = 1-cdfhat
    whole_df['cdfhat'] = cdfhat
    #now the df has [theta, nu, cdfhat] as columns
    #combine this with the original df
    # original_df = load_2d_train_df(MLE, with_lambda_D=False)
    # merged_df = pd.merge(original_df, eval_df, on=['theta', 'nu', 'lambda_D'])
    
    #the whole dataframe is returned, with an extra column for prediction
    print('\nWHOLE DF WITH PREDICTION\n' , whole_df.head())
    return whole_df
    # return merged_df


def get_conf_set_at_given_tau(with_lambda_D, PARAMS, tau, NM_Pair=(3,7),  Generate_eval_data=False):

    N, M = NM_Pair 
    df = evaluate_2d_model_on_df(with_lambda_D=with_lambda_D, PARAMS=PARAMS)
     
    # Grenoble Data: N=3, M=7
    if N is not None and M is not None:
        print(f'\nSelecting for N = {N}, M={M}')
        # df= df[(df['N']==N & df['M']==M)]
        df= df[df['N']==N ]
        df= df[df['M']==M ]
    # print('THE COLUMNS HERE ARE', df.head())
    tau=float(tau)
    # cdf = 1-tau
    cdfhat_df_tau = df[df['cdfhat'] <= tau]
    # cdfhat_df_tau = df[df['cdfhat'] <= tau]
    print(f'The dataframe below corresponds to the {tau} confidence set')
    print(cdfhat_df_tau.head())
    return cdfhat_df_tau
    
def calc_coverage_at_given_tau(MLE, with_lambda_D, tau, Npoints):
    tau=float(tau)
    cdfhat_tau_df = get_conf_set_at_given_tau(MLE=True, with_lambda_D=True, tau=tau)
    # cdfhat_tau_df = cdfhat_tau_df[:801]
    theta_edge = cdfhat_tau_df['theta']
    nu_edge = cdfhat_tau_df['nu']
    cdfhat_at_tau = cdfhat_tau_df['cdfhat']
    model = load_model(MLE=MLE, with_lambda_D=with_lambda_D)
    for ind, (theta, nu, cdfhat) in enumerate(zip(theta_edge, nu_edge, cdfhat_at_tau)):
            N = st.poisson.rvs(theta+nu, size=Npoints)
            M = st.poisson.rvs(nu, size=Npoints)
            lambda_D = lambda_test_2d(N, M, theta, nu, MLE).flatten()
            # print(lambda_D)
            theta_arr = np.full(lambda_D.shape, theta)
            nu_arr = np.full(lambda_D.shape, nu)
            eval_arr = np.empty((Npoints, 3), dtype=float)
            eval_arr[:,0]=theta_arr
            eval_arr[:,1]=nu_arr
            eval_arr[:,2] = lambda_D
            # print(eval_arr.shape)
            # print(theta_arr.shape, nu_arr.shape, lambda_D.shape)
            eval_tensor= torch.from_numpy(eval_arr).float()
            # print(eval_tensor)
            cdfhat_at_point=  model(eval_tensor).view(-1,).detach().numpy()
            # cdfhat_at_point=1-cdfhat_at_point
            coverage_prob = np.mean(cdfhat_at_point <=  tau)
            print(coverage_prob)
        

def plot_coverage_levels(PARAMS, with_lambda_D, Npoints, saveplot=False):
    FONTSIZE = 16
    font = {'family' : 'serif',
            'weight' : 'normal',
            'size'   : FONTSIZE}
    mp.rc('font', **font)
    fig,ax = plt.subplots(1,2,
                        figsize=(9,4),
                        )
    plt.subplots_adjust(hspace=0.01)
    plt.subplots_adjust(wspace=0.3)
    xmin, xmax = 0., 501
    ymin, ymax  = 0.5,1
    tau_levels = [0.68, 0.8, 0.9, 0.95]
    MLE=PARAMS['MLE']
        #LOAD MODEL
    model= load_untrained_model(PARAMS)
    trained_model= load_model(model=model, 
                                    PARAMS=PARAMS, 
                                    pth_string=PARAMS['pth_string']
                                    )
    trained_model.eval()
    
    cdfhat_95_df = get_conf_set_at_given_tau(with_lambda_D=with_lambda_D, PARAMS=PARAMS, tau=0.95)
    cdfhat_95_df.sample(frac=1)
    cdfhat_95_df=cdfhat_95_df[:501]
    theta_edge_95 = cdfhat_95_df['theta']
    nu_edge_95 = cdfhat_95_df['nu']
    ax[0].set_xlim(0, 7)#theta
    ax[0].set_ylim(0,10)#nu
    ax[0].scatter(theta_edge_95, nu_edge_95, s=1, c='black', label='95\% CL set')

    ax[0].set_xlabel(r'$\mu$', fontsize=18)
    ax[0].set_ylabel(r'$\nu$', fontsize=18)
    ax[0].legend(fontsize=14)
    ax[0].grid()
    # ax[0].set_xticks([0, 5, 10, 15, 20])
    # ax[0].set_yticks([0, 5, 10, 15, 20])
                       
    ax[1].set_ylim(ymin, ymax)
    ax[1].set_xlim(xmin, xmax)
    ax[1].set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax[1].set_xlabel('parameter point', fontsize=18)
    ax[1].set_ylabel('coverage', fontsize=18)

    # cdfhat_tau_df = get_conf_set_at_given_tau(with_lambda_D=with_lambda_D, PARAMS=PARAMS, tau=0.68)
    # cdfhat_tau_df.sample(frac=1)
    # tau=0.68
    colors = ['red', 'darkorange', 'royalblue', 'darkgreen']
    for tau, color in zip(tau_levels, colors):
        # ax.axhline(y=tau)
        cdfhat_tau_df = get_conf_set_at_given_tau(with_lambda_D=with_lambda_D, PARAMS=PARAMS, tau=tau)
        cdfhat_tau_df.sample(frac=1)
        # cdfhat_tau_df=cdfhat_tau_df[:501]
        theta_edge = cdfhat_tau_df['theta']
        nu_edge = cdfhat_tau_df['nu']
        # cdfhat_at_tau = cdfhat_tau_df['cdfhat']
    # cdfhat_tau_df=cdfhat_tau_df[:501]
    # theta_edge = cdfhat_tau_df['theta']
    # nu_edge = cdfhat_tau_df['nu']
        for ind, (theta, nu) in enumerate(zip(theta_edge, nu_edge)):

            N = st.poisson.rvs(theta+nu, size=Npoints)
            M = st.poisson.rvs(nu, size=Npoints)
            lambda_D = lambda_test_2d(N, M, theta, nu, MLE).flatten()
            # print(lambda_D)
            theta_arr = np.full(lambda_D.shape, theta)
            nu_arr = np.full(lambda_D.shape, nu)
            eval_arr = np.empty((Npoints, 3), dtype=float)
            eval_arr[:,0]=theta_arr
            eval_arr[:,1]=nu_arr
            eval_arr[:,2] = lambda_D
            # print(eval_arr.shape)
            # print(theta_arr.shape, nu_arr.shape, lambda_D.shape)
            eval_tensor= torch.from_numpy(eval_arr).float()
            # print(eval_tensor)
            cdfhat_at_point=  trained_model(eval_tensor).view(-1,).detach().numpy()
            # cdfhat_at_point = 1- cdfhat_at_point
            coverage_prob = np.mean(cdfhat_at_point <= tau)
            ax[1].scatter(ind, coverage_prob, s=1, c= color)
            ax[1].plot([0, 501], [tau,tau], c=color, linewidth=2)
    ax[1].grid()
    plt.tight_layout()    
            
    if saveplot==True:
        plt.savefig(os.path.join(os.environ['LFI_PIVOT_BASE'], 'images', 'JUNE_6_ON_OFF_coverage_at_0.68_0.8_0.9_0.95_500_POINTS.eps'))
    fig.show(); plt.show()
        
            
if __name__ == '__main__':
    # PARAMS_lambdaD_nonMLE = {
    # "n_layers": int(5),
    # "hidden_size": int(11),
    # "dropout": float(0.13),
    # "NFEATURES":int(3),
    # "activation": "LeakyReLU",
    # # 'optimizer_name':'NAdam',
    #     'optimizer_name':'RMSprop',
    # 'starting_learning_rate':float(0.0064),
    # 'momentum':float(0.6),
    # 'batch_size':int(1000),
    # 'n_iterations': int(2e6),
    # 'traces_step':int(400),
    # 'L2':float(0.1),
    # 'MLE':False,
    # 'with_lambda_D':True,
    # 'pth_string':'model_lambda_D_nonMLE.pth'
    # }
    PARAMS_lambdaD_nonMLE = {
    "n_layers": int(6),
    "hidden_size": int(6),
    "dropout": float(0.13),
    "NFEATURES":int(3),
    "activation": "PReLU",
    'optimizer_name':'NAdam',
        # 'optimizer_name':'RMSprop',
    'starting_learning_rate':float(0.0006),
    'momentum':float(0.9),
    'batch_size':int(256*4),
    'n_iterations': int(3e4),
    'traces_step':int(100),
    'L2':float(0.1),
    'MLE':False,
    'with_lambda_D':True,
    'pth_string':'JUNE_1_model_lambda_D_nonMLE.pth'
    }
    # eval_df = load_2d_train_df(MLE=True, with_lambda_D=True)
    # train_t, train_x, test_t,  test_x = getwholedata_2d(MLE=True, valid=False,with_lambda_D=True)
    # print(train_x.shape)



    # calc_coverage_at_given_tau(MLE=False, with_lambda_D=True, tau=0.68, Npoints=400)
    
#     model_lambda_D_nonMLE = load_untrained_model(PARAMS_lambdaD_nonMLE)
#     model_lambda_D_nonMLE = load_model(model=model_lambda_D_nonMLE, 
#                                         PARAMS=PARAMS_lambdaD_nonMLE, 
#                                         pth_string=PARAMS_lambdaD_nonMLE['pth_string']
#                                         )
    
#     cdfhat_df = evaluate_2d_model_on_df(with_lambda_D=True, PARAMS=PARAMS_lambdaD_nonMLE)
#     print(cdfhat_df.head())
#     cdfhat_68 = get_conf_set_at_given_tau(with_lambda_D=True, PARAMS=PARAMS_lambdaD_nonMLE, tau=0.68)
#     print(cdfhat_68.head())   
# # for each fixed parameter point $\theta_f = \{ \mu_f, \nu_f \}$., 
    plot_coverage_levels(PARAMS=PARAMS_lambdaD_nonMLE, with_lambda_D=True, Npoints=5000, saveplot=True)
