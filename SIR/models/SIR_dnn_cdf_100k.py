
name     = 'SIR_dnn_cdf_100k'
features = ['lo', 'alpha', 'beta']
target   = 'Zo'
nodes    = 25

import torch
import torch.nn as nn

model = nn.Sequential(nn.Linear( len(features), nodes),
                      nn.ReLU(),
                      
                      nn.Linear(nodes, nodes),
                      nn.ReLU(),
                      
                      nn.Linear(nodes, nodes),
                      nn.ReLU(), 
                      
                      nn.Linear(nodes, nodes),
                      nn.ReLU(),                    
                      
                      nn.Linear(nodes, nodes),
                      nn.ReLU(), 
                      
                      nn.Linear(nodes, nodes),
                      nn.ReLU(),                    
                      
                      nn.Linear(nodes, 1), 
                      nn.Sigmoid()) 
