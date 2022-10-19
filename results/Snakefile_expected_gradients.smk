import os
import random

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.nn import BatchNorm2d, Conv1d, Conv2d, ModuleList, Parameter
import torch.nn.functional as F
import xarray as xr
import pandas as pd
#import sys; sys.path.insert(0, '..')

from river_dl.torch_models import gwnet, RGCN_v1

class gwnet_wrapper(nn.Module):
    def __init__(self, device, num_nodes, dropout=0.3, supports=None, gcn_bool=True,
                 addaptadj=True, aptinit=None, in_dim=2, out_dim=12,
                 residual_channels=32, dilation_channels=32,
                 skip_channels=256, end_channels=512, kernel_size=2, blocks=4, layers=2,
                 apt_size=10, cat_feat_gc=False,weights_path=None,nsegs=455):
        super().__init__()
        self.nsegs=nsegs
        self.out_dim = out_dim
        self.model=gwnet(device,num_nodes, dropout, supports, gcn_bool,
                 addaptadj, aptinit, in_dim, out_dim,
                 residual_channels, dilation_channels,
                 skip_channels, end_channels, kernel_size, blocks, layers,
                 apt_size, cat_feat_gc)

        if weights_path is not None:
            self.model.load_state_dict(torch.load(weights_path, map_location=device))
            self.model.to(device)

    def forward(self, x):
        shape_x=x.shape
        x = x.reshape(int(shape_x[0] / self.nsegs), self.nsegs, shape_x[1], shape_x[2]).movedim((1,2,3),(2,3,1))
        x=self.model(x)
        x = x.squeeze(dim=0)
        return torch.movedim(x, (0, 1, 2), (2, 0, 1)).reshape(shape_x[0],self.out_dim,1)
        #return torch.movedim(x, (0, 1, 2), (2, 0, 1)).reshape(shape_x)

def expected_gradients(x, x_set, adj_matrix, model, n_samples, temporal_focus=None, spatial_focus=None):

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    n_series = x_set.shape[0]
    n_segs = adj_matrix.shape[0]
    num_vars = x_set.shape[2]
    seq_len = x_set.shape[1]

    x_set_4D = x_set.reshape(n_series//n_segs,n_segs,seq_len,num_vars)

    for k in range(n_samples):
        # SAMPLE A RANDOM BASELINE INPUT
        rand_year = np.random.choice(n_series//n_segs) # rand_time may be more accurate
        baseline_x = x_set_4D[rand_year].to(device)

        # SAMPLE A RANDOM SCALE ALONG THE DIFFERENCE
        scale = np.random.uniform()

        # SAME IG CALCULATION
        x_diff = x - baseline_x
        curr_x = baseline_x + scale*x_diff
        if curr_x.requires_grad == False:
            curr_x.requires_grad = True
        model.zero_grad()
        y = model(curr_x)

        # GET GRADIENT
        if temporal_focus == None and spatial_focus == None:
            gradients = torch.autograd.grad(y[:, :, :], curr_x, torch.ones_like(y[:, :, :]))
        elif temporal_focus == None and spatial_focus != None:
            gradients = torch.autograd.grad(y[spatial_focus, :, :], curr_x, torch.ones_like(y[spatial_focus, :, :]))
        elif temporal_focus != None and spatial_focus == None:
            gradients = torch.autograd.grad(y[:, temporal_focus, :], curr_x, torch.ones_like(y[:, temporal_focus, :]))
        else:
            gradients = torch.autograd.grad(y[spatial_focus, temporal_focus, :], curr_x, torch.ones_like(y[spatial_focus, temporal_focus, :]))

        if k == 0:
            expected_gradients = x_diff*gradients[0] * 1/n_samples
        else:
            expected_gradients = expected_gradients + ((x_diff*gradients[0]) * 1/n_samples)

    return(expected_gradients.detach().cpu().numpy())

def reach_egs(model, data_in,reach, pred_length, seg_ids):
    reach = np.where(seg_ids == reach)[0][0]
    #batches = data_in['x_trn'].shape[0]//n_segs
    num_sequences = int(365/pred_length)
    egs = []
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    for i in range(0,455*num_sequences,455):
        
        start_ind = i
        end_ind = i+455
        x = torch.from_numpy(data_in['x_trn']).to(device).float()[start_ind:end_ind]
        EG_vals = expected_gradients(x, 
                                     torch.from_numpy(data_in['x_trn']).to(device).float(),
                                     data_in['dist_matrix'],
                                     model, 
                                     n_samples=200, 
                                     spatial_focus=reach)
        EG_vals=np.abs(EG_vals)
        eg_sum = np.sum(EG_vals.flatten())
        EG_vals = EG_vals/eg_sum
        #print(EG_vals.flatten().sum())
        #EG_vals[reach,:,:] = np.nan
        reduced = np.sum(EG_vals,axis=1) ## Sum across time steps to get cumulative EG per reach
        egs.append(reduced)
    egs = np.asarray(egs).mean(axis=0)
    return egs

def seasonal_egs(model, data_in, num_rand, n_segs=455):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    dates = data_in['times_trn']
    months = dates.astype('datetime64[M]').astype(int) % 12 + 1 #d.DatetimeIndex(river_dl['times_tst'])
    seasons = [[6,7,8],[9,10,11],[12,1,2],[3,4,5]]
    season_labs = ['JJA','SON','DJF','MAM']
    egs_months= pd.DataFrame(columns=np.concatenate([data_in['x_vars'],['season','metric','seq_num']]))
    for m in range(4):
        mask= np.isin(months[:,-1,0],seasons[m])
        batches = data_in['x_trn'][mask].shape[0]//n_segs
        rand_batch = np.random.choice(range(1,batches),num_rand,replace='False')
        egs=[]
        for i in rand_batch:
            start_ind = n_segs*i
            end_ind = n_segs*i+n_segs
            x = torch.from_numpy(data_in['x_trn'][mask]).to(device).float()[start_ind:end_ind]
            EG_vals = expected_gradients(x, 
                                    torch.from_numpy(data_in['x_trn']).to(device).float(),
                                    data_in['dist_matrix'],
                                    model, 
                                    n_samples=200, 
                                    temporal_focus=-1)
            reduced = np.mean(EG_vals,axis=0) #reduce across segments
            egs.append(reduced)
        egs_mean = pd.DataFrame(columns=data_in['x_vars'],data= np.asarray(egs).mean(axis=0)) #reduce across batches
        egs_mean['season']=season_labs[m]
        egs_mean['metric']='mean'
        egs_mean['seq_num'] = np.arange(data_in['x_trn'].shape[1])
        egs_std = pd.DataFrame(columns=data_in['x_vars'],data= np.asarray(egs).std(axis=0)) #reduce across batches
        egs_std['season']=season_labs[m]
        egs_std['metric']='sd'
        egs_std['seq_num'] = np.arange(data_in['x_trn'].shape[1])
                                   
        egs_months = pd.concat([egs_months,egs_mean, egs_std])
        
    return egs_months
                             
## Master Rule
rule all:
    input:
        expand("results/xai_outputs/egs_reach_anual/{model}_{run}_{reach}_egs.csv",
               model=['RGCN','GWN'],
               run= ['ptft','nopt','coast_ptft','coast_nopt'], #['pt','nopt','ptft','coast_ptft', 'head_nopt','head_ptft','coast_nopt'],
               reach=['4189','4206', '1487', '1577', '2318']),
        expand("results/xai_outputs/egs_seasonal/{model}_{run}_seasonal_egs.csv",
               model=['RGCN','GWN'],
               run=['ptft'])#,'nopt','coast_ptft','coast_nopt'], #['pt','nopt','ptft','coast_ptft', 'head_nopt','head_ptft','coast_nopt'])
     

def return_weights_file(wildcards):
    if wildcards.model == 'GWN':
        run_number = 6
    elif wildcards.model == "RGCN":
        run_number = 2
    if wildcards.run == 'pt':
        return f'results/baseline/{wildcards.model}/rep_pre_{run_number}/pretrained_weights.pth'
    if wildcards.run == 'ptft':
        return f'results/baseline/{wildcards.model}/full_train/rep_{run_number}/finetuned_weights.pth'
    if wildcards.run == 'nopt':
        return f'results/baseline/{wildcards.model}/no_pt/rep_{run_number}/finetuned_weights.pth'
    if wildcards.run == 'coast_ptft':
        return f'results/LLO/coastal/{wildcards.model}/full_train/rep_{run_number}/finetuned_weights.pth'
    if wildcards.run == 'coast_nopt':
        return f'results/LLO/coastal/{wildcards.model}/no_pt/rep_{run_number}/finetuned_weights.pth'
    if wildcards.run == 'head_nopt':
        return f'results/LLO/piedmont/{wildcards.model}/no_pt/rep_{run_number}/finetuned_weights.pth'
    if wildcards.run == 'head_ptft':
        return f'results/LLO/piedmont/{wildcards.model}/full_train/rep_{run_number}/finetuned_weights.pth'
    if wildcards.run == 'rs_adj':
        return f'results/baseline/{wildcards.model}/reset_spatial/rep_{run_number}/finetuned_weights.pth'


rule gwn_egs:
    input:
        f"results/baseline/GWN/prepped.npz",
    output:
        "results/xai_outputs/egs_reach_anual/{model}_{run}_{reach}_egs.csv"
    wildcard_constraints:
        model='GWN'
    params: weights = return_weights_file
    run:
        ### GWN  ##6 is best baseline rep
        weights_file = params.weights
        prepped_file = f"results/baseline/GWN/prepped.npz"
        
        data_gwn = np.load(prepped_file)
        adj_matrix_gwn = data_gwn['dist_matrix']
        num_vars = len(data_gwn['x_vars'])
        x_vars= data_gwn['x_vars']
        seq_len_gwn = data_gwn['x_trn'].shape[1]
        n_segs = adj_matrix_gwn.shape[0]
        out_dim_gwn = 15
        seg_ids = data_gwn['ids_trn'][-455:][:,0,:].flatten()
        n_hid = 20
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(device)
        supports = [torch.tensor(adj_matrix_gwn).to(device).float()]
        gwn = gwnet_wrapper(device,n_segs,supports=supports,aptinit=supports[0],
                            in_dim=num_vars,out_dim=out_dim_gwn,residual_channels=n_hid,
                            dilation_channels=n_hid,skip_channels=n_hid*4,
                            end_channels=n_hid*8,kernel_size=2,blocks=4,layers=4, 
                            weights_path=weights_file,
                            nsegs=n_segs)
        gwn.eval()
        diffs = reach_egs(gwn, data_gwn, int(wildcards.reach),15,seg_ids)
        diffs_df = pd.DataFrame(columns=x_vars,data=diffs)
        diffs_df['seg_id_nat']=seg_ids
        diffs_df['model']='GWN'
        diffs_df['run']=wildcards.run
        diffs_df['target_reach']=wildcards.reach
        
        diffs_df.to_csv(f"results/xai_outputs/egs_reach_anual/GWN_{wildcards.run}_{wildcards.reach}_egs.csv",
                        index=False)
   
rule rgcn_egs:
    input:
        f"results/baseline/RGCN/prepped.npz"
    output:
        "results/xai_outputs/egs_reach_anual/{model}_{run}_{reach}_egs.csv"
    wildcard_constraints:
        model='RGCN'
    params: weights = return_weights_file
    run:
        ### GWN  ##6 is best baseline rep
        weights_file = params.weights
        prepped_file = f"results/baseline/RGCN/prepped.npz"
        data_rgcn = np.load(prepped_file)
        adj_matrix_rgcn = data_rgcn['dist_matrix']
        seq_len_rgcn = data_rgcn['x_trn'].shape[1]
        seg_ids = data_rgcn['ids_trn'][-455:][:,0,:].flatten()
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        num_vars = len(data_rgcn['x_vars'])
        n_segs = adj_matrix_rgcn.shape[0]
        x_vars= data_rgcn['x_vars']
        
        
        rgcn = RGCN_v1(num_vars, 20, adj_matrix_rgcn, recur_dropout=0.3, device=device)
        rgcn.load_state_dict(torch.load(weights_file, map_location=device))
        rgcn.to(device)
        rgcn.eval()
        
        diffs = reach_egs(rgcn, data_rgcn, int(wildcards.reach), 90,seg_ids)
        diffs_df = pd.DataFrame(columns=x_vars,data=diffs)
        diffs_df['seg_id_nat']=seg_ids
        diffs_df['model']='RGCN'
        diffs_df['run']=wildcards.run
        diffs_df['target_reach']=wildcards.reach
        
        diffs_df.to_csv(f"results/xai_outputs/egs_reach_anual/RGCN_{wildcards.run}_{wildcards.reach}_egs.csv",
                        index=False)

                             
rule gwn_seaonal_egs:
    input:
        f"results/baseline/GWN/prepped.npz",
    output:
        "results/xai_outputs/egs_seasonal/{model}_{run}_seasonal_egs.csv"
    wildcard_constraints:
        model='GWN'
    params: weights = return_weights_file
    run:
        ### GWN  ##6 is best baseline rep
        weights_file = params.weights
        prepped_file = f"results/baseline/GWN/prepped.npz"
        data_gwn = np.load(prepped_file)
        adj_matrix_gwn = data_gwn['dist_matrix']
        num_vars = len(data_gwn['x_vars'])
        x_vars= data_gwn['x_vars']
        seq_len_gwn = data_gwn['x_trn'].shape[1]
        n_segs = adj_matrix_gwn.shape[0]
        out_dim_gwn = 15
        seg_ids = data_gwn['ids_trn'][-455:][:,0,:].flatten()
        n_hid = 20
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(device)
        supports = [torch.tensor(adj_matrix_gwn).to(device).float()]
        gwn = gwnet_wrapper(device,n_segs,supports=supports,aptinit=supports[0],
                            in_dim=num_vars,out_dim=out_dim_gwn,residual_channels=n_hid,
                            dilation_channels=n_hid,skip_channels=n_hid*4,
                            end_channels=n_hid*8,kernel_size=2,blocks=4,layers=4, 
                            weights_path=weights_file,
                            nsegs=n_segs)
        gwn.eval()
        egs = seasonal_egs(gwn, data_gwn,30)
        egs.to_csv(f"results/xai_outputs/egs_seasonal/GWN_{wildcards.run}_seasonal_egs.csv",
                        index=False)
   
rule rgcn_seasonal_egs:
    input:
        f"results/baseline/RGCN/prepped.npz"
    output:
        "results/xai_outputs/egs_seasonal/{model}_{run}_seasonal_egs.csv"
    wildcard_constraints:
        model='RGCN'
    params: weights = return_weights_file
    run:
        ### GWN  ##6 is best baseline rep
        weights_file = params.weights
        prepped_file = f"results/baseline/RGCN/prepped.npz"
        data_rgcn = np.load(prepped_file)
        adj_matrix_rgcn = data_rgcn['dist_matrix']
        seq_len_rgcn = data_rgcn['x_trn'].shape[1]
        seg_ids = data_rgcn['ids_trn'][-455:][:,0,:].flatten()
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        num_vars = len(data_rgcn['x_vars'])
        n_segs = adj_matrix_rgcn.shape[0]
        x_vars= data_rgcn['x_vars']
    
        rgcn = RGCN_v1(num_vars, 20, adj_matrix_rgcn, recur_dropout=0.3, device=device)
        rgcn.load_state_dict(torch.load(weights_file, map_location=device))
        rgcn.to(device)
        rgcn.eval()
        
        egs = seasonal_egs(rgcn, data_rgcn, 30)
        
        egs.to_csv(f"results/xai_outputs/egs_seasonal/RGCN_{wildcards.run}_seasonal_egs.csv",
                        index=False)
        