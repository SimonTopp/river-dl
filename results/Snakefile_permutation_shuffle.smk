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

def noise_segs(model, input_data, n_segs, pred_length):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    data_year = input_data['x_tst'][pd.DatetimeIndex(input_data['times_tst'][:,-1,0]).year >= 2014]
    num_sequences = int(data_year.shape[0]/455)
    diffs = []
    x_full = torch.from_numpy(input_data['x_tst']).to(device).float()
    for i in range(0,455*num_sequences,455):
        start_ind = i
        end_ind = i+455
        x = torch.from_numpy(data_year[start_ind:end_ind]).to(device).float()
        for j in range(n_segs):
            with torch.no_grad():
                y_hat_original = model(x)[:,-pred_length:,:]
            x_hypothesis = x_full.detach().clone()[torch.randperm(x.shape[0])]
            x_hypothesis[j] = x[j]
            with torch.no_grad():
                y_hat_hypothesis = model(x_hypothesis)[:,-pred_length:,:]
            y_diff = y_hat_original[j, :].detach().cpu() - y_hat_hypothesis[j, :].detach().cpu()
            y_diff = np.mean(np.abs(y_diff.numpy()))
            diffs.append(y_diff)
    diffs=np.asarray(diffs).reshape(num_sequences,n_segs).mean(axis=0)
    diffs = diffs*input_data['y_std'][0]
    return diffs


def compare_temporally_altered(model, data_in, num_rand, season, season_label, n_segs=455):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    dates = data_in['times_tst']
    months = dates.astype('datetime64[M]').astype(int) % 12 + 1 
    mask= np.isin(months[:,-1,0],season)
    batches = data_in['x_tst'][mask].shape[0]//n_segs
    sd_full = data_in['y_std'][0]
    rand_batch = np.random.choice(range(1,batches),num_rand,replace='False')
    data_in_full = torch.from_numpy(data_in['x_tst']).to(device).float()
    data_in = data_in_full[mask]
    noised_out=[]
    for i in rand_batch:
        start_ind = n_segs*i-n_segs
        end_ind = n_segs*i
        x = data_in[start_ind:end_ind]
        y_hat_original = model(x)
        # shuffle first n days
        noised = []
        for j in range(1,data_in.shape[1],1):
            x_hypothesis = x.detach().clone()
            x_hypothesis[:, :-j] = data_in_full[torch.randperm(455)][:,torch.randperm(x.shape[1]-j)]
            y_hat_hypothesis = model(x_hypothesis)
            diff = torch.abs(y_hat_original[:,-1]-y_hat_hypothesis[:,-1]).squeeze()
            noised.append(diff.detach().cpu().numpy().mean())    
        noised_out.append(np.asarray(noised))
    mean = np.mean(noised_out, axis=0)*sd_full 
    sd = np.std(noised_out,axis=0)*sd_full
    out = pd.DataFrame({'diffs_mean':mean,'diffs_sd':sd,'seq_num':np.arange(len(mean))})
    out['season']=season_label
    return out

## Master Rule
rule all:
    input:
        expand("results/xai_outputs/noise_annual_shuffle/{model}_{run}_reach_noise.csv",
            model=['GWN','RGCN'],
            run=['nopt','ptft'],#'pt','coast_ptft', 'head_nopt','head_ptft','coast_nopt', 'rs_adj'
        ),
        expand("results/xai_outputs/noise_seasonal_shuffle/{model}_{run}_seasonal_noise.csv",
            model=['GWN','RGCN'],
            run=['nopt','ptft']#,'pt','coast_ptft', 'head_nopt','head_ptft','coast_nopt', 'rs_adj'],
        )
        

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


rule gwn_reach_noise:
    input:
        f"results/baseline/GWN/prepped.npz",
    output:
        "results/xai_outputs/noise_annual_shuffle/{model}_{run}_reach_noise.csv"
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
        diffs = noise_segs(gwn, data_gwn,n_segs,15)
        diffs_df = pd.DataFrame({'seg_id_nat':seg_ids, 
                                 'diffs':diffs, 
                                 'model':'GWN',
                                 'run':wildcards.run})
        
        diffs_df.to_csv(f"results/xai_outputs/noise_annual_shuffle/GWN_{wildcards.run}_reach_noise.csv",
                        index=False)
   
rule rgcn_reach_noise:
    input:
        f"results/baseline/RGCN/prepped.npz"
    output:
        "results/xai_outputs/noise_annual_shuffle/{model}_{run}_reach_noise.csv"
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
        
        rgcn = RGCN_v1(num_vars, 20, adj_matrix_rgcn, recur_dropout=0.3, device=device)
        rgcn.load_state_dict(torch.load(weights_file, map_location=device))
        rgcn.to(device)
        rgcn.eval()
        diffs = noise_segs(rgcn, data_rgcn,n_segs,90)
        diffs_df = pd.DataFrame({'seg_id_nat':seg_ids, 
                                 'diffs':diffs, 
                                 'model':'RGCN',
                                 'run':wildcards.run})
        
        diffs_df.to_csv(f"results/xai_outputs/noise_annual_shuffle/RGCN_{wildcards.run}_reach_noise.csv",
                        index=False)
        
        

rule gwn_seasonal_noise:
    input:
        f"results/baseline/GWN/prepped.npz",
    output:
        "results/xai_outputs/noise_seasonal_shuffle/{model}_{run}_seasonal_noise.csv"
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
        
        seasons=[[6,7,8],[9,10,11],[12,1,2],[3,4,5]]
        jja = compare_temporally_altered(gwn,data_gwn,30,[6,7,8],'JJA')
        son = compare_temporally_altered(gwn,data_gwn,30,[9,10,11],'SON')
        djf = compare_temporally_altered(gwn,data_gwn,30,[12,1,2],'DJF')
        mam = compare_temporally_altered(gwn,data_gwn,30,[3,4,5],'MAM')
        
        out = pd.concat([jja,son,djf,mam])
        
        out.to_csv(f"results/xai_outputs/noise_seasonal_shuffle/GWN_{wildcards.run}_seasonal_noise.csv",
                        index=False)
   
rule rgcn_seasonal_noise:
    input:
        f"results/baseline/RGCN/prepped.npz"
    output:
        "results/xai_outputs/noise_seasonal_shuffle/{model}_{run}_seasonal_noise.csv"
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
        
        rgcn = RGCN_v1(num_vars, 20, adj_matrix_rgcn, recur_dropout=0.3, device=device)
        rgcn.load_state_dict(torch.load(weights_file, map_location=device))
        rgcn.to(device)
        rgcn.eval()
        
        jja = compare_temporally_altered(rgcn,data_rgcn,30,[6,7,8],'JJA')
        son = compare_temporally_altered(rgcn,data_rgcn,30,[9,10,11],'SON')
        djf = compare_temporally_altered(rgcn,data_rgcn,30,[12,1,2],'DJF')
        mam = compare_temporally_altered(rgcn,data_rgcn,30,[3,4,5],'MAM')
        
        out = pd.concat([jja,son,djf,mam])
        
        out.to_csv(f"results/xai_outputs/noise_seasonal_shuffle/RGCN_{wildcards.run}_seasonal_noise.csv",
                        index=False)
   