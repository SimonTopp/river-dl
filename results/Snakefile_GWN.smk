import os
import random

import numpy as np
import torch
import torch.optim as optim
import xarray as xr
import pandas as pd

from river_dl.preproc_utils import asRunConfig
from river_dl.preproc_utils import prep_all_data
from river_dl.torch_utils import train_torch
from river_dl.torch_utils import rmse_masked, reshape_for_gwn
from river_dl.evaluate import combined_metrics
from river_dl.torch_models import gwnet
from river_dl.predict import predict_from_io_data

### Set a couple of variables to use as wildcards
code_dir = ''
replicates = range(config['num_reps'])
model = [config['model']]
out_main = [config['results_dir']]


### Couple of helper function for specifying scenarios
### Pull 10 hottest/coldest years
def pull_lto_years(air_temp, ntest=10, nval=5, test='max'):
    """
    Pull years with min and max temperature values for leave time out tests
    :param obs: [csv] temperature observations to pull min and max years from
    :param ntest: [int] number of years to set aside for testing
    :param nval: [int] number of years to set aside for validation
    :parm test: [str] whether to set aside max temp years or min temp years for testing (option = 'max','min')
    """
    temp_df = xr.open_zarr(air_temp,consolidated=False)
    temp_df = temp_df['seg_tave_air'].to_dataframe()
    temp_df['date'] = pd.to_datetime(temp_df.index.get_level_values(0))
    temp_df['water_year'] = temp_df.date.dt.year.where(temp_df.date.dt.month < 10,temp_df.date.dt.year + 1)
    temp_df['month'] = temp_df.date.dt.month
    temp_df = temp_df[temp_df.month.isin([6, 7, 8])]
    mean_temp = temp_df[['seg_tave_air', 'water_year']].groupby('water_year').mean()
    if test == 'max':
        years_test = mean_temp.sort_values('seg_tave_air',ascending=False)[0:ntest]
    if test == 'min':
        years_test = mean_temp.sort_values('seg_tave_air')[0:ntest]

    mean_temp = mean_temp[~mean_temp.index.isin(years_test.index)]

    ## We prefer consecutive years, we'll say 2000 and next five years that aren't in test
    ## This overlaps with original val years and has some back to back years for hot runs
    years_val = mean_temp.loc[mean_temp.index > 2000].iloc[0:nval]

    years_train = mean_temp[~mean_temp.index.isin(years_val.index)]

    def wy_to_date(wys):
        return {'start': [f"{i - 1}-10-01" for i in wys.index], 'end': [f"{i}-9-30" for i in wys.index]}

    train_out = wy_to_date(years_train)
    test_out = wy_to_date(years_test)
    val_out = wy_to_date(years_val)

    return train_out, test_out, val_out


### Get ID's for geographic generalizations scenarios
def return_seg_masks(group, group_csv):
    llo = pd.read_csv(group_csv).dropna()
    llo.seg_id_nat = llo.seg_id_nat.astype(int)
    if group == 'coastal':
        segs_out = {'train': llo.seg_id_nat[llo.test_group != 'Coastal_Plains'].to_list(),
                    'test': llo.seg_id_nat[llo.test_group == 'Coastal_Plains'].to_list()}
    elif group == 'piedmont':
        segs_out = {'train': llo.seg_id_nat[llo.test_group != 'Piedmont'].to_list(),
                    'test': llo.seg_id_nat[llo.test_group == 'Piedmont'].to_list()}
    elif group == 'appalachians':
        segs_out = {'train': llo.seg_id_nat[llo.test_group != 'Appalachians'].to_list(),
                    'test': llo.seg_id_nat[llo.test_group == 'Appalachians'].to_list()}
    else:
        raise ValueError("Please select valid holdout group")
    return segs_out


### Pull 10 dryest water years including those containing official droughts
def pull_drought_years(precip_data, ntest=10, nval=5):
    temp_df = xr.open_zarr(precip_data,consolidated=False)
    temp_df = temp_df['seg_rain'].to_dataframe()
    temp_df['date'] = pd.to_datetime(temp_df.index.get_level_values(0))
    temp_df['water_year'] = temp_df.date.dt.year.where(temp_df.date.dt.month < 10,temp_df.date.dt.year + 1)
    cum_precip = temp_df[['seg_rain', 'water_year']].groupby('water_year').sum()

    # Water years containing officially declared droughts
    # https://www.state.nj.us/drbc/library/documents/drought/DRBdrought-overview_feb2019.pdf
    drought_years = np.array([1981, 1982, 1985, 1986, 1999, 2002, 2003])
    cum_precip = cum_precip.loc[~cum_precip.index.isin(drought_years)]
    years_test = cum_precip.sort_values('seg_rain')[0:ntest - len(drought_years)]
    years_test = np.concatenate((drought_years, years_test.index.values))
    years_test.sort()
    cum_precip = cum_precip.loc[~cum_precip.index.isin(years_test)]

    ## We prefer consecutive years, we'll say 2000 and next five years that aren't in test
    ## This overlaps with original val years and has some back to back years for hot runs
    years_val = cum_precip.loc[cum_precip.index > 2000].iloc[0:nval].index.values
    years_train = cum_precip.loc[~cum_precip.index.isin(years_val)].index.values

    def wy_to_date(wys):
        return {'start': [f"{i - 1}-10-01" for i in wys], 'end': [f"{i}-9-30" for i in wys]}

    train_out = wy_to_date(years_train)
    test_out = wy_to_date(years_test)
    val_out = wy_to_date(years_val)

    return train_out, test_out, val_out


### Wrap all the above into one parameter function to determine scenario settings from
### the wildcards
def get_prep_args(wildcards):
    if wildcards.outdir == 'baseline':
        train = {'start': config['train_start_date'], 'end': config['train_end_date']}
        test = {'start': config['test_start_date'], 'end': config['test_end_date']}
        val = {'start': config['val_start_date'], 'end': config['val_end_date']}
        seg_masks = {'train': None, 'test': None}
        return train, test, val, seg_masks
    elif wildcards.outdir.split('/')[0] == 'LTO':
        train, test, val = pull_lto_years(config['sntemp_file'],test=wildcards.outdir.split('/')[1])
        seg_masks = {'train': None, 'test': None}
        return train, test, val, seg_masks
    elif wildcards.outdir.split('/')[0] == 'LLO':
        train = {'start': ['1979-10-01', '2010-10-01'], 'end': ['2006-09-30', '2021-09-30']}
        test = {'start': ['1979-10-01', '2010-10-01'], 'end': ['2006-09-30', '2021-09-30']}
        val = {'start': ['2006-10-01'], 'end': ['2010-09-30']}
        seg_masks = return_seg_masks(wildcards.outdir.split('/')[1],config['llo_csv'])
        return train, test, val, seg_masks
    elif wildcards.outdir == 'Drought':
        train, test, val = pull_drought_years(config['sntemp_file'])
        seg_masks = {'train': None, 'test': None}
        return train, test, val, seg_masks


## Master Rule
rule all:
    input:
        expand("results/{outdir}/GWN/{train_type}/rep_{replicate}/{metric_type}_metrics.csv",
            out_main=out_main,
            outdir=['baseline', 'LTO/min', 'LTO/max', 'LLO/appalachians', 'LLO/piedmont', 'LLO/coastal', 'Drought'],
            train_type=['full_train','no_pt','reset_spatial'],
            replicate=replicates,
            metric_type=['overall', 'month', 'reach'],
            model=model
        ),
        expand("results/asRunConfig_GWN.yml",model=model),
        expand("results/Snakefile_GWN.smk",model=model)


## Log config file
rule as_run_config:
    output:
        "results/asRunConfig_GWN.yml"
    group: "prep"
    run:
        asRunConfig(config,code_dir,output[0])

## Make a copy of Snakefile to re-run
rule copy_snakefile:
    output:
        "results/Snakefile_GWN.smk"
    group: "prep"
    shell:
        """
        scp Snakefile.smk {output[0]}
        """


## Prep the IO data for all the various runs
rule prep_io_data:
    input:
        config['sntemp_file'],
        config['obs_file'],
        config['dist_matrix_file'],
    output:
        "results/{outdir}/GWN/prepped.npz"
    threads: 2
    params: prep_args=get_prep_args
    group: "prep"
    run:
        prep_all_data(
            x_data_file=input[0],
            pretrain_file=input[0],
            y_data_file=input[1],
            distfile=input[2],
            x_vars=config['x_vars'],
            y_vars_pretrain=config['y_vars_pretrain'],
            y_vars_finetune=config['y_vars_finetune'],
            catch_prop_file=None,
            exclude_file=None,
            train_start_date=params.prep_args[0]['start'],
            train_end_date=params.prep_args[0]['end'],
            val_start_date=params.prep_args[2]['start'],
            val_end_date=params.prep_args[2]['end'],
            test_start_date=params.prep_args[1]['start'],
            test_end_date=params.prep_args[1]['end'],
            segs=None,
            out_file=output[0],
            trn_offset=config['trn_offset'],
            tst_val_offset=config['tst_val_offset'],
            seq_len=config['seq_length'],
            test_sites=params.prep_args[3]['test'],
        )


# Pretrain the model on process based model, we'll just do this once and then fine-tune to all the
# scenarios.  Using f-strings here avoids confusion with wildcards

# Pretrain the model on process based model, we'll just do this once and then fine-tune to all the
# scenarios.  Using f-strings here avoids confusion with wildcards
rule pre_train:
    input:
        "results/baseline/GWN/prepped.npz"
    output:
        "results/baseline/GWN/rep_pre_{replicate}/pretrained_weights.pth",
        "results/baseline/GWN/rep_pre_{replicate}/pretrain_log.csv"
    threads: 4
    group: 'pre_train'
    run:
        os.system("module load analytics cuda11.3/toolkit/11.3.0")
        os.system("export LD_LIBRARY_PATH=/cm/shared/apps/nvidia/TensorRT-6.0.1.5/lib:/cm/shared/apps/nvidia/cudnn_8.0.5/lib64:$LD_LIBRARY_PATH")

        data = np.load(input[0])
        data = reshape_for_gwn(data,keep_portion=config['trn_offset'])
        adj_mx = data['dist_matrix']
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        supports = [torch.tensor(adj_mx).to(device).float()]
        in_dim = len(data['x_vars'])
        out_dim = data['y_obs_trn'].shape[3]
        num_nodes = adj_mx.shape[0]
        lrate = config['pretrain_learning_rate']
        model = gwnet(device,num_nodes,supports=supports,aptinit=supports[0],
            in_dim=in_dim,out_dim=out_dim,residual_channels=config['hidden_size'],
            dilation_channels=config['hidden_size'],skip_channels=config['hidden_size'] * 4,
            end_channels=config['hidden_size'] * 8,kernel_size=2,blocks=4,layers=4,
            dropout=config['dropout'])

        opt = optim.Adam(model.parameters(),lr=lrate,weight_decay=0.0001)

        train_torch(model,
            loss_function=rmse_masked,
            optimizer=opt,
            x_train=data['x_pre_full'],
            y_train=data['y_pre_full'],
            max_epochs=config['pt_epochs'],
            early_stopping_patience=None,
            batch_size=config['batch_size'],
            weights_file=output[0],
            log_file=output[1],
            device=device,
            shuffle=True)


# Finetune/train the model on observations
rule finetune_train_full:
    input:
        expand("results/baseline/GWN/{rep}/pretrained_weights.pth",
            rep=[f"rep_pre_{i}" for i in replicates]),
        expand("results/baseline/GWN/{rep}/pretrain_log.csv",
            rep=[f"rep_pre_{i}" for i in replicates]),
        "results/{outdir}/GWN/prepped.npz"

    output:
        "results/{outdir}/GWN/full_train/rep_{replicate}/finetuned_weights.pth",
        "results/{outdir}/GWN/full_train/rep_{replicate}/finetune_log.csv"
    threads: 4
    group: 'train'
    run:
        os.system("module load analytics cuda11.3/toolkit/11.3.0")
        os.system("export LD_LIBRARY_PATH=/cm/shared/apps/nvidia/TensorRT-6.0.1.5/lib:/cm/shared/apps/nvidia/cudnn_8.0.5/lib64:$LD_LIBRARY_PATH")

        data = np.load(input[-1])
        data = reshape_for_gwn(data,keep_portion=config['trn_offset'])
        adj_mx = data['dist_matrix']
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        supports = [torch.tensor(adj_mx).to(device).float()]
        in_dim = len(data['x_vars'])
        out_dim = data['y_obs_trn'].shape[3]
        num_nodes = adj_mx.shape[0]
        lrate = config['finetune_learning_rate']
        model = gwnet(device,num_nodes,supports=supports,aptinit=supports[0],
            in_dim=in_dim,out_dim=out_dim,residual_channels=config['hidden_size'],
            dilation_channels=config['hidden_size'],skip_channels=config['hidden_size'] * 4,
            end_channels=config['hidden_size'] * 8,kernel_size=2,blocks=4,layers=4,
            dropout=config['dropout'])

        opt = optim.Adam(model.parameters(),lr=lrate,weight_decay=0.0001)
        model.load_state_dict(torch.load(input[int(wildcards.replicate)]))

        train_torch(model,
            loss_function=rmse_masked,
            optimizer=opt,
            x_train=data['x_trn'],
            y_train=data['y_obs_trn'],
            x_val=data['x_val'],
            y_val=data['y_obs_val'],
            max_epochs=config['ft_epochs'],
            early_stopping_patience=config['early_stopping'],
            batch_size=config['batch_size'],
            weights_file=output[0],
            log_file=output[1],
            device=device,
            shuffle=True)

# Finetune/train the model on observations
rule finetune_train_no_pt:
    input:
        "results/{outdir}/GWN/prepped.npz",

    output:
        "results/{outdir}/GWN/no_pt/rep_{replicate}/finetuned_weights.pth",
        "results/{outdir}/GWN/no_pt/rep_{replicate}/finetune_log.csv"
    threads: 4
    group: 'train'
    run:
        os.system("module load analytics cuda11.3/toolkit/11.3.0")
        os.system("export LD_LIBRARY_PATH=/cm/shared/apps/nvidia/TensorRT-6.0.1.5/lib:/cm/shared/apps/nvidia/cudnn_8.0.5/lib64:$LD_LIBRARY_PATH")

        data = np.load(input[0])
        data = reshape_for_gwn(data,keep_portion=config['trn_offset'])
        adj_mx = data['dist_matrix']
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        supports = [torch.tensor(adj_mx).to(device).float()]
        in_dim = len(data['x_vars'])
        out_dim = data['y_obs_trn'].shape[3]
        num_nodes = adj_mx.shape[0]
        lrate = config['finetune_learning_rate']
        model = gwnet(device,num_nodes,supports=supports,aptinit=supports[0],
            in_dim=in_dim,out_dim=out_dim,residual_channels=config['hidden_size'],
            dilation_channels=config['hidden_size'],skip_channels=config['hidden_size'] * 4,
            end_channels=config['hidden_size'] * 8,kernel_size=2,blocks=4,layers=4,
            dropout=config['dropout'])

        opt = optim.Adam(model.parameters(),lr=lrate,weight_decay=0.0001)

        train_torch(model,
            loss_function=rmse_masked,
            optimizer=opt,
            x_train=data['x_trn'],
            y_train=data['y_obs_trn'],
            x_val=data['x_val'],
            y_val=data['y_obs_val'],
            max_epochs=config['ft_epochs'],
            early_stopping_patience=config['early_stopping'],
            batch_size=config['batch_size'],
            weights_file=output[0],
            log_file=output[1],
            device=device,
            shuffle=True)

# Finetune/train the model on observations
rule finetune_train_reset_adj:
    input:
        expand("results/baseline/GWN/{rep}/pretrained_weights.pth",
            rep=[f"rep_pre_{i}" for i in replicates]),
        expand("results/baseline/GWN/{rep}/pretrain_log.csv",
            rep=[f"rep_pre_{i}" for i in replicates]),
        "results/{outdir}/GWN/prepped.npz"

    output:
        "results/{outdir}/GWN/reset_spatial/rep_{replicate}/finetuned_weights.pth",
        "results/{outdir}/GWN/reset_spatial/rep_{replicate}/finetune_log.csv"
    threads: 4
    group: 'train'
    run:
        os.system("module load analytics cuda11.3/toolkit/11.3.0")
        os.system("export LD_LIBRARY_PATH=/cm/shared/apps/nvidia/TensorRT-6.0.1.5/lib:/cm/shared/apps/nvidia/cudnn_8.0.5/lib64:$LD_LIBRARY_PATH")

        data = np.load(input[-1])
        data = reshape_for_gwn(data,keep_portion=config['trn_offset'])
        adj_mx = data['dist_matrix']
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        supports = [torch.tensor(adj_mx).to(device).float()]
        in_dim = len(data['x_vars'])
        out_dim = data['y_obs_trn'].shape[3]
        num_nodes = adj_mx.shape[0]
        lrate = config['finetune_learning_rate']
        model = gwnet(device,num_nodes,supports=supports,aptinit=supports[0],
            in_dim=in_dim,out_dim=out_dim,residual_channels=config['hidden_size'],
            dilation_channels=config['hidden_size'],skip_channels=config['hidden_size'] * 4,
            end_channels=config['hidden_size'] * 8,kernel_size=2,blocks=4,layers=4,
            dropout=config['dropout'])

        opt = optim.Adam(model.parameters(),lr=lrate,weight_decay=0.0001)
        model.load_state_dict(torch.load(input[int(wildcards.replicate)]))

        ## Re-initialize the adaptive adjacency matrix for training
        n1, n2 = torch.randn(num_nodes,10), torch.randn(10,num_nodes)
        model.nodevec1 = torch.nn.Parameter(n1.to(device),requires_grad=True)
        model.nodevec2 = torch.nn.Parameter(n2.to(device),requires_grad=True)

        train_torch(model,
            loss_function=rmse_masked,
            optimizer=opt,
            x_train=data['x_trn'],
            y_train=data['y_obs_trn'],
            x_val=data['x_val'],
            y_val=data['y_obs_val'],
            max_epochs=config['ft_epochs'],
            early_stopping_patience=config['early_stopping'],
            batch_size=config['batch_size'],
            weights_file=output[0],
            log_file=output[1],
            device=device,
            shuffle=True)

rule make_predictions:
    input:
        "results/{outdir}/GWN/{train_type}/rep_{replicate}/finetuned_weights.pth",
        "results/{outdir}/GWN/prepped.npz"
    output:
        "results/{outdir}/GWN/{train_type}/rep_{replicate}/{partition}_preds.feather",
    group: 'train'
    threads: 3
    run:
        os.system("module load analytics cuda11.3/toolkit/11.3.0")
        os.system("export LD_LIBRARY_PATH=/cm/shared/apps/nvidia/TensorRT-6.0.1.5/lib:/cm/shared/apps/nvidia/cudnn_8.0.5/lib64:$LD_LIBRARY_PATH")

        data = np.load(input[1])
        data = reshape_for_gwn(data,keep_portion=config['trn_offset'])
        adj_mx = data['dist_matrix']
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        supports = [torch.tensor(adj_mx).to(device).float()]
        in_dim = len(data['x_vars'])
        out_dim = data['y_obs_trn'].shape[3]
        num_nodes = adj_mx.shape[0]
        lrate = config['finetune_learning_rate']
        model = gwnet(device,num_nodes,supports=supports,aptinit=supports[0],
            in_dim=in_dim,out_dim=out_dim,residual_channels=config['hidden_size'],
            dilation_channels=config['hidden_size'],skip_channels=config['hidden_size'] * 4,
            end_channels=config['hidden_size'] * 8,kernel_size=2,blocks=4,layers=4,
            dropout=config['dropout'])

        opt = optim.Adam(model.parameters(),lr=lrate,weight_decay=0.0001)
        model.load_state_dict(torch.load(input[0]))

        predict_from_io_data(model=model,
            io_data=data,
            partition=wildcards.partition,
            outfile=output[0],
            trn_offset=config['trn_offset'],
            tst_val_offset=config['trn_offset'])


def get_grp_arg(wildcards):
    if wildcards.metric_type == 'overall':
        return None
    elif wildcards.metric_type == 'month':
        return 'month'
    elif wildcards.metric_type == 'reach':
        return 'seg_id_nat'
    elif wildcards.metric_type == 'month_reach':
        return ['seg_id_nat', 'month']


rule combine_metrics:
    input:
        config['obs_file'],
        "results/{outdir}/GWN/{train_type}/rep_{replicate}/trn_preds.feather",
        "results/{outdir}/GWN/{train_type}/rep_{replicate}/val_preds.feather",
        "results/{outdir}/GWN/{train_type}/rep_{replicate}/tst_preds.feather"
    output:
        "results/{outdir}/GWN/{train_type}/rep_{replicate}/{metric_type}_metrics.csv"
    group: 'train'
    threads: 3
    params:
        grp_arg=get_grp_arg,
        prep_args=get_prep_args
    run:
        combined_metrics(obs_file=input[0],
            pred_trn=input[1],
            pred_val=input[2],
            pred_tst=input[3],
            group=params.grp_arg,
            outfile=output[0],
            test_sites=params.prep_args[3]['test'],
            train_sites=params.prep_args[3]['train'],
        )
