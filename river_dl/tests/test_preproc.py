import pytest
from river_dl import preproc_utils

# segments in test dataset
segs = [2012, 2007, 2014, 2037]
# date range in test dataset
min_date = "2003-09-15"
max_date = "2006-10-15"

obs_flow = "test_data/obs_flow.csv"
obs_temp = "test_data/obs_temp.csv"
sntemp = "test_data/test_data"


def test_read_exclude():
    exclude_file = "test_data/exclude_2007.yml"
    ex0 = preproc_utils.read_exclude_segs_file(exclude_file)
    assert ex0 == [{"seg_id_nats_ex": [2007]}]
    exclude_file = "test_data/exclude1.yml"
    ex1 = preproc_utils.read_exclude_segs_file(exclude_file)
    assert ex1 == [
        {"seg_id_nats_ex": [2007], "start_date": "2005-03-15"},
        {"seg_id_nats_ex": [2012], "end_date": "2005-03-15"},
    ]


def test_prep_data():
    prepped_data = preproc_utils.prep_all_data(
        x_data_file="test_data/test_data",
        y_data_file="test_data/obs_temp_flow",
        train_start_date="2003-09-15",
        train_end_date="2004-09-16",
        val_start_date="2004-09-17",
        val_end_date="2005-09-18",
        test_start_date="2005-09-19",
        test_end_date="2006-09-20",
        spatial_idx_name="segs_test",
        time_idx_name="times_test",
        x_vars=["seg_rain", "seg_tave_air"],
        y_vars_finetune=["temp_c", "discharge_cms"],
    )

    assert "x_trn" in prepped_data.keys()
    assert "x_val" in prepped_data.keys()
    assert "x_tst" in prepped_data.keys()
    assert "ids_trn" in prepped_data.keys()
    assert "ids_val" in prepped_data.keys()
    assert "ids_tst" in prepped_data.keys()
    assert "times_trn" in prepped_data.keys()
    assert "times_val" in prepped_data.keys()
    assert "times_tst" in prepped_data.keys()
    assert "y_obs_trn" in prepped_data.keys()
    assert "y_obs_val" in prepped_data.keys()
    assert "y_obs_tst" in prepped_data.keys()
    assert "y_obs_vars" in prepped_data.keys()
    assert "y_mean" in prepped_data.keys()
    assert "y_std" in prepped_data.keys()
    assert "x_vars" in prepped_data.keys()
    assert "x_mean" in prepped_data.keys()
    assert "x_std" in prepped_data.keys()


def test_prep_data_w_pretrain():
    prepped_data = preproc_utils.prep_all_data(
        x_data_file="test_data/test_data",
        y_data_file="test_data/obs_temp_flow",
        pretrain_file="test_data/test_data",
        train_start_date="2003-09-15",
        train_end_date="2004-09-16",
        val_start_date="2004-09-17",
        val_end_date="2005-09-18",
        test_start_date="2005-09-19",
        test_end_date="2006-09-20",
        spatial_idx_name="segs_test",
        time_idx_name="times_test",
        x_vars=["seg_rain", "seg_tave_air"],
        y_vars_finetune=["temp_c", "discharge_cms"],
        y_vars_pretrain=["seg_tave_water", "seg_outflow"],
    )

    assert "x_trn" in prepped_data.keys()
    assert "x_val" in prepped_data.keys()
    assert "x_tst" in prepped_data.keys()
    assert "ids_trn" in prepped_data.keys()
    assert "ids_val" in prepped_data.keys()
    assert "ids_tst" in prepped_data.keys()
    assert "times_trn" in prepped_data.keys()
    assert "times_val" in prepped_data.keys()
    assert "times_tst" in prepped_data.keys()
    assert "y_pre_trn" in prepped_data.keys()
    assert "y_pre_full" in prepped_data.keys()
    assert "y_obs_trn" in prepped_data.keys()
    assert "y_obs_val" in prepped_data.keys()
    assert "y_obs_tst" in prepped_data.keys()
    assert "y_obs_vars" in prepped_data.keys()
    assert "y_pre_vars" in prepped_data.keys()
    assert "y_mean" in prepped_data.keys()
    assert "y_std" in prepped_data.keys()
    assert "x_vars" in prepped_data.keys()
    assert "x_mean" in prepped_data.keys()
    assert "x_std" in prepped_data.keys()


def test_prep_data_w_pretrain_file_no_y_pretrain():
    with pytest.raises(ValueError):
        preproc_utils.prep_all_data(
            x_data_file="test_data/test_data",
            y_data_file="test_data/obs_temp_flow",
            pretrain_file="test_data/obs_temp_flow",
            train_start_date="2003-09-15",
            train_end_date="2004-09-16",
            val_start_date="2004-09-17",
            val_end_date="2005-09-18",
            test_start_date="2005-09-19",
            test_end_date="2006-09-20",
            spatial_idx_name="segs_test",
            time_idx_name="times_test",
            x_vars=["seg_rain", "seg_tave_air"],
            y_vars_finetune=["temp_c", "discharge_cms"],
        )


def test_prep_data_no_validation_dates():
    prepped_data = preproc_utils.prep_all_data(
            x_data_file="test_data/test_data",
            y_data_file="test_data/obs_temp_flow",
            pretrain_file="test_data/test_data",
            train_start_date="2003-09-15",
            train_end_date="2004-09-16",
            test_start_date="2005-09-19",
            test_end_date="2006-09-20",
            spatial_idx_name="segs_test",
            time_idx_name="times_test",
            x_vars=["seg_rain", "seg_tave_air"],
            y_vars_finetune=["temp_c", "discharge_cms"],
            y_vars_pretrain=["seg_tave_water", "seg_outflow"],
        )

    assert "x_trn" in prepped_data.keys()
    assert "x_val" in prepped_data.keys()
    assert "x_tst" in prepped_data.keys()
    assert "ids_trn" in prepped_data.keys()
    assert "ids_val" in prepped_data.keys()
    assert "ids_tst" in prepped_data.keys()
    assert "times_trn" in prepped_data.keys()
    assert "times_val" in prepped_data.keys()
    assert "times_tst" in prepped_data.keys()
    assert "y_pre_trn" in prepped_data.keys()
    assert "y_pre_full" in prepped_data.keys()
    assert "y_obs_trn" in prepped_data.keys()
    assert "y_obs_val" in prepped_data.keys()
    assert "y_obs_tst" in prepped_data.keys()

    assert prepped_data["x_trn"] is not None
    assert prepped_data["x_val"] is None
    assert prepped_data["x_tst"] is not None
    assert prepped_data["ids_trn"] is not None
    assert prepped_data["ids_val"] is None
    assert prepped_data["ids_tst"] is not None
    assert prepped_data["times_trn"] is not None
    assert prepped_data["times_val"] is None
    assert prepped_data["times_tst"] is not None
    assert prepped_data["y_pre_trn"] is not None
    assert prepped_data["y_obs_trn"] is not None
    assert prepped_data["y_obs_val"] is None
    assert prepped_data["y_obs_tst"] is not None

def test_prep_data_no_test_dates():
    prepped_data = preproc_utils.prep_all_data(
            x_data_file="test_data/test_data",
            y_data_file="test_data/obs_temp_flow",
            pretrain_file="test_data/test_data",
            train_start_date="2003-09-15",
            train_end_date="2004-09-16",
            val_start_date="2004-09-17",
            val_end_date="2005-09-18",
            spatial_idx_name="segs_test",
            time_idx_name="times_test",
            x_vars=["seg_rain", "seg_tave_air"],
            y_vars_finetune=["temp_c", "discharge_cms"],
            y_vars_pretrain=["seg_tave_water", "seg_outflow"],
        )

    assert "x_trn" in prepped_data.keys()
    assert "x_val" in prepped_data.keys()
    assert "x_tst" in prepped_data.keys()
    assert "ids_trn" in prepped_data.keys()
    assert "ids_val" in prepped_data.keys()
    assert "ids_tst" in prepped_data.keys()
    assert "times_trn" in prepped_data.keys()
    assert "times_val" in prepped_data.keys()
    assert "times_tst" in prepped_data.keys()
    assert "y_pre_trn" in prepped_data.keys()
    assert "y_pre_full" in prepped_data.keys()
    assert "y_obs_trn" in prepped_data.keys()
    assert "y_obs_val" in prepped_data.keys()
    assert "y_obs_tst" in prepped_data.keys()

    assert prepped_data["x_trn"] is not None
    assert prepped_data["x_val"] is not None
    assert prepped_data["x_tst"] is None
    assert prepped_data["ids_trn"] is not None
    assert prepped_data["ids_val"] is not None
    assert prepped_data["ids_tst"] is None
    assert prepped_data["times_trn"] is not None
    assert prepped_data["times_val"] is not None
    assert prepped_data["times_tst"] is None
    assert prepped_data["y_pre_trn"] is not None
    assert prepped_data["y_obs_trn"] is not None
    assert prepped_data["y_obs_val"] is not None
    assert prepped_data["y_obs_tst"] is None

def test_prep_data_no_val_end():
    with pytest.raises(ValueError):
        prepped_data = preproc_utils.prep_all_data(
                x_data_file="test_data/test_data",
                y_data_file="test_data/obs_temp_flow",
                pretrain_file="test_data/test_data",
                train_start_date="2003-09-15",
                train_end_date="2004-09-16",
                val_start_date="2004-09-17",
                spatial_idx_name="segs_test",
                time_idx_name="times_test",
                x_vars=["seg_rain", "seg_tave_air"],
                y_vars_finetune=["temp_c", "discharge_cms"],
                y_vars_pretrain=["seg_tave_water", "seg_outflow"],
            )

def test_prep_data_no_val_start():
    with pytest.raises(ValueError):
        prepped_data = preproc_utils.prep_all_data(
                x_data_file="test_data/test_data",
                y_data_file="test_data/obs_temp_flow",
                pretrain_file="test_data/test_data",
                train_start_date="2003-09-15",
                train_end_date="2004-09-16",
                val_end_date="2004-09-17",
                spatial_idx_name="segs_test",
                time_idx_name="times_test",
                x_vars=["seg_rain", "seg_tave_air"],
                y_vars_finetune=["temp_c", "discharge_cms"],
            )

def test_prep_data_no_test_end():
    with pytest.raises(ValueError):
        prepped_data = preproc_utils.prep_all_data(
                x_data_file="test_data/test_data",
                y_data_file="test_data/obs_temp_flow",
                pretrain_file="test_data/test_data",
                train_start_date="2003-09-15",
                train_end_date="2004-09-16",
                test_start_date="2004-09-17",
                spatial_idx_name="segs_test",
                time_idx_name="times_test",
                x_vars=["seg_rain", "seg_tave_air"],
                y_vars_finetune=["temp_c", "discharge_cms"],
            )

def test_prep_data_no_test_start():
    with pytest.raises(ValueError):
        prepped_data = preproc_utils.prep_all_data(
                x_data_file="test_data/test_data",
                y_data_file="test_data/obs_temp_flow",
                pretrain_file="test_data/test_data",
                train_start_date="2003-09-15",
                train_end_date="2004-09-16",
                test_end_date="2004-09-17",
                spatial_idx_name="segs_test",
                time_idx_name="times_test",
                x_vars=["seg_rain", "seg_tave_air"],
                y_vars_finetune=["temp_c", "discharge_cms"],
                y_vars_pretrain=["seg_tave_water", "seg_outflow"],
            )

def test_prep_data_no_train():
    with pytest.raises(TypeError):
        prepped_data = preproc_utils.prep_all_data(
                x_data_file="test_data/test_data",
                y_data_file="test_data/obs_temp_flow",
                pretrain_file="test_data/test_data",
                test_start_date="2003-09-15",
                test_end_date="2004-09-16",
                val_start_date="2003-09-15",
                val_end_date="2004-09-16",
                spatial_idx_name="segs_test",
                time_idx_name="times_test",
                x_vars=["seg_rain", "seg_tave_air"],
                y_vars_finetune=["temp_c", "discharge_cms"],
            )


def test_prep_data_val_test_sites():
    prepped_data = preproc_utils.prep_all_data(
            x_data_file="test_data/test_data",
            y_data_file="test_data/obs_temp_flow",
            pretrain_file="test_data/test_data",
            train_start_date="2003-09-15",
            train_end_date="2004-09-16",
            val_start_date="2004-09-17",
            val_end_date="2005-09-18",
            val_sites=[2007],
            test_sites=[2012],
            spatial_idx_name="segs_test",
            time_idx_name="times_test",
            x_vars=["seg_rain", "seg_tave_air"],
            y_vars_finetune=["temp_c", "discharge_cms"],
            y_vars_pretrain=["seg_tave_water", "seg_outflow"],
        )

    assert "x_trn" in prepped_data.keys()
    assert "x_val" in prepped_data.keys()
    assert "x_tst" in prepped_data.keys()
    assert "ids_trn" in prepped_data.keys()
    assert "ids_val" in prepped_data.keys()
    assert "ids_tst" in prepped_data.keys()
    assert "times_trn" in prepped_data.keys()
    assert "times_val" in prepped_data.keys()
    assert "times_tst" in prepped_data.keys()
    assert "y_pre_trn" in prepped_data.keys()
    assert "y_pre_full" in prepped_data.keys()
    assert "y_obs_trn" in prepped_data.keys()
    assert "y_obs_val" in prepped_data.keys()
    assert "y_obs_tst" in prepped_data.keys()

    assert 2007 not in prepped_data["ids_trn"] 
    assert 2012 not in prepped_data["ids_trn"] 
    assert 2014 in prepped_data["ids_trn"] 
    assert 2037 in prepped_data["ids_trn"] 

    assert 2007 in prepped_data["ids_val"] 
    assert 2014 in prepped_data["ids_val"] 
    assert 2037 in prepped_data["ids_val"] 
    assert 2012 not in prepped_data["ids_val"] 

    assert prepped_data["ids_tst"] is None


def test_prep_data_val_test_sites_test_dates():
    prepped_data = preproc_utils.prep_all_data(
            x_data_file="test_data/test_data",
            y_data_file="test_data/obs_temp_flow",
            pretrain_file="test_data/test_data",
            train_start_date="2003-09-15",
            train_end_date="2004-09-16",
            val_start_date="2004-09-17",
            val_end_date="2005-09-18",
            test_start_date="2005-09-19",
            test_end_date="2006-09-20",
            val_sites=[2007],
            test_sites=[2012],
            spatial_idx_name="segs_test",
            time_idx_name="times_test",
            x_vars=["seg_rain", "seg_tave_air"],
            y_vars_finetune=["temp_c", "discharge_cms"],
            y_vars_pretrain=["seg_tave_water", "seg_outflow"],
        )

    assert "x_trn" in prepped_data.keys()
    assert "x_val" in prepped_data.keys()
    assert "x_tst" in prepped_data.keys()
    assert "ids_trn" in prepped_data.keys()
    assert "ids_val" in prepped_data.keys()
    assert "ids_tst" in prepped_data.keys()
    assert "times_trn" in prepped_data.keys()
    assert "times_val" in prepped_data.keys()
    assert "times_tst" in prepped_data.keys()
    assert "y_pre_trn" in prepped_data.keys()
    assert "y_pre_full" in prepped_data.keys()
    assert "y_obs_trn" in prepped_data.keys()
    assert "y_obs_val" in prepped_data.keys()
    assert "y_obs_tst" in prepped_data.keys()

    assert 2007 not in prepped_data["ids_trn"] 
    assert 2012 not in prepped_data["ids_trn"] 
    assert 2014 in prepped_data["ids_trn"] 
    assert 2037 in prepped_data["ids_trn"] 

    assert 2007 in prepped_data["ids_val"] 
    assert 2014 in prepped_data["ids_val"] 
    assert 2037 in prepped_data["ids_val"] 
    assert 2012 not in prepped_data["ids_val"] 

    assert 2012 in prepped_data["ids_tst"]
    assert 2007 in prepped_data["ids_tst"] 
    assert 2014 in prepped_data["ids_tst"] 
    assert 2037 in prepped_data["ids_tst"] 


def test_prep_data_no_val_tst_sites_test_dates():
    prepped_data = preproc_utils.prep_all_data(
            x_data_file="test_data/test_data",
            y_data_file="test_data/obs_temp_flow",
            pretrain_file="test_data/test_data",
            train_start_date="2003-09-15",
            train_end_date="2004-09-16",
            val_start_date="2004-09-17",
            val_end_date="2005-09-18",
            test_start_date="2005-09-19",
            test_end_date="2006-09-20",
            spatial_idx_name="segs_test",
            time_idx_name="times_test",
            x_vars=["seg_rain", "seg_tave_air"],
            y_vars_finetune=["temp_c", "discharge_cms"],
            y_vars_pretrain=["seg_tave_water", "seg_outflow"],
        )

    assert "x_trn" in prepped_data.keys()
    assert "x_val" in prepped_data.keys()
    assert "x_tst" in prepped_data.keys()
    assert "ids_trn" in prepped_data.keys()
    assert "ids_val" in prepped_data.keys()
    assert "ids_tst" in prepped_data.keys()
    assert "times_trn" in prepped_data.keys()
    assert "times_val" in prepped_data.keys()
    assert "times_tst" in prepped_data.keys()
    assert "y_pre_trn" in prepped_data.keys()
    assert "y_pre_full" in prepped_data.keys()
    assert "y_obs_trn" in prepped_data.keys()
    assert "y_obs_val" in prepped_data.keys()
    assert "y_obs_tst" in prepped_data.keys()

    assert 2007 in prepped_data["ids_trn"] 
    assert 2012 in prepped_data["ids_trn"] 
    assert 2014 in prepped_data["ids_trn"] 
    assert 2037 in prepped_data["ids_trn"] 

    assert 2007 in prepped_data["ids_val"] 
    assert 2014 in prepped_data["ids_val"] 
    assert 2037 in prepped_data["ids_val"] 
    assert 2012 in prepped_data["ids_val"] 

    assert 2012 in prepped_data["ids_tst"]
    assert 2007 in prepped_data["ids_tst"] 
    assert 2014 in prepped_data["ids_tst"] 
    assert 2037 in prepped_data["ids_tst"] 

