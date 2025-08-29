# -*- coding: utf-8 -*-
"""
Created on Fri Aug 29 13:09:21 2025

@author: rehmer
"""
from pathlib import Path
from hscvatio.hdf5 import CVAT_DataMgr

# %%
project_name = 'test1'


# %% Path to existing hdf5 file
hdf5_path = Path.cwd() / 'data' /  (project_name+'.hdf5')

# %%  Initialize hdf5 manager
data_mgr = CVAT_DataMgr(hdf5_path.as_posix(),mode='a')

# %% Get an overview over the data
hdf5_index = data_mgr.load_index()

# %% Get dataset 17 (the only one in the hdf5 file)
annotated_datasets = data_mgr.get_datasets([17])

