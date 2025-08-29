# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 10:50:25 2023

@author: Rehmer
"""

from pathlib import Path
from hscvatio.hdf5 import CVAT_DataMgr


#%% CVAT login data
username = 'your_username'
password = 'your_password'

credentials = {'user':username,
               'pw':password}

# %% Name of the cvat project from which to download annotations (must exist)
project_name = 'test1'

# %% Specify where the data that should be imported is located
hdf5_path = Path.cwd() / 'data' /  (project_name+'.hdf5')

# %%  Initialize hdf5 manager
data_mgr = CVAT_DataMgr(hdf5_path.as_posix(),mode='a')


# %% Provide CVAT credentials to the CVAT_DataMgr
data_mgr.cvat_credentials = credentials

# %% Sync hdf5 with cvat server
data_mgr.sync_with_cvat(project_name)
hdf5_index = data_mgr.load_index()
    
# %% Find existing annotations Download annotations
download_idx = [i for i in hdf5_index.index if \
                (hdf5_index.loc[i,'stage'] == 'validation') |\
                (hdf5_index.loc[i,'stage'] == 'acceptance')   ]


# %% Download annotations
for i in download_idx:
    
    data_mgr.cvat_download_annotation([i],mode='w')
    
    
