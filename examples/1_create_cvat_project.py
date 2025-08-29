# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 10:50:25 2023

@author: Rehmer
"""

from pathlib import Path
import numpy as np

from hscvatio.hdf5 import CVAT_DataMgr

from hspytools.cv import filters
from hspytools.tparray import TPArray
from hspytools.tparray import SensorTypes

#%% CVAT login data
username = 'your_username'
password = 'your_password'

credentials = {'user':username,
               'pw':password}


# %% Name of the cvat project that is to be created (should not exist in advance)
project_name = 'test1'

# %% Directory containing the .bds or .txt files containing htpa data
video_path = Path.cwd() / 'data/'

# %% Path to the .csv file containing the metadata corresponding to the htpa data
meta_path = Path.cwd() /  'data/meta.csv'

# %% The htpa data, metadata and later annotations will all be written to 
# a .hdf5 file. Specify the path to that file here (should not exist in advance)
hdf5_path = Path.cwd() / 'data' / (project_name+'.hdf5')
    
# %%  Initialize an empty container for the video sequences
data_mgr = CVAT_DataMgr(hdf5_path.as_posix(),mode='a')

# %% Add the recorded data to that container. .bds and .txt files will be 
# automatically converted to pandas dataframes before stored in the hdf5 file
data_mgr.import_video_sequences(video_path,meta_path)

# %% The htpa videos have now been imported to the specified hdf5 file.
# To get an overwiew over the data in the hdf5 file one can load the so-called
# index:
hdf5_file_index = data_mgr.load_index()
print(hdf5_file_index)

# %% One can also load any imported video via the load_video() method, which 
# which takes the index of the video to load as an argument. The return value
# is a pandas dataframe
sequence_1 = data_mgr.load_video(17)

#%% To adress a specific frame in sequence_1 use the built-in pandas method loc
# and in order to adress specific columns (containing pixels) use the TPArray
# class' _pix attribute:

idx = 1110
pix_cols = TPArray(width=60,height=40)._pix

import matplotlib.pyplot as plt
plt.imshow(sequence_1.loc[idx,pix_cols].values.reshape((40,60)))

# %% Provide CVAT credentials to the CVAT_DataMgr
data_mgr.cvat_credentials = credentials

# %% Push videos as tasks to the CVAT server
data_mgr = CVAT_DataMgr(hdf5_path.as_posix(),mode='a')

bds_index = data_mgr.load_index() 

for i in bds_index.index:

    data_mgr.cvat_upload_task([i],              
                              project_name,
                              fps=4,            
                              k=2)




