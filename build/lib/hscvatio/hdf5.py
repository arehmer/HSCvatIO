# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 16:06:32 2023

@author: Rehmer
"""
import warnings
from pathlib import Path
import pandas as pd
import numpy as np
# import pickle as pkl
import matplotlib
import matplotlib.pyplot as plt
import h5py

import matplotlib as mpl
mpl.rc('image', cmap='plasma')

from hspytools.hdf5_mgr import hdf5_mgr
from hspytools.tparray import TPArray
from hspytools.readers import HTPAdGUI_FileReader

from hscvatio.sdk import CVATTaskMgr, CVATAutoAnnot

# from hsod.cv.detectors import BlobDetector
# from hsod.cv.tracktors import Tracktor
# from hsod.cv.preproc_engines import PreprocEngAWA_CV
 

def custom_formatwarning(msg, category, filename, lineno, line=None):
    return f"{category.__name__}: {msg}\n"

warnings.formatwarning = custom_formatwarning

class CVAT_DataMgr(hdf5_mgr):
    """
    Class with methods for retrieving annotated objects from an image sequence 
    based on a JSON file with COCO formatted annotations
    """
    def __init__(self,hdf5_path,**kwargs):
        
        
        kwargs['hdf5Index_dtypes'] = {'BDSname':'str', 'Width':'uint16', 'Height':'uint16',
                                      'RecDate':'str', 'ArrayType':'uint16', 'SensorType':'str',
                                      'SensorID':'uint32', 'rotation':'uint16',
                                      'angle':'uint16','min_dist':'float64', 'max_dist':'float64',
                                      'controller':'str', 'exp_name':'str', 'exp_cat1':'str',
                                      'exp_cat2':'str','exp_cat3':'str', 'exp_cat4':'str',
                                      'exp_cat5':'str','Remarks':'str', 'bdslength':'uint32',
                                      'address':'str','cvat_task':'float64', 'cvat_fps':'float64',
                                      'stage':'str'}
        
        super(CVAT_DataMgr, self).__init__(hdf5_path, **kwargs)
        
        # self.hdf5Index_dtypes = 
        
        
        
    @property
    def cvat_credentials(self):
        # user = str(self._cvat_credentials['user'])
        # pw = str(self._cvat_credentials['pw'])
        # return(user,pw)
        return self._cvat_credentials
    
    @cvat_credentials.setter
    def cvat_credentials(self,cred:dict):
        if not isinstance(cred,dict):
            raise TypeError('Credentials must be provided as a dict with keys'+\
                            '"user" and "pw"!')
                            
        # When read from the hdf5 file, strings are bytestrings. Convert
        for key in cred.keys():
            if isinstance(cred[key],str):
                pass
            elif isinstance(cred[key],bytes):
                cred[key] = cred[key].decode('utf-8')
            else:
                raise TypeError(f'{key} must be provided as str or  '+\
                                'bytes but is provided as f{type(dtypes[key])}')
         
        self._cvat_credentials = cred
        
        self._save_attributes_to_hdf5()        
            
        
    def import_video_sequences(self,video_path,path_meta,**kwargs):
        """
        Function for adding a video sequence to the container by specifying 
        the path to the file containing the sequence. The added sequence can 
        then be found in the hdf5-file under videos/<name>/data.
        
        Parameters
        ----------
        path : str
            path to the txt-file, bds-file or pickled DataFrame containing
            the video sequence  
        name : str
            Name that the video sequence should be given in the container
        size : tuple
            A tuple specifying the width and height of the frame in the format
            (width,height)
        mode: char
            A character specifying what should be done if a video with that name 
            already exists in the hdf5 file. 
            'w': overwrite
            
        Returns
        -------
        None.
        """
        
        mode = kwargs.pop('mode','a')
        
        # Check if video_path is given as pathlib path
        if not isinstance(video_path,Path):
            print('video_path must be given as pathlib.Path object')
            return None
        
        # Check if video_path leads to a directory or a file
        if video_path.is_file():
            files = [video_path]
        elif video_path.is_dir():
            files = [file for file in video_path.iterdir()]
        
        # Load the index of the file container
        df_index = self.load_index()
        

        # Load the csv containing the metadata
        meta = pd.read_csv(path_meta)
        
        # Make INDX the index
        meta =  meta.set_index('INDX')
        
        # Replace '(null)'
        meta = meta.replace('(null)',None)
        
        # Remove the file ending .rar in meta
        for i in meta.index:
            bds_name = Path(meta.loc[i,'BDSname'])
            meta.loc[i,'BDSname'] = bds_name.stem
                
        # Go through all files in a loop and write them to the hdf5 file
        for file in files:
            
            # Console output
            print('Importing ' + file.name)
            
            # Check if file is .bds or .text, otherwise ignore and move on
            if not file.suffix.casefold() == '.txt' and \
                not file.suffix.casefold() == '.bds':
                    print(file.name + ' is neither .bds or .txt. Ignore.')
                    continue
            
            # Dict for fields to write
            data_to_write = {}
                    
            # Name in hdf5-group is video + the current unique id
            # video_name = 'video'+str(unique_id)
            video_name = file.stem
            
            # Check if video already exists
            with h5py.File(self._hdf5_path,'a') as hdf5_file:
                
                exist = False
                if 'videos/'+video_name in hdf5_file:
                    exist = True
                    print('Video ' + video_name + ' already exists.\n')
            
            
            if exist==True and mode == 'w':
                print('Data is deleted and rewritten.\n')
                self.delete_video('videos/'+video_name)
            elif exist==True and mode != 'w':
                print('No data is written to group. Pass mode="w" to overwrite existing data.')
                continue
            
            # Get all the metadata for that file from the metadata DataFrame
            df_row = meta.loc[meta['BDSname']==file.stem].copy()
            
            # if none or more than one bds were found, something went wrong
            if len(df_row)==0:
                print('No .bds found for ' + file.stem + '.')
                print('BDS not imported.')
                continue
            elif len(df_row)>1:
                print('More than one .bds found for ' + file.stem + '.')
                print('BDS not imported.')
                continue
            
            # Load video from file
            size = (int(df_row.iloc[0]['Width']),int(df_row.iloc[0]['Height']))
            
            FileReader = HTPAdGUI_FileReader(TPArray(width = size[0],
                                                     height = size[1]))
            
            df_video,_ = FileReader.read_htpa_video(file)
                        
            
            # Add the address of the new group to the index
            df_row['address'] = 'videos/'+video_name
            
            # Append metadata to the index
            df_index = self.load_index()
            df_index = pd.concat([df_index,df_row])
            
            # Write to dictionary
            data_to_write['videos/'+video_name + '/data'] = df_video  
            data_to_write['index'] = df_index
            
            # Write to file
            try:
                self._write_fields(data_to_write)
                print(file.name + ' successfully imported.')
            except:
                print('Some error occured when writing ' + file.name + '.' )

        return None
              
    def delete_video(self,bds_idx):
        
        # load bds index
        bds_index = self.load_index()
        
        # Name of video
        bds_name = bds_index.loc[bds_idx,'BDSname']
        
        # Find address of video to be deleteddelete_video
        bds_address = bds_index.loc[bds_idx,'address']
        
        # Delete the video and its entry in bds_index 
        with h5py.File(self._hdf5_path, 'a') as hdf5_file:
            try:
                del hdf5_file[bds_address]
                bds_index = bds_index.drop(index = bds_idx)

            except:
                print(bds_name + " couldn't be deleted or doesn't exist anymore")
                bds_index = bds_index.drop(index = bds_idx)     
        
        # write the new bds index to the file
        self._write_fields({'index':bds_index})
        
        # load json index
        json_index = self.load_JSONindex()
        
        # Check if a annotation exists for the video
        if bds_idx in json_index.index:
            print('Annotations for ' + bds_name + ' found.')
            self.delete_annotation(bds_idx)
        
    def delete_annotation(self,json_idx):
        
        # load JSON index
        json_index = self.load_JSONindex()
        
        # Name of annotation
        json_name = json_index.loc[json_idx,'JSONname']
        
        # Find row with this address and drop it
        json_address = json_index.loc[json_idx,'json_address']
               

        # delete the group from the file
        with h5py.File(self._hdf5_path, 'a') as hdf5_file:
            try:
                del hdf5_file[json_address]
                json_index = json_index.drop(index = json_idx)
            except:
                print(json_name + " couldn't be deleted or doesn't exist anymore.")
            
        # write the new json index to the file
        self._write_fields({'JSONindex':json_index})
        
    def get_videos(self):
        # open file
        hdf5_file = h5py.File(self._hdf5_path, 'r')  
        
        video_names = list(hdf5_file['videos'].keys())
        
        return video_names
    
    # def filter_bds(self,filt:PreprocEngAWA_CV,bds_idx,**kwargs):
        
        
    #     mode = kwargs.pop('mode','a')
        
    #     # Load index
    #     bds_index = self.load_index()
        
    #     # Create address for filered image 
    #     bds_address = bds_index.loc[bds_idx,'address'].item()
        
    #     # Check if filtered video already exists
    #     with h5py.File(self._hdf5_path,  'r') as hdf5_file:
            
    #         exist = False
            
    #         if bds_address+'/filt' in hdf5_file:
    #             exist = True
    #             print('Filtered video already exists.\n')
                
    #     if exist==True and mode == 'w':
    #         print('Data is deleted and rewritten.\n')
    #         self.delete_video(bds_address+'/filt')
    #     elif exist==True and mode != 'w':
    #         print('No data is written to group. Pass mode="w" \n to overwrite existing data.')
    #         return None

    #     # Reload index in case it was changed through deletion
    #     bds_index = self.load_index()
    #     bds_row = bds_index.loc[bds_idx]
    #     bds_name = bds_row['BDSname'].item()

    #     # Dict for fields to write
    #     data_to_write = {}
        
    #     # Load specified video sequence
    #     video = self.load_video(bds_row.index)

    #     # Get shape
    #     w = bds_row['Width'].item()
    #     h = bds_row['Height'].item()
        
    #     # Keep only the columns with pixels
    #     video = video[TPArray(w, h)._pix]
        
    #     # Filter data 
    #     print('Filtering ' + bds_name)
    #     filt_video = filt.process(video = video)     

    #     # Add filtered sequence to data_to_write
    #     data_to_write[bds_address + '/filt/data'] = filt_video  
        
    #     # Update index
    #     filt_row = bds_row.copy()
    #     # Change address
    #     filt_row['address'] = bds_address + '/filt'
    #     # Adapt BDSname
    #     filt_row['BDSname'] =  Path(filt_row['BDSname'].item()).stem \
    #         + '_filt'
        
    #     # Get a new unique index
    #     filt_row.index._data = self._unique_INDX(df = bds_index)
        
    #     # Append row
    #     bds_index = pd.concat([bds_index,filt_row])
        
    #     # Add the updated index with the downscaled images to data_to_write
    #     data_to_write['index'] = bds_index
        
    #     # If annotations exist, make an entry into json_index that points to
    #     # the filtered video. Changing the annotation itself is not necessary
    #     # as the filter doesn't change the video sequence (no time delay)

    #     # Load json index
    #     json_index = self.load_JSONindex()
        
    #     # Find all annotations pointing to opriginal bds
    #     annots = json_index.loc[json_index['BDSname']==bds_name].copy()
        
    #     # Copy rows, change BDSname and address to filtered video and append
    #     annots['BDSname'] =  filt_row['BDSname'].item()
    #     annots['bds_address'] = filt_row['address'].item()
        
    #     # get a new unique index for each annotation
    #     for a_idx in annots.index:
            
    #         annot_row = annots.loc[[a_idx]].copy()
            
    #         # Get a new unique index for that row
    #         annot_row.index._data = self._unique_INDX(df = json_index)
            
    #         # write updated json_index to file
    #         json_index = pd.concat([json_index,annot_row])
        
    #     # Add annotations to data_to_write
    #     data_to_write['JSONindex'] = json_index  
        
    #     try:
    #         # Write dataframes to specified address
    #         self._write_fields(data_to_write)
    #         print(bds_name + ' successfully filtered.')

    #     except:
    #         print('Error occured when filtering ' + bds_name + '.' )        

    #     return None
    
    def get_datasets(self,bds_idx:pd.Index,**kwargs):
        """
        Given the index of annotations, this function loads the annotations
        as well as the corresponding videos and returns them in a list
        """
                       
        # Load bds index and json index
        bds_index = self.load_index()
        json_index = self.load_JSONindex()
        
        # Initialize a dictionary for storing the videos and corresponding
        # annotations in. 
        datasets = {}
        
        for b_idx in bds_idx:
            
            # Load the video
            video = self.load_video(b_idx)
            
            # # Load the CVAT task number
            cvat_task = int(bds_index.loc[b_idx,'cvat_task'])
                        
            # Load the annotation
            annot = self.load_annotation(b_idx)
            
                   
            # Add to dict
            datasets.update({cvat_task:{'video':video,'annot':annot}})
        
        # return
        return datasets
   
    def load_JSONindex(self):
        
        # Check if json index already exists
        with h5py.File(self._hdf5_path,'a') as hdf5_file:
            if 'JSONindex' in hdf5_file:
                exist = True
            else:
                exist = False
                
        if exist:
            json_index = self.load_df('JSONindex')
        else:
            columns=['BDSname','JSONname','json_address','bds_address']
            json_index = pd.DataFrame(data=[],
                                      columns=columns)
            
        return json_index
    
    def _flatten_dict(self,nested_dict):    
        """
        Takes a nested dictionary (dict in dict) and returns a flattened
        dictionary instead
        

        Parameters
        ----------
        nested_dict : dict
            Nested dictionary.

        Returns
        -------
        dict.
        """
        
        res = {}
        
        if isinstance(nested_dict, dict):
            keys = list(nested_dict.keys())
            for k in keys:
                flattened_dict = self._flatten_dict(nested_dict[k])
                
                if isinstance(flattened_dict,dict):
                    for key, val in flattened_dict.items():
                        # key = list(key)
                        # key.insert(0, k)
                        # key = key[-1]
                        res[key] = val
                 
                else:
                    res[k] = nested_dict[k]
                
        else:
            res = None
            
        return res

    def _unique_INDX(self,df,**kwargs):
        """
        Create a unique index for new entries in the bds index
        """
        
        
        # Initialize
        unique_idx = kwargs.pop('idx',None)
        
        # If an existing index was provided, check if its unique already
        if unique_idx is not None:
            if unique_idx.item() not in df.index:
                return unique_idx._data
        
        # If no index was provided, generate an insanely high index and 
        # decrement until a new one has been found
        unique_idx = df.index[0:1].copy()
        unique_idx._data = np.array([9999], dtype=df.index.dtype)
        
        # Decrement the index until a new index has been found
        unique = False
        
        while unique == False:
            
            if unique_idx.item() not in df.index:
                unique = True
            else:
                unique_idx = unique_idx - 1
   
        return unique_idx._data
      
    def _idx_to_int(self,idx):
        
        # Convert to list
        if isinstance(idx,pd.Index):
            idx = list(idx)
        elif  isinstance(idx,int):
            return idx
        
        # Check length
        if len(idx)>1:
            ValueError('Passed index must have length 1!')
            return None
        
        if not isinstance(idx[0],int):
            TypeError('Passed index must be an integer')
            return None
        
        return idx[0]
                

    def load_annotation(self,
                        idx:int):
        
        # Convert index to int
        idx = self._idx_to_int(idx)
        
        json_index = self.load_JSONindex()
        
        if idx in json_index.index:
            address = json_index.loc[idx,'json_address']        
            return pd.read_hdf(self._hdf5_path, address + '/data')
        else:
            bds_index = self.load_index()
            task_name = bds_index.loc[idx,'BDSname']
            warnings.warn('No annotation exists for task ' + task_name +'! ' + \
                          'Returning empty DataFrame instead.\n')
            return pd.DataFrame(data=[])
    
    def load_video(self,
                   idx:int,
                   **kwargs):
        
        # Convert index to int
        idx = self._idx_to_int(idx)
            
        # Load index
        bds_index = self.load_index()
        
        address = bds_index.loc[idx,'address'] + '/data'

        # Load and return specified video sequence
        return pd.read_hdf(self._hdf5_path, address)
    
    def load_size(self,video_name,**kwargs):
        
        appendix = kwargs.pop('appendix','')
        
        address = 'videos/' + video_name 
        
        size = pd.read_hdf(self._hdf5_path, address + '/size' + appendix)
        
        w = int(size.loc[0,'width'])
        h = int(size.loc[0,'height'])
        
        return (w,h)
        

    def write_to_mp4(self,idx,mp4_path,**kwargs):
        """
        Writes a video sequence already stored in the hdf5-file under the name
        video_name to an mp4-file. Video will be saved in the folder specified 
        in folder_path under the file name <video_name>.mp4
        
        Parameters
        ----------
        video_name : str
            name under which the video is stored in the hdf5-file. If a 
            postprocessed version of the video is to be addressed, then pass
            'video_name/filtered' for example, or 'video_name/gradient'
        folder_path : str
            Path to the folder where the video should be saved. 
  
        Returns
        -------
        None.
        """
        
        print('''For annotation purposes it is highly recommended to export the 
              video sequence as png-images via write_to_png(). mp4 performs
              interpolations between frames that to not reflect the original
              sensor image and can deaviate from it.''')
        

        # Convert index to int
        idx = self._idx_to_int(idx)

        # Load specified video sequence
        df_video = self.load_video(idx)
        
        # Load index
        bds_index = self.load_index()
        
        # Initialize TPArray object
        tparray = TPArray(width = bds_index.loc[idx,'Width'],
                          height = bds_index.loc[idx,'Height'])
               
        # Initialize HTPAGUIReader object
        htpa_reader = HTPAdGUI_FileReader(tparray.width,
                                          tparray.height)
        
        htpa_reader.export_mp4(df_video, mp4_path)
        
        
        return None
    
    
    def cvat_update(self):
        """
        In case hdf5 got ever deleted, calling this function will reinstate
        the connection between hdf5-file and CVAT project.
        By looping through all tasks on the CVAT server in the specified project
        this function will bring together the DataFrame in the hdf5 file and 
        the task on the server via the unique file name 

        Returns
        -------
        None.

        """
        
        print("Function not yet implemented")
        
        return None
    
    def cvat_upload_task(self,idx,cvat_project,fps=4,k=2,**kwargs):
        """
        

        Parameters
        ----------
        idx : TYPE
            DESCRIPTION.
        cvat_project : TYPE
            DESCRIPTION.
        fps : float, optional
            The sensor data is downsampled to the specified sampling rate in
            frames per seconds (fps). The default is 4.
        k : int, optional
            In order to provide the labeller with more context information, the
            downsampled sensor data can be upsampled by a factor of k. 
            The default is 2.

        Returns
        -------
        None.

        """
        
        # Load index
        bds_index = self.load_index()
        
        # Load the video
        video = self.load_video(idx)
        
        # Get bds name of video and use it as task name
        task_name = bds_index.loc[idx,'BDSname'].item()
        
        # Initialize the CVATTaskMgr
        user = self.cvat_credentials['user']
        pw = self.cvat_credentials['pw']
        task_mgr = CVATTaskMgr(credentials = (user,pw))
        
        try:
                        
            # Set project attribute of CVATTaskMgr by passing project name
            task_mgr.project = cvat_project
            
            # If the project does not already exist, create it
            if task_mgr.project is None:
                task_mgr.create_cvat_project(cvat_project)      
            
        except Exception as e:
            print(f"An error occurred when creating task '{task_name}': {e}")
            
            return None
        
        # Check if the task to be uploaded already has a cvat number,
        # if so, check if this task already exists on the the CVAT server
        if 'cvat_task' in bds_index.columns:
            if not np.isnan(bds_index.loc[idx,'cvat_task'].item()):
                
                cvat_id = int(bds_index.loc[idx,'cvat_task'].item())
                    
                # Get all ids of the tasks in this project
                tasks = {task.id:task for task in task_mgr.project.get_tasks()}
                   
                if cvat_id in tasks.keys():
                    
                    good_input = False
                    
                    while True:
                        print('Task ' + str(cvat_id) + ' already exists in project ' +\
                              task_mgr.project.name + '.')
                        d = input("Do you want to delete the task on the server (y/n):")
                        
                        if d == 'y':
                            tasks[cvat_id].remove()
                            print('Deleted task ' + task_name + '.')
                            break
                        elif d == 'n':
                            return None
                
        print('Uploading task ' + task_name + ' to CVAT.')
        
        # Write the whole video as frames of png to a folder 
        folder_path = self.video_to_png(idx,**kwargs)
        
        # Inititalize object of thermopile class
        tparray = TPArray(width = bds_index.loc[idx,'Width'].item(),
                          height = bds_index.loc[idx,'Height'].item())
        
        # Create the task on the server
        task = task_mgr.create_task_from_data(video,
                                              task_name,
                                              folder_path,
                                              fps,
                                              k,
                                              tparray)
            
        # Enter the task id into the bds_index
        bds_index.loc[idx,'cvat_task'] = task.id
        bds_index.loc[idx,'cvat_fps'] = fps
        
        # Save the updated index
        try:
            self._write_fields({'index':bds_index})
            print('Task ' + task_name + ' successfully created.')
        except:
            print('Some error occured when creating task ' + \
                  task_name + '.' )
        
        
        return None
    
    def cvat_automatic_annotation(self,idx,**kwargs):
        
        # Check if a model is provided to use for automatic labelling, if 
        # not use BlobDetector
        model = kwargs.pop('model',BlobDetector)
        
        # Load index
        bds_index = self.load_index()
        
        # Convert index to int
        idx = self._idx_to_int(idx)
        
        # Get the desired fps specified at upload
        fps = int(bds_index.loc[idx,'cvat_fps'])
        
        # Get the task id
        task_id = int(bds_index.loc[idx,'cvat_task'])
        
        # Load video
        video_sequence = self.load_video(idx)
        
        # Initialize Array Type
        tparray = TPArray(bds_index.loc[idx,'Width'].item(),
                          bds_index.loc[idx,'Height'].item())
        
        # Load the model used for annotation
        model = BlobDetector(tparray._width, tparray._height)
        
        # Pass everything to the CVATAutoAnnot class
        cvat_auto_annot = CVATAutoAnnot(video_sequence,
                                        model,
                                        tparray,
                                        task_id,
                                        fps)
        auto_annotate(cvat_auto_annot)
        
    def cvat_automatic_shape_annotation(self,idx,**kwargs):
        
        # Check if a model is provided to use for automatic labelling, if 
        # not use BlobDetector
        model = kwargs.pop('model',BlobDetector)
        
        # Load index
        bds_index = self.load_index()
        
        # Convert index to int
        idx = self._idx_to_int(idx)
        
        # Get the desired fps specified at upload
        fps = int(bds_index.loc[idx,'cvat_fps'])
        
        # Get the task id
        task_id = int(bds_index.loc[idx,'cvat_task'])
        
        # Load video
        video_sequence = self.load_video(idx)
        
        # Initialize Array Type
        tparray = TPArray(bds_index.loc[idx,'Width'].item(),
                          bds_index.loc[idx,'Height'].item())
        
        # Load the model used for annotation
        model = BlobDetector(tparray._width, tparray._height)
        
        # Pass everything to the CVATAutoAnnot class
        cvat_auto_annot = CVATAutoAnnot(video_sequence,
                                        model,
                                        tparray,
                                        task_id,
                                        fps)
        auto_annotate(cvat_auto_annot)
    
    # def cvat_automatic_track_annotation(self,task_id:int,model:Tracktor,
    #                                     **kwargs:dict):
        
        
    #     # Load index
    #     bds_index = self.load_index()
        
    #     # Convert index to int
    #     # idx = self._idx_to_int(idx)
        
    #     # Get the desired fps specified at upload
    #     # fps = int(bds_index.loc[idx,'cvat_fps'])
        
    #     # Get the index of video 
    #     idx = bds_index.loc[bds_index['cvat_task'] == task_id].index
        
    #     # Load video
    #     video_sequence = self.load_video(idx)
               
    #     # Perform tracking on whole video
    #     tracks = model.track_video(video_sequence)
        
    #     # Use CVAT Task Manager to set the annotations on CVAT server to the 
    #     # tracking result
    #     cvat_mgr = CVATTaskMgr()
    #     cvat_mgr.set_task_annotation(task_id,tracks = tracks.copy())
        
    #     # upload cvat annotation by calling task.set_annotation
    #     # # Initialize Array Type
    #     # tparray = TPArray(bds_index.loc[idx,'Width'].item(),
    #     #                   bds_index.loc[idx,'Height'].item())
        
    #     # # Load the model used for annotation
    #     # model = BlobDetector(tparray._width, tparray._height)
        
    #     # # Pass everything to the CVATAutoAnnot class
    #     # cvat_auto_annot = CVATAutoAnnot(video_sequence,
    #     #                                 model,
    #     #                                 tparray,
    #     #                                 task_id,
    #     #                                 fps)
    #     # auto_annotate(cvat_auto_annot)
        
    #     return tracks
    
    def sync_with_cvat(self,cvat_project):
        
        # Load index
        bds_index = self.load_index()
        
        # Initialize the CVATTaskMgr
        task_mgr = CVATTaskMgr()
        task_mgr.project = cvat_project
        
        # If no project was found, return with Exception
        if task_mgr.project is None:
            raise Exception('No project with name ' + \
                            cvat_project +\
                            ' found on server.')
        
        # Create a dictionary of all tasks and their names
        task_list = task_list=task_mgr._project.get_tasks()
        
        task_dict = {}
        for task in task_list:
            # Get name and id of task
            task_name = task.api.retrieve(task.id)[0]['name']
            task_id = task.id
            
            # Get stage of the job
            jobs = task.get_jobs()
            
            if len(jobs) != 1:
                stage = 'None'
            else:
                stage = jobs[0].stage
                
            
            task_dict[task_name] = {'cvat_task':task_id,
                                    'stage':stage}
        
        # Set all task ids in bds_index to nan and then assign new task_ids
        bds_index['cvat_task'] = np.nan
        
        for task_name,task_info in task_dict.items():
            bds_iloc = [bdsname.split('.')[0] in task_name for bdsname in bds_index['BDSname']]
            bds_iloc = np.where(bds_iloc)

            if len(bds_iloc) == 0:
                print('Task ' + task_name + ' exists on the CVAT server ' +\
                      'but not in the local hdf5 file.')
            elif len(bds_iloc) > 1:
                print('Multiple .bds with the same name ' + task_name + \
                      ' exists in the local hdf5 file.')
            else:
                bds_idx = bds_index.index[bds_iloc]
                bds_index.loc[bds_idx,'cvat_task'] = task_info['cvat_task']
                bds_index.loc[bds_idx,'stage'] = task_info['stage']
        
        # Cast to integer
        bds_index = bds_index.astype({'cvat_task':float})
        # Write new index to hdf5 file
        try:
            self._write_fields({'index':bds_index})
            print('CVAT task ids successfully updated.')
        except:
            print('Some error occured when writing to hdf5 file' )
        
        
    
    def cvat_download_annotation(self,idx,**kwargs):
        
        mode = kwargs.pop('mode','a')
        
        # Load index
        bds_index = self.load_index()
        json_index = self.load_JSONindex()
        
        # Convert index to int
        idx = self._idx_to_int(idx)
        
        # Get the cvat id of this task on the server
        task_id = int(bds_index.loc[idx,'cvat_task'])
        
        # Create an instance of the CVATTaskMgr
        task_mgr = CVATTaskMgr()
        
        # Get the task
        task = task_mgr.get_task(task_id)
        
        # Check if annotation already exists in hdf5 file
        with h5py.File(self._hdf5_path,'a') as hdf5_file:
            
            exist = False
            if 'annotations/'+task.name in hdf5_file:
                exist = True
                print('Annotation ' + task.name + ' already exists.\n')

        if exist==True and mode == 'w':
            print('Data is deleted and rewritten.\n')
            self.delete_annotation(idx)
        elif exist==True and mode != 'w':
            print('No data is written to group. Pass mode="w" \n to overwrite existing data.')
            return None
        
        # Reload index in case it was changed through deletion
        json_index = self.load_JSONindex()
        
        # Dict for fields to write to hdf5
        data_to_write = {}
        
        # Get annotation for that task
        annot = task_mgr.get_task_annotation(task_id)
        
        if annot is None:
            return None 
        
        
        # Make an entry in the JSON_index for the new annotation
        json_row = pd.DataFrame(data=[],columns=json_index.columns,
                                 index=[0])
        
        
        json_row.loc[0,'BDSname'] = bds_index.loc[idx,'BDSname']
        json_row.loc[0,'bds_address'] = bds_index.loc[idx,'address']
        
        json_row.loc[0,'JSONname'] = task.name
        json_row.loc[0,'json_address'] = 'annotations/'+task.name

        json_row.index._data = bds_index.loc[[idx]].index._data
        
        json_index = pd.concat((json_index,json_row))
        
        # Write to dictionary
        data_to_write['annotations/'+task.name + '/data'] = annot  
        data_to_write['JSONindex'] = json_index
            
        # Write to file
        try:
            self._write_fields(data_to_write)
            print(task.name + ' successfully imported.')
        except:
            print('Some error occured when writing ' + task.name + '.' )        
        
        return None
    
    def plot_bboxes(self,bb_imgs,bb_annot,width,height,**kwargs):
        
        # First get number of bounding boxes to plot
        N = len(bb_annot)
        
        if N>200:
            m = 'Number exceeds maximum of 200 bounding boxes. ' + \
                'A random sample will be plotted'
            print(m)
            
            bb_annot = bb_annot.sample(n=200)
            
        # Calculate number of rows and columns needed
        num_rowcol = int(np.ceil(np.sqrt(len(bb_annot))))
        
        # Plot bboxes
        fig, ax = plt.subplots(num_rowcol,num_rowcol)
        
        ax = ax.flatten()

        c = 0

        for annot_idx in bb_annot.index:
            
            ax[c].imshow(bb_imgs.loc[annot_idx].values.reshape(height,
                                                               width))
            
            ax[c].set_xticks([])
            ax[c].set_yticks([])
            
            ax[c].set_title(annot_idx)

            c = c + 1
            
        return None
    