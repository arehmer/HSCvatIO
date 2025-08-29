import xml.etree.ElementTree as ET
from pathlib import Path
import re

import numpy as np
import pandas as pd

import PIL.Image

from typing import List

import cvat_sdk

import cvat_sdk.models as models
import cvat_sdk.auto_annotation as cvataa

from cvat_sdk.core.proxies.tasks import Task
from cvat_sdk.core.proxies.projects import Project
from cvat_sdk.models import TrackedShapeRequest,LabeledTrackRequest,LabeledDataRequest



class CVATTaskMgr():
    
    def __init__(self, credentials:tuple)->None:
               
       
        self.credentials = credentials
        
        self._upload_options = {'image_quality':100,
                               'sorting_method':'natural',
                               'use_cache':True}
        
        self._project_labels = [{'attributes': [{'default_value': 'undefined',
                                                'input_type': 'select',
                                                'mutable': True,
                                                'name': 'turn',
                                                'values': ['front',
                                                            'one_quarter',
                                                            'profile',
                                                            'three_quarter',
                                                            'back',
                                                            'undefined']},
                                                {'default_value': 'undefined',
                                                'input_type': 'select',
                                                'mutable': True,
                                                'name': 'pose',
                                                'values': ['standing', 'sitting', 'lying', 'undefined']},
                                                {'default_value': 'false',
                                                'input_type': 'checkbox',
                                                'mutable': True,
                                                'name': 'identifiable',
                                                'values': ['false']},
                                                {'default_value': 'true',
                                                'input_type': 'checkbox',
                                                'mutable': True,
                                                'name': 'in_frame',
                                                'values': ['false']}],
                                'color': '#1bd459',
                                'has_parent': False,
                                'name': 'person',
                                'parent_id': None,
                                'sublabels': [],
                                'type': 'rectangle'}]
        
    @property
    def project(self)->Project:
        return self._project
    
    @project.setter
    def project(self,project_name:str):
        
        # Connect to server and try to find the project
        with cvat_sdk.make_client(host="https://app.cvat.ai/",
                         credentials=self.credentials) as client:
            
            project_id = None
            
            for p in client.projects.list():
                if p.name ==  project_name:
                    project_id = p.id
            
            if project_id == None:
                print("No project with name '" + project_name + \
                          "' found on server.")
                    
                self._project = None
                    
            else:
                # Get the label structure of the project
                self._project = client.projects.retrieve(project_id)
                # self.labels = project.get_labels()
    
    def create_cvat_project(self,
                            project_name:str):
        
        # Check if a project with the given name already exists
        with cvat_sdk.make_client(host="https://app.cvat.ai/",
                         credentials=self.credentials) as client:
            
            project_id = None
            
            for p in client.projects.list():
                if p.name ==  project_name:
                    project_id = p.id
                    
            # If the project already exists, do 
            if project_id is not None:
                raise Exception("The project '" + project_name + \
                          "' already exists.")
                return None
                      
            # Call API to create project and set attribute
            self._project = \
                client.projects.create({'name':project_name,
                                        'labels':self._project_labels})
                                       
            print("Successfully created project '" + project_name + "'.")
            
            return None
        
  
    def create_task_from_data(self,
                              video:pd.DataFrame,
                              task_name:str,
                              path:Path,
                              fps:int,
                              k:int,
                              tparray)->Task:
                            
        # Calculate which frames to upload to meet the target fps given the
        # sensors sampling rate
        # Get the sampling rate of the sensor
        fs = tparray._fs
        
        
        # Get the index of the video sequence
        video_idx = video.index
        
        # If fps is specified, subsample original data such that the subsampled
        # data has the specified fps
        if not fps==-1:
            
            # Create an array with time stamps for the frames
            t = np.arange(0,(len(video_idx)-1)/fs,1/fs)
            
            # Create an array at which to sample the video to be as close to the 
            # k-times the desired fps as possible
            ts = np.arange(0,t[-1]+1/(k*fps),1/(k*fps))
            
            # For each desired sampling point find the closest actual sample
            t_idx = [np.argmin(abs(t-i)) for i in ts]
            video_idx = video_idx[t_idx]
        
        
        # Create a list with the expected file names 
        files = [path / (str(idx)+'.png') for idx in video_idx]
        
        with cvat_sdk.make_client(host="https://app.cvat.ai/",
                         credentials=self.credentials) as client:
            
            task_spec = {'name':task_name,
                         'project_id':self.project.id}
                        
            task = client.tasks.create_from_data(spec=task_spec,
                                                 resources=files,
                                                 data_params=self._upload_options)
            
        return task       
    
    def get_task_annotation(self,task_id:int):
        
        # Retrieve task from server
        task = self.get_task(task_id)
        
        # If the task contains more than one job, this code probably 
        # doesn't work right. Throw an exception and return nothing
        if len(task.get_jobs())>1:
            raise Exception('Task contains more than one job. Check if ' + \
                            'code works properly in this case and adapt ' +\
                            'code if necessary. Then delete this exception.')
            return None
        
        # Get the annotation
        annot = task.get_annotations().to_dict()

        # Check if any annotations exist
        if len(annot['shapes']) == 0 and len(annot['tracks']) == 0:
            # if no annotation exists, create a fake temporary annotation,

            task_meta = task.get_meta().to_dict()
            fake_track = pd.DataFrame()
            fake_track.loc[0,'image_id'] = \
                int(task_meta['frames'][0]['name'].split('.')[0])
            fake_track.loc[0,['xtl','ytl','xbr','ybr']] = [0,0,1,1]

            self.set_task_annotation(task_id,
                                     fake_track)
            
            # download it
            annot = task.get_annotations().to_dict()
            
            # construct a DataFrame from it
            annot = self.cvat_annot_to_pd(annot,task)
            
            # Drop the column elements
            annot = annot.drop(columns='elements')
            
            # Drop the rows with the fake annotation
            annot = annot.drop(index = annot.index)
            
            # Delete fake annoations from server again
            task.remove_annotations()
            
            return annot
           
        # Pass both the annotation and the task to cvat_annot_to_pd to parse
        # the annotation to a dataframe
        annot = self.cvat_annot_to_pd(annot,task)
        
        # At this point the column "elements" does not contain any information
        # and makes trouble down the road for being a list.
        annot = annot.drop(columns='elements')
                
        return annot

    def pd_to_LabeledShapeRequest(self,shapes:pd.DataFrame(),label_id):
        
        if len(shapes) == 0:
            return []
    
    def pd_to_LabeledTrackRequest(self,tracks:pd.DataFrame(),label_id):
        
        if len(tracks) == 0:
            return []
        
        # Filter out all frames that are not multiples of 4
        tracks = tracks.loc[tracks['frame'] % 4 == 0]
        
        # for each track, generate a labelled track request
        track_requests = []        
        
        for track_id in tracks.index.unique():
            
            # Get all bounding boxes from that track
            track = tracks.loc[[track_id]].copy()
            
            # Make frame the index
            track = track.set_index(keys='frame', drop = True)
            
            # Construct the LabeledTrackRequest
            track_request = {'frame':int(track.index[0])}
            
            track_request['shapes'] = []
            
            for frame_id in track.index:
                
                # opt = {}
                
                # Only every 4th frame is labelled
                # if frame_id % 4 != 0:
                #     continue

                frame = track.loc[frame_id]
                points = [float(c) for c in frame[['xtl','ytl','xbr','ybr']]]
                
                trs = TrackedShapeRequest(type = 'rectangle',
                                          frame = int(frame_id),
                                          points = points)
                
                track_request['shapes'].append(trs)
                
                # Switch track to oustide in next frame to avoid inter-
                # polation by cvat
                trs_out = TrackedShapeRequest(type = 'rectangle',
                                              frame = int(frame_id)+1,
                                              points = points,
                                              outside = True)
                
                track_request['shapes'].append(trs_out)
                
            # To end track completely, the outside property of the last shape 
            # must be set to true
            # trs_end = TrackedShapeRequest(type = 'rectangle',
            #                               frame = int(frame_id)+1,
            #                               points = points,
            #                               outside = True)
            
            # track_request['shapes'].append(trs_end)
            
            # Convert from dict to LabeledTrackRequest
            track_request = models.LabeledTrackRequest(**track_request,
                                                       label_id = int(label_id))
            track_requests.append(track_request)
        
        return track_requests
    
    def set_task_annotation(self,task_id:int,tracks:pd.DataFrame=[],
                            shapes:pd.DataFrame=[]):
        
        # Retrieve task from server
        task = self.get_task(task_id)
        
        # If the task contains more than one job, this code probably 
        # doesn't work right. Throw an exception and return nothing
        if len(task.get_jobs())>1:
            raise Exception('Task contains more than one job. Check if ' + \
                            'code works properly in this case and adapt ' +\
                            'code if necessary. Then delete this exception.')
            return None    
               
        # Get the id for the "person"-label that is used by this task
        labels = task.get_labels()
        
        label_id = labels[0].to_dict()['id']
        
        if len(labels)!=1:
            raise Exception('Code works only for one label and if that ' +\
                            'label is "person". If this project uses' +\
                            'multiple or different labels, adapt the code.')
            return None
        
        # Get the frames that are actually on the server. The server version is
        # a downsampled version of the local (full) version. Since tracking is
        # performed on the full version, one needs to filter out the frames
        # which are not on the server
        meta = task.get_meta().to_dict()
        
        # Construct a mapping image_id -> frame number on server
        server_frames = {int(meta['frames'][f]['name'].split('.')[0]) : f \
                         for f in range(len(meta['frames']))}
        
        tracks = tracks.loc[tracks['image_id'].isin(list(server_frames.keys()))]
        
        # Map image_ids to frame number on cvat server
        tracks['frame'] = tracks['image_id'].map(server_frames)
        # tracks = tracks.rename(columns={'image_id':'frame'})
        
        # Construct the LabeledTrackRequest
        track_request = self.pd_to_LabeledTrackRequest(tracks,label_id)

        # Construct the LabeledShapeRequest
        shape_request = self.pd_to_LabeledShapeRequest(shapes,label_id)
        
        # Construct LabeledDataRequest (annotations to upload)
        annot_upload = models.LabeledDataRequest(shapes = shape_request,
                                                 tracks = track_request)
        
        # Remove previous annotations
        task.remove_annotations()
        
        # Upload new annotations (shapes and tracks)
        task.set_annotations(annot_upload)
        
        print('Uploaded annotations to task ' + str(task_id) + \
              ' sucessfully.')
        return task
        
        
    
    def get_task(self,task_id:int):
        
        # Connect to server
        with cvat_sdk.make_client(host="https://app.cvat.ai/",
                         credentials=self.credentials) as client:
            
            # retrieve the task
            task = client.tasks.retrieve(task_id)
            
        return task
    
    def pd_to_cvat_annot(self,annot:pd.DataFrame)->dict:
        """
        Function for parsing a dataframe containing annotations to the dict
        format that can be used to pass LabeledShapeRequests and 
        LabeledTrackRequest to CVAT.
        

        Parameters
        ----------
        annot : pd.DataFrame
            DESCRIPTION.

        Returns
        -------
        dict
            DESCRIPTION.

        """
        
        
    
    def cvat_annot_to_pd(self,annot:dict,task:Task)->pd.DataFrame:
        """
        
                

        Parameters
        ----------
        annot : dict
            DESCRIPTION.
        labels : list
            DESCRIPTION.

        Returns
        -------
        None.

        """
        
        Warning('Function handles only shapes at this point. If tracks '+\
                      'are to be parsed to the dataframe, the code must be ' +\
                      'adapted!')
        
        meta = task.get_meta()
            
        # Parse shapes to dataframe
        df_shapes = self._cvat_shapes_to_pd(annot['shapes'])
        
        # Add a column track_id for compatibility with tracks. Set column to
        # -1 to indicate it's not a track
        if df_shapes is not None:
            df_shapes['track_id'] = -1
        
        # Loop over all labelled tracks next
        df_tracks = self._cvat_tracks_to_pd(annot['tracks'])
        
        # If no annotations exist
        
        # Merge tracks and shapes
        df_annot = pd.concat([df_shapes,df_tracks],axis=0)
        
        # For some readability, replace the column names of attribute_ids with
        # their corresponding clear-name label
        df_annot = self._rename_and_recast_attributes(df_annot,task)
        
        # The frame is the index in cvat. If not all images of a video are 
        # uploaded, the frame on cvat does not match the original frame in the 
        # video. The original frame can be recovered from the file name.
        meta = meta.to_dict()
        
        df_annot['image_id'] = None
        
        for f in range(0,len(meta['frames'])):
            
            image_name = meta['frames'][f]['name'].split('.')[0]
            image_id = re.findall(r'\d', image_name)
            image_id = int(''.join(image_id))
            
            df_annot.loc[df_annot['frame']==f,'image_id'] = image_id
        
        # Cast image_id to int
        df_annot = df_annot.astype({'image_id':np.uint64})
                
        return df_annot
    
    def _rename_and_recast_attributes(self,df_annot:pd.DataFrame,
                                      task:Task):
        """
        CVAT returns annotations with attributes referenced by their id instead
        of their clear name. This function renames the columns by replacing the 
        attribute IDs with their names and casting the column to the proper type
        

        Parameters
        ----------
        df_annot : pd.DataFrame
            DESCRIPTION.
        task : Task
            DESCRIPTION.

        Returns
        -------
        None.

        """
        
        # Get all task labels 
        labels = task.get_labels()
        
        # Loop through all labels
        for label in labels:
            
            # Cast label to dict
            label = label.to_dict()
            
            # Add the name of the label to the annotations
            df_annot.loc[df_annot['label_id']==label['id'],'label_name'] = \
                label['name']
            
            # Go through all attributes of this label
            for attr in label['attributes']:
                
                # Rename the corresponding column from attribute id to 
                # attribute name
                df_annot = \
                    df_annot.rename(columns={attr['id']:str(attr['name'])})
                
                # Cast the column to the proper type, depending on the type
                # of the attribute
                if attr['input_type'] == 'checkbox':
                    # Checkbox types are mapped to boolean
                    df_annot[attr['name']] = \
                        df_annot[attr['name']].map({'true': True, 'false': False})
                elif attr['input_type'] == 'select':
                    # Select type is kept as a string
                    pass
                else:
                    print('Input type ' + attr['input_type'] + 'is unknown.')
            
            
        return df_annot
        
    
    def _cvat_shapes_to_pd(self,shapes:list):
        
        # Initialize empty list which contains rows of the dataframe
        df = [] 
        
        # Loop over all labeled shapes (objects which are not part of a track)
        for shape in shapes:
            
            # Unpack the nested lists in each entry
            
            # Bounding box coordinates
            shape['xtl'] = np.round(shape['points'][0])
            shape['ytl'] = np.round(shape['points'][1])
            shape['xbr'] = np.round(shape['points'][2])
            shape['ybr'] = np.round(shape['points'][3])
            
            del shape['points']
            
            # Attributes
            for a in shape['attributes']:
                shape[a['spec_id']] = a['value']
                
            del shape['attributes']
            
            # from_dict() expects a dict of lists, therefore do
            df_dict = {key:[value] for key, value in shape.items()}
            
            df.append(pd.DataFrame().from_dict(df_dict))
        
        if len(df)>0:
            
            # Concatenate all rows 
            df = pd.concat(df)
            
            # Cast coordinate columns to integer
            df[['xtl','ytl','xbr','ybr']] = \
                df[['xtl','ytl','xbr','ybr']].astype(np.int64)
            
            # Take the unique internal cvat id as index for dataframe
            df = df.set_index('id',drop=True)
            
            return df
        
        else:
            
            return None
        
    def _cvat_tracks_to_pd(self,tracks:list):
        
        # check if empty
        if len(tracks) == 0:
            return None
        
        # Initialize a dataframe to save the shapes of each track in
        df = []
                
        for track in tracks:
            
            # Each track contains of multiple shapes. In each frame where the
            # box of the track was manually or automatically determined exists
            # a shape
            df_shapes = self._cvat_shapes_to_pd(track['shapes'])

            # # To be able to concatenate dataframe, the index needs to be unique
            # if len(df)>0:
            #     idx_last = df[-1].index[-1]
            #     df_shapes.index = range(idx_last+1,idx_last+1+len(df_shapes))

            # Delete the shapes from the dict since they are now already in the 
            # dataframe
            del track['shapes']
            
            # Unpack the nested attributes to dictionary of this track
            for a in track['attributes']:
                track[a['spec_id']] = a['value']
                
            # Delete the nested attributes
            del track['attributes']
            
            # Unpack all of the rest into a dataframe
            df_meta = {key:[value] for key, value in track.items()}
            df_meta = pd.DataFrame().from_dict(df_meta)
            
            # Rename the column id into track_id to avoid mix-up with the
            # (shape) id
            df_meta = df_meta.rename(columns={'id':'track_id'})
            
            # Drop the frame column. It's the beginning of the track but also
            # a property of the shapes and can be reconstructed from that
            df_meta = df_meta.drop(columns='frame')
            df_meta = df_meta.astype({'track_id':int})
            
            # Replicate dataframe containing metadata to same length as df_shapes 
            df_meta = pd.concat([df_meta for i in df_shapes.index])
            df_meta.index = df_shapes.index
            # Concatenate the dataframe containing the shape and track
            # information
            df_track = pd.concat([df_shapes,df_meta],axis=1)
            
            # Add dataframe with track information to list
            df.append(df_track)
            
        # Concatenate all track dataframes
        df = pd.concat(df,axis=0)
        

        
        return df
        
    


class CVATAutoAnnot():
    
    def __init__(self,video_sequence,model,tparray,task_id,fps,**kwargs):
        
        self.model = model
        self.task_id = task_id
        self.video_sequence = video_sequence
        self.tparray = tparray
        
        fs = tparray._fs
        
        # Calculate at which time stamps to sample to annotate at desired 
        # frame rate
        
        # Get the index of the video sequence
        video_idx = self.video_sequence.index
        
        # Create an array with time stamps for the frames
        t = np.arange(0,(len(video_idx)-1)/fs,1/fs)
        
        # Create an array at which to sample the video to be as close to  
        # the desired fps as possible
        ts = np.arange(0,t[-1],1/(fps))
        
        # For each desired sampling point find the closest actual sample
        t_idx = [np.argmin(abs(t-i)) for i in ts]
        self.annotate_idx = video_idx[t_idx]
                
        return None
    
    @property
    def spec(self) -> cvataa.DetectionFunctionSpec:
        
        # The DetectionFunctionSpec is defined by the model itself
        DetFuncSpec = self.model._DetectionFunctionSpec
        
        
        return DetFuncSpec
    
    def detect(self, context, image: PIL.Image.Image)-> List[models.LabeledShapeRequest]:
        
        # context.frame_name is the name of the uploaded file. The files are
        # named according to their index in the original dataframe
        # Hence using that index is more consistent than relying on CVATS
        # internal labelling
        
        idx = int(context.frame_name.split('.')[0])
        
        bboxes = []
        
        # Label every nth frame
        if idx in self.annotate_idx:
            # Load the original image from the video sequence, choose only 
            # columns with pixel values
            img = self.video_sequence.loc[idx,self.tparray._pix].values
            
            # Reshape the image from row vector to array
            img = img.reshape(self.tparray._npsize)
            
            # Apply the model to the frame
            bboxes = self.model.predict_cvat(img)
            
        return bboxes

