import os

import pandas as pd
import numpy as np
import itertools

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from pathlib import Path

from hsod.cv.misc import IoU 

class AnnotMgr():
        
    
    def __init__(self,tparray):
        
        self.tparray = tparray
    
    def get_bboxes(self,video: pd.DataFrame,
               annot: pd.DataFrame,
               det_size: tuple,
               **kwargs)->dict:
            
        """
        Receives the size of a detection window and goes through every annotation
        on the original scale only.
        
        It adapts the bounding box slightly in the sense, that the adapted 
        bounding box must not become smaller than in the annotation and a 
        specified error in term of the IoU must not beexceeded
        ----------
        name : TYPE
            DESCRIPTION.
        det_width : TYPE
            DESCRIPTION.
        det_height : TYPE
            DESCRIPTION.
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        
        bbox_dict = {}
        
        # Loop over every annotation
        for a in annot.index:
            
            # Get the bounding box coordinates
            xtl = annot.loc[a,'xtl']
            ytl = annot.loc[a,'ytl']
            xbr = annot.loc[a,'xbr']
            ybr = annot.loc[a,'ybr']
            
            # Get id of the frame in the video
            frame_id = annot.loc[a,'image_id']
            
            # Get the frame
            frame = video.loc[frame_id,self.tparray._pix].values
            frame = frame.reshape(self.tparray._npsize)
            
            # Extract the box from the frame
            bbox_dict[a] = frame[ytl:ybr,xtl:xbr]
        
        return bbox_dict

    def get_frames(self,annot:pd.DataFrame,**kwargs)->dict:
            
        """
        Fetches the frame to every annotation from the provided video
        ----------
        name : TYPE
            DESCRIPTION.
        det_height : TYPE
            DESCRIPTION.
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        
        frame = kwargs.pop('frame',None)
        image_id = kwargs.pop('image_id',None)
        video = kwargs.pop('video',None)              
        
        # If a single frame is provided, this function is superfluous
        # because its purpose is to extract only those  frames from a sequence 
        # for which annotations exist 
        if frame is not None:
                            
            return {image_id:frame}
        
        elif video is not None:
        
            # Initialize a dictionary to store frames in
            frame_dict = {}
            
            # Loop over every annotation
            for a in annot.index:
                
                # Get id of the frame in the video
                image_id = annot.loc[a,'image_id'].astype(int)
                                
                # Get the frame
                try:
                    frame = video.loc[image_id,self.tparray._pix].values
                except:
                    pass
                frame = frame.reshape(self.tparray._npsize)
                                            
                frame_dict[image_id] = frame
    
            return frame_dict

    def add_frame(self,annot,pix_frame):
        
        # Add a frame by shifting the upper left and bottom right corner
        for coord in ['xtl','ytl']:
            annot[coord] = annot[coord] - pix_frame
        
        for coord in ['xbr','ybr']:
            annot[coord] = annot[coord] + pix_frame
            
        # Then remove all boxes that are outside of the frame
        
        annot = annot.loc[(annot['xtl']>=0) &\
                          (annot['ytl']>=0) &\
                          (annot['xbr']<=self.tparray._width) &\
                          (annot['ybr']<=self.tparray._height)]
                          
        return annot
        
    def IoU_with_DetWndw(self,annot:pd.DataFrame,
                         det_size:tuple):
        
        # Calculate the IoU with the desired detection window size 
        annot['IoU'] = np.nan
                   
        for a in annot.index:
            tl = (annot.loc[a,'xtl'],annot.loc[a,'ytl'])
            br = (annot.loc[a,'xbr'],annot.loc[a,'ybr'])
            
            br_det = (tl[0]+det_size[0],tl[1]+det_size[1])
            
            annot.loc[a,'IoU'] = IoU((tl,br),
                                     (tl,br_det))
            
        # Cast corners of bounding box to integer
        annot = annot.astype({'xtl':int, 'ytl':int, 'xbr':int,
        'ybr':int})
            
        return annot
    
    def calc_pmetrics(self,annot_true:pd.DataFrame,
                      annot_pred:pd.DataFrame,
                      **kwargs):
        """
        Function for calculating the mean average precision 

        Parameters
        ----------
        annot_true : TYPE
            DESCRIPTION.
        annot_pred : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        
        IoU_AP = kwargs.pop('IoU_AP',np.arange(0.5,0.9,0.05))
        IoU_AP = (np.array(IoU_AP)*100).astype(int)
        
        conf_AP = kwargs.pop('conf_AP',
                                  np.arange(-2,2,0.1))

        # 1. Apply only to positive predicted boxes
        prop_pos = annot_pred.loc[annot_pred['label']!=0].copy()
        
        # 2. Calculate the IoU of every positive box to any ground truth box
        prop_pos = self.IoU_of_2annots(prop_pos, annot_true)
        
        prop_pos = prop_pos.reset_index().set_index('id')
        
        # 2. Calculate the confusion matrix for different thresholds for
        # IoU and classifier confidence       
        mult_index = pd.MultiIndex.from_product([IoU_AP,conf_AP],
                                                names = ['IoU_lim','conf_lim'])
        
        # Initialize DataFrame for test statistics
        statistics =  pd.DataFrame(data=[],
                                   columns=['TP','FP','FN'],
                                   index = mult_index,
                                   dtype=int)
        
        for idx in statistics.index:

            
            # Calculate the maximal IoU of every predicted box to all
            # ground truth boxes
            conf_matrix,classification = self._confusion_matrix(annot_true,
                                                                prop_pos.copy(),
                                                                IoU_lim=idx[0]/100,
                                                                conf_lim=idx[1])
            
            statistics.loc[idx] = conf_matrix.loc[idx]
        
        # Calculate precision and recall
        statistics = self._prec_rec(statistics)
        
        # Calculate the F1 score
        statistics = self._F1(statistics)
        
        # Calculate the Average Precision 
        AP = self._AP(statistics.copy())
        
        # If only one parameterization is tested, also return a dataframe
        # containing annotations and classification errors 
        if len(mult_index) == 1:
            return {'metrics':statistics,
                    'AP':AP,
                    'classification':classification}
        else:
            return {'metrics':statistics,
                    'AP':AP}
    
    def plot_detector_perf(self,video:pd.DataFrame,classification:dict,
                           path:Path()):
        """
        Goes through the video frame by frame and draws bounding boxes for each
        true positive, false positive and false negative classification.
        Writes each plot to a .png file in a specified folder

        Parameters
        ----------
        video : pd.DataFrame
            DESCRIPTION.
        pmetrics : dict
            DESCRIPTION.

        Returns
        -------
        None.

        """
        
        # Use Agg to not actually show the figure when executing the code
        matplotlib.use('Agg')
        
        # Set image_id as index of annotations
        TP = classification['TP']#.reset_index().set_index('image_id')
        FP = classification['FP']#.reset_index().set_index('image_id')
        FN = classification['FN']#.reset_index().set_index('image_id')
        
        
        # Create folder for saving images if it doesn't exist
        if not path.is_dir():
            path.mkdir(parents=True)
        
        # Go over every image_id and get the appropriate image
        for i in video.index:
            
            # Get all TP,FP and FN for that frame
            tp = TP.loc[TP['image_id']==i]
            fp = FP.loc[FP['image_id']==i]
            fn = FN.loc[FN['image_id']==i]
            
            # write them to a dictionary to loop over
            detect_res = {'tp':tp,'fp':fp,'fn':fn}
        
            # If no classifications exist for this frame, skip and continue
            # to next frame
            if all([len(c) == 0 for c in [tp,fp,fn]]):
                continue
            
            # Otherwise get the image and plot it
            frame = video.loc[i,self.tparray._pix].values
            frame = frame.reshape(self.tparray._npsize)
            
            # Get maximum and minimum of data for scaling purposes
            vmin = frame.min()
            vmax = frame.max()
            
            fig,ax = plt.subplots(1,1)
            fig.suptitle('image_id: ' + str(i))
            ax.imshow(frame,vmin=vmin,vmax=vmax)
            
            # Then plot all bounding boxes
            # opt = {'tp':{'edgecolor':'b','facecolor':'b','alpha':0.3},
            #        'fp':{'edgecolor':'r','facecolor':'r','alpha':0.3},
            #        'fn':{'edgecolor':'m','facecolor':'m','alpha':0.3}}
            
            opt = {'tp':{'edgecolor':'b', 'facecolor':'none'},
                   'fp':{'edgecolor':'r', 'facecolor':'none'},
                   'fn':{'edgecolor':'m', 'facecolor':'none'}}
            
            # Loop over all classification types  
            for key in detect_res.keys():
                clas = detect_res[key]
                # Loop over all boxes
                for b in clas.index:
                    
                    # Get geometrics of the box
                    # Subtract 0.5 from the bottom left corner because 
                    # matplotlib draws boxes "through" pixels, not including 
                    # them
                    bl = (clas.loc[b,'xtl']-0.5,clas.loc[b,'ybr']-0.5)
                    w = clas.loc[b,'xbr'] - clas.loc[b,'xtl']
                    h = clas.loc[b,'ytl'] - clas.loc[b,'ybr']
                    
                    # Plot as patch
                    bbox = patches.Rectangle(bl,w,h,**opt[key])  
                    ax.add_patch(bbox)
            
            title_str = 'True Positives: '+ str(len(tp)) + '\n' +\
                        'False Positives: '+ str(len(fp)) + '\n' +\
                        'False Negatives: '+ str(len(fn)) 
            
            ax.set_title(title_str)
                
            
            # Check if file for frame already exists and remove if true 
            file_path = path/ (str(i) + '.png')
            
            if file_path.exists():
                os.remove(file_path)
                
            # After plotting all boxes, save the figure to file and close it
            plt.savefig(file_path)
            plt.close(fig)
            
            # Change backend to show figures again
            matplotlib.use('Qt5Agg')
            
        return None
        
    
    def _AP(self,df:pd.DataFrame):
        
        
        # Calculate the area under the precision recall curve for each IoU
        # threshold
        idx = df.index.get_level_values('IoU_lim').unique()
        AP = pd.DataFrame(data = [],
                          columns = ['AP'],
                          index = idx)
        
        for IoU_lim in AP.index:
            
            # Get all precision and recall values for this IoU threshold
            precrec = df.loc[IoU_lim]
            
            # Add a dummy value for recall=0 and precision=1
            precrec.loc[99,['rec','prec']] = [0,1]
            
            # Delete duplicates
            precrec = precrec.drop_duplicates()
            
            # Sort ascending by recall
            precrec = precrec.sort_values('rec',axis=0,ascending=True)
            
            # Calculate area under the curve via rectangle rule
            precrec['dr'] = precrec['rec'].diff()
            precrec['prec_dr'] = precrec['dr'] * precrec['prec']
            AP_i = sum(precrec.loc[precrec['prec_dr'].notna(),'prec_dr'])
            
            AP.loc[IoU_lim,'AP'] = AP_i
            
            
        return AP
    
    def _F1(self,df:pd.DataFrame):
        """
        Given a DataFrame containing precision and recall in the columns
        calculates the F1 score

        Parameters
        ----------
        prec_rec : pd.DataFrame
            DESCRIPTION.

        Returns
        -------
        None.

        """
        
        df['F1'] = 2*(df['prec'] * df['rec']) /\
            (df['prec'] + df['rec'])
    
        return df
    
    def _prec_rec(self,df:pd.DataFrame):
        """
        Given a DataFrame containing true positives (TP), false positives (FP),
        and false negatives (FN) in columns calculates precision and recall.

        Parameters
        ----------
        conf_matrix : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        
        # Calculate precision and recall for every confidence and IoU
        # threshold provided
        df['prec'] = df['TP'] / (df['TP'] + df['FP'])
        df['rec'] = df['TP'] / (df['TP'] + df['FN'])
        
        return df
        
    def _confusion_matrix(self,annot_true:pd.DataFrame,annot_pred:pd.DataFrame,
                          IoU_lim:float, conf_lim:float):
        
       
        # Calculate the maximal IoU of every predicted box to all
        # ground truth boxes
        # annot_pred = self.IoU_of_2annots(annot_pred, annot_true, 
        #                                  method='max')
        
        # annot_pred = annot_pred.reset_index().set_index('id')
        
        
        # Sort the predictions by confidence score
        annot_pred = annot_pred.sort_values('score',ascending=False)

        # Initialize confusion matrix
        mult_index = pd.MultiIndex.from_arrays([[int(IoU_lim*100)],[conf_lim]],
                                                names = ['IoU_lim','conf_lim'])
        conf_matrix =  pd.DataFrame(data=[],
                                    columns=['TP','FP','FN'],
                                    index = mult_index)
        
        # True positves are all predicted bounding boxes, that are above the
        # confidence threshold, and have IoU larger than IoU_lim
        TP = annot_pred.loc[(annot_pred['score']>=conf_lim) & \
                            (annot_pred['IoU_max']>=IoU_lim)]
        
        # Drop predictions of the same true bounding box
        TP = TP.drop_duplicates(subset='IoU_idxmax',keep='first')
        TP['classification'] = 'TP'
        
        # False positives are all remaining predictions above the confidence
        # threshold 
        FP = annot_pred.copy().drop(TP.index)
        FP = FP.loc[(FP['score']>=conf_lim)]
        FP['classification'] = 'FP'
        
        # False negatives are all true bounding boxes, that have not
        # been predicted
        FN = annot_true.loc[~annot_true.index.isin(TP['IoU_idxmax'])].copy()
        FN['classification'] = 'FN'
        
        
        classification = {'TP':TP,'FP':FP,'FN':FN}
      
        # Calculate entries of confusion matrix        
        conf_matrix.loc[mult_index[0],'TP'] = len(TP)
        conf_matrix.loc[mult_index[0],'FP'] = len(FP)
        conf_matrix.loc[mult_index[0],'FN'] = len(FN)
                    
        return conf_matrix,classification



    def IoU_of_2annots(self,annot1:pd.DataFrame,
                       annot2:pd.DataFrame,
                       **kwargs):
        """
        Function goes through two DataFrames containing annotations frame
        by frame. For each frame, the pairwise between all bounding
        boxes is calculated and the maximal IoU of every box in annot1 to 
        the corresponding box in annot2 is kept along the information,
        which the corresponding box in annot2 is. 

        

        Parameters
        ----------
        annot1 : pd.DataFrame
            DESCRIPTION.
        annot2 : pd.DataFrame
            DESCRIPTION.
        method : TYPE
            DESCRIPTION.
        IoU_lim : TYPE
            DESCRIPTION.
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        
        # method = kwargs.pop('method','<=')
        # IoU_lim = kwargs.pop('IoU_lim',None)
        

        if len(annot1)==0:
            # If annot1 itself is empty, return the empty annot1 as well
            annot1['IoU_max'] = pd.Series(dtype='int')
            annot1['IoU_idxmax'] = pd.Series(dtype='int')
            return annot1
        
        if len(annot2)==0:
            # If no DataFrame to compare annot1 to is provided, return 
            # annot 1 with some additional data for compatiblity
            annot1.loc[:,'IoU_max'] = 0
            annot1.loc[:,'IoU_idxmax'] = int(-99)
            return annot1

            
        # annot1 and annot2 might have different indices, remember what the
        # original index of each annotation was
        idx_annot1 = annot1.index.name
        idx_annot2 = annot2.index.name
        
        # Set index to image_id to compare only annotations that refer to the
        # same frame
        annot1 = annot1.reset_index().set_index('image_id')
        annot2 = annot2.reset_index().set_index('image_id')
        
        # Initialize a list for filtered annotations
        filt_annot1 = []
        
        # Loop over all frames (image_ids) of annot1
        for image_id in annot1.index.unique():
            
                # Get all annotations that refer to that frame from annot1
                frame_annot1 = annot1.loc[[image_id]].copy()
                frame_annot1 = frame_annot1.reset_index()#.set_index('id')
                
                # Get all annotations that refer to that frame from annot2
                if image_id in annot2.index:
                    frame_annot2 = annot2.loc[[image_id]].copy()
                    frame_annot2 = frame_annot2.reset_index()#.set_index('id')
                else:
                    # filt_annot1.append(frame_annot1)
                    continue
                
                # Calculate pairwise IoU of all annotations
                IoU_pw = self._pairwise_IoU(frame_annot1,frame_annot2)
                    
                # Get maximal IoU of each box in annot1 to any box in 
                # annot2
                frame_annot1['IoU_max'] = IoU_pw.max(axis=1)
                
                # Get index of box in annot2 that has maximal IoU to box in
                # annot1
                idx_max = IoU_pw.idxmax(axis=1)
                frame_annot1['IoU_idxmax'] = \
                    frame_annot2.loc[idx_max,idx_annot2].values

                filt_annot1.append(frame_annot1) 
        
        # Concatenate all DataFrames of annotations (if any)
        if len(filt_annot1)>0:
            
            # Concatenate to DataFrame
            filt_annot1 = pd.concat(filt_annot1)
            
            # Set index of annot1 back to original index
            filt_annot1 = filt_annot1.reset_index(drop=True).set_index(idx_annot1)
        else:
            # Else leave filt_annot1 as an empty list
            pass
        
        return filt_annot1

    
    def _pairwise_IoU(self,frame_annot1:pd.DataFrame,
                      frame_annot2:pd.DataFrame)->pd.DataFrame:
        """
        Calculates pairwise IoU of all bounding boxes given two different 
        annotations

        Parameters
        ----------
        frame_annot1 : pd.DataFrame
            DESCRIPTION.
        frame_annot2 : pd.DataFrame
            DESCRIPTION.

        Returns
        -------
        IoU_pw : pd.DataFrame.
       
        Structure of IoU_pw
                        annot2_idx0     annot2_idx1     ... 
        annot1_idx0     IoU_00          IoU_01
        annot1_idx1     IoU_10          IoU_11
            .
            .
            .
        
        
        """
        IoU_pw = pd.DataFrame(data=[],
                              columns=frame_annot2.index,
                              index = frame_annot1.index)
        IoU_pw = IoU_pw.astype(float)
        
        for a1,a2 in itertools.product(frame_annot1.index,
                                       frame_annot2.index):
            
            # Get corners of bounding box from annot 1
            tl_a1,br_a1 = self._get_bbox_corners(frame_annot1.loc[a1])
            tl_a2,br_a2 = self._get_bbox_corners(frame_annot2.loc[a2])

            
            IoU_pw.loc[a1,a2] = IoU((tl_a1,br_a1),(tl_a2,br_a2))
            
        return IoU_pw
            
    
    def _get_bbox_corners(self,annot_row):
        """
        Function receives a row from a DataFrame with annotations and return 
        the upper left and lower right corner as tuples

        Parameters
        ----------
        annot_row : TYPE
            DESCRIPTION.

        Returns
        -------
        tl : TYPE
            DESCRIPTION.
        br : TYPE
            DESCRIPTION.

        """
        
        if not isinstance(annot_row,pd.Series):
            raise TypeError('annot_row mus be pd.Series not ' + \
                            str(type(annot_row)))
        
        tl = (np.int32(annot_row['xtl']),
              np.int32(annot_row['ytl']))
        
        br = (np.int32(annot_row['xbr']),
              np.int32(annot_row['ybr']))
        
        return tl,br