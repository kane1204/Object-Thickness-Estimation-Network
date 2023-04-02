import os
import cv2
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from skimage import io
from copy import deepcopy



class ThicknessDataset(Dataset):
    """Thickness dataset."""
    def __init__(self, root_dir, mode=0, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.mode = mode # Mode 0 is normal operation with all samples loaded. Mode 1 is for excluding the big things and 2 only loading the big things
        self.bigthings = ['02958343', '02691156', '04530566', '04468005', '03790512', '02924116', '04460130']

        self.data = []
        self.labels = self.load_labels()
        self.df_cols = ['cam_pos', 'catagory_id', 'catagory', 'model_id', 'sample_no', 'depth_map_path', 'thick_map_path', 'img_path']
        self.dataframe = self.load_data()
        
    def load_labels(self):
        """Load labels from root_dir"""
        # read the Taxonomy.txt file which has format """<catagory name> - <catagory_id>"""
        # return a dictionary with key as catagory_id and value as CATAGORY_NAME
        labels = {}
        with open(os.path.join(self.root_dir, "Taxonomy.txt"), 'r') as f:
            for line in f:
                line = line.strip().split(" - ")
                labels[line[1]] = line[0]
        return labels
    

    def load_data(self):
        """Load data from root_dir"""
        tempDf = pd.DataFrame(columns=self.df_cols)
        # Data should be loaded from each folder in root_dir into a pandas dataframe
        for catagory in os.listdir(self.root_dir)[:-1]:
            # Load data from folder
            # Append data and labels to self.data and self.labels
            if self.mode == 1 and catagory not in  self.bigthings:
                tempDf = self.load_model_and_samples(tempDf, catagory)
            elif self.mode == 2 and catagory in self.bigthings:
                tempDf = self.load_model_and_samples(tempDf, catagory)
            elif self.mode == 0:
                tempDf = self.load_model_and_samples(tempDf, catagory)
        tempDf = tempDf.replace({'catagory': self.labels})
        return tempDf
    
    def load_model_and_samples(self, tempDf, catagory ):
        for model in os.listdir(os.path.join(self.root_dir, catagory)):
                # Load data from model
                # Append data and labels to self.data and self.labels
                for sample in os.listdir(os.path.join(self.root_dir, catagory, model)):
                    # Load data from file
                    # Append data and labels to self.data and self.labels
                    filepath = os.path.join(self.root_dir, catagory, model, sample)
                    cam_pos = np.load(os.path.join(filepath,"cam_pos.npy"))
                    depth_map_path = os.path.join(filepath, 'depth_map.npy')
                    thick_map_path = os.path.join(filepath, 'thicc_map.npy')
                    img_path = os.path.join(filepath, 'img.png')
                    # append to dataframe
                    record = pd.DataFrame([{'cam_pos': cam_pos, 'catagory_id': catagory, 'catagory': catagory, 'model_id': model, 'sample_no': sample, 'depth_map_path': depth_map_path, 'thick_map_path': thick_map_path, 'img_path': img_path}])
                    # print(record)
                    tempDf = pd.concat([tempDf, record], ignore_index=True)
        return tempDf

    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = {}
        sample['cam_pos'] = self.dataframe.iloc[idx,self.df_cols.index('cam_pos')]
        sample['catagory'] = self.dataframe.iloc[idx,self.df_cols.index('catagory')]
        sample['catagory_id'] = self.dataframe.iloc[idx,self.df_cols.index('catagory_id')]
        sample['model_id'] = self.dataframe.iloc[idx,self.df_cols.index('model_id')]
        sample['sample_no'] = self.dataframe.iloc[idx,self.df_cols.index('sample_no')]
        
        sample['depth_map'] = np.load(self.dataframe.iloc[idx,self.df_cols.index('depth_map_path')]).reshape(1,128,128)
        sample['thick_map'] = np.load(self.dataframe.iloc[idx,self.df_cols.index('thick_map_path')]).reshape(1,128,128)
        vis = np.load(self.dataframe.iloc[idx,self.df_cols.index('thick_map_path')]).reshape(1,128,128)
        vis = vis!=0
        vis = vis.astype(int)
        sample['visibility'] = vis
        sample['img_loc'] = self.dataframe.iloc[idx,self.df_cols.index('img_path')]
        sample['img'] = cv2.imread(self.dataframe.iloc[idx,self.df_cols.index('img_path')], cv2.COLOR_BGR2RGB)
        # This is the correct method to read the image but due to training on this error i will continue to use the above method
        # img = cv2.imread(self.dataframe.iloc[idx,self.df_cols.index('img_path')])
        # sample['img'] =  cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
         
        if self.transform is not None:
            sample['img'] = self.transform(image = sample['img'])['image']
        

        # Olds
        # if self.transform:
        #     sample = self.transform(sample)
        return sample
    
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        cols = sample.keys()
        sample = deepcopy(sample)
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img = sample['img'].transpose((2, 0, 1))
        img_tensor = torch.IntTensor(img)
        # assign all values to tensor and return dic

        dic ={
            'cam_pos': torch.FloatTensor(sample['cam_pos']),
            'catagory': sample['catagory'],
            'model_id': sample['model_id'],
            'sample_no': sample['sample_no'],
            'depth_map': torch.FloatTensor(sample['depth_map']),
            'thick_map': torch.FloatTensor(sample['thick_map']),
            'img': img_tensor
        }
        
        return dic