import os
import glob
import pickle
import torch
import numpy as np
import pandas as pd
import pyvista as pv
import pytorch_lightning as pl
import torch_geometric.transforms as T

from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from typing import List



import h5py



class DataModule(pl.LightningDataModule):
    def __init__(self, dataset, path_train, path_test, val_split, batch_size, num_workers):
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.dataset = dataset(path_train, path_test)

        val_n = int(self.dataset.train_split[-1]*val_split)
        train_n = self.dataset.train_split[-1] + 1 - val_n
        self.train_set, self.val_set = torch.utils.data.random_split( torch.utils.data.Subset(self.dataset, self.dataset.train_split), lengths = [train_n, val_n])
        self.test_set = torch.utils.data.Subset(self.dataset, self.dataset.test_split)

       

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)



class h5_dataset(Dataset):
    def __init__(self, path_train: str , path_test: str ):
        super().__init__(path_train, transform=None, pre_transform=None, pre_filter=None)
        self.path_train = path_train
        self.path_test = path_test


        print('Loading training data...')
        f = h5py.File( self.path_train )
        self.h5_points = f['data'][:,:,0:3]
        self.h5_normals = f['data'][:,:,3:]
        self.h5_names = np.array(f['name'][:],dtype=str)
        self.h5_labels = np.array(f['label'][:],dtype=int)
        f.close()
        self.train_split = np.array(range(self.h5_points.shape[0]))
        print('Loading testing data...')
        f = h5py.File(  self.path_test )
        self.h5_points = np.concatenate([self.h5_points,f['data'][:,:,0:3]])
        self.h5_normals = np.concatenate([self.h5_normals,f['data'][:,:,3:]])
        self.h5_names = np.concatenate([self.h5_names, np.array(f['name'][:],dtype=str)])
        self.h5_labels = np.concatenate([self.h5_labels, np.array(f['label'][:],dtype=str)])
        self.test_split = np.array(range(len(f['label'][:]))) + self.train_split[-1] + 1
        f.close()
    
    def normalize_data( self, pointset ) :
        # Normalizes a point set
        
        # Normalizes position
        mean = torch.mean(pointset, dim=0)
        pointset = pointset - mean

        # Normalizes scale of a 3D model so that it is enclosed by a sphere whose radius is 0.5
        radius = 0.5
        norms = torch.norm(pointset, dim=1)
        max_norm = torch.max(norms)
        pointset = pointset * (radius / max_norm)

        return pointset


    def len(self):
        return len(self.h5_names)
    
    def get_points(self,idx):
        points = torch.from_numpy(np.array(self.h5_points[idx], dtype='float32'))
        points_normalized = self.normalize_data( points )
        return points_normalized

    def get_normals(self,idx):
        return torch.from_numpy(np.array(self.h5_normals[idx], dtype='float32'))
    
    def get_data(self, idx):
        pos = self.get_points(idx)
        normal = self.get_normals(idx)
        data = Data(
            pos=pos,
            normal=normal
        )
        return data

    def get(self, idx):
        sample = {}
        sample['x'] = self.get_data(idx)
        if idx in self.train_split:
            sample['train_set'] = 1
        else:
            sample['train_set'] = 0
        sample['idx'] = torch.from_numpy(np.array(idx, dtype='int64'))
        sample['y'] = torch.from_numpy(np.array(self.h5_labels[idx], dtype='int64'))
        return sample

    @property
    def num_features(self) -> int:
        return len(self[0]['x'])


