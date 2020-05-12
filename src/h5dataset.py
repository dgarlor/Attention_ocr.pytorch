# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 16:36:17 2019

@author: dgarcialorenzo
"""
from torch.utils.data import Dataset
import h5pickle
import numpy as np

class H5Dataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, h5file, datasetImage='/train/image', 
                 datasetProf='/train/prof', transform=None):
        """
        Args:
            h5file (string): Path to the h5 file
            datasetImage (string): dataset for images
            datasetProf (string): dataset for professors            
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.isLoaded = False
        self.alphabet = "0123456789abcdefghijklmnopqrstuvwxyz"
        try:
            h5 = h5pickle.File( h5file, 'r', skip_cache=False)
            self.images = h5[ datasetImage]
            self.prof = h5[ datasetProf]
 
            self.length, self.width, self.height, self.channels = self.images.shape
            self.prof_lenght = self.prof.shape[1]
            self.isLoaded = True
        except OSError as e:
            print(" ** Error using H5 file",e)
            return
        except ValueError as e:
            print(" ** Value error",e)
            return
        
        #self.lenght = min(self.length, 512)
        self.reject = int( np.max(self.prof))
            
        # not implemented yet
        self.transform = transform

    def __len__(self):
        return self.length

    def blank(self):
        return self.reject
    
    def imaDims(self):
        """ Return width,height """
        return self.width, self.height
    
    def __getitem__(self, idx):
        tab = self.images[idx]
        from PIL import Image
        image = Image.fromarray(self.images[idx].transpose([2,1,0]).reshape(self.height, self.width))
        prof = self.prof[idx]
        prof_length = np.sum(prof < self.reject)
        if self.transform:
            image = self.transform(image)

        sprof = "".join([self.alphabet[p] for p in prof if p < len(self.alphabet)])
        return (image, sprof)
    
if __name__ == "__main__":
    import torch
    filename = r"C:\PROJECTS\plaques\data\plaques_tf1.14_cnt3_h72_NumAlpha_ActiveL.h5"
    train_dataset = H5Dataset(filename)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=16,
        shuffle=True,
        num_workers=int(2),
        )
    it_train = iter(train_loader)
    for epoch in range(5):
        train_iter = it_train.next()
        print(train_iter[1].shape)
        print(train_iter[0].shape)
    from PIL import Image
    ima = train_iter[0][0,0,:,:]
    a=Image.fromarray(ima.numpy()*255).convert('L')
    