import os
import json
import numpy as np
from torch.utils.data import Dataset
from hydra.utils import get_original_cwd
from .io import IO
from dataset_modules.custom_transformations import pc_random_rotation, random_shuffle, pc_normalization, pcd_scale_and_translate
import torch

class Classification_Dataset(Dataset):
    def __init__(self, config, subset='train'):
        self.data_root = os.path.join(get_original_cwd(),config.data_path)
        self.npoints = config.npoints
        self.random_rotation = config.random_rotation
        self.subset = subset
        self.data_list_file = os.path.join(self.data_root, f'{self.subset}.txt')

        self.category_path = os.path.join(self.data_root, "cat_dict.json")
        with open(self.category_path, 'r') as f:
            self.cat_dict = json.load(f)
        
        with open(self.data_list_file, 'r') as f:
            lines = f.readlines()
        
        self.file_list = []
        for line in lines:
            line = line.strip()
            # Parse the dental dataset format: category/filename_(number).type
            # Extract taxonomy_id (category) and model_id from the path
            assert '/' in line, f"Invalid line format: {line}"

            category_name = line.split('/')[0]  # e.g., 'modelteil_ring3' or '1gliedrig_fortsatz'
            taxonomy_id = self.cat_dict[category_name]  # e.g., '14' or '3'
            filename = line.split('/')[1]  # e.g., 'modelteil_ring3_(14).npy'
            model_id = filename.split('.')[0]  # e.g., 'modelteil_ring3_(14)'
            
            self.file_list.append({
                'taxonomy_id': taxonomy_id,
                'model_id': model_id,
                'file_path': line
            })

    def __getitem__(self, index: int):
        sample = self.file_list[index]
        points  = IO.get(os.path.join(self.data_root, sample['file_path'])).astype(np.float32)
        if points.shape[0] != self.npoints:
            raise ValueError(f"Point cloud {sample['file_path']} has only {points.shape[0]} points, which is not equal to the configured npoints {self.npoints}.")
        
        points = torch.from_numpy(points).float()  # N 3
        points = random_shuffle(points)
  
        if self.subset == 'train':
            points = pcd_scale_and_translate(points)
            
        if self.random_rotation:
            points = pc_random_rotation(points)

        points = pc_normalization(points)

        class_label = torch.tensor(int(sample['taxonomy_id'])).long()
        return points, class_label

    def __len__(self):
        return len(self.file_list)


