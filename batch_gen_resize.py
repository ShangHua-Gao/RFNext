#!/usr/bin/python2.7

import torch
import numpy as np
import random
from torch.utils.data import Dataset
import time
class SegDataset(Dataset):
    def __init__(self, num_classes, actions_dict, gt_path, features_path, sample_rate, vid):
        self.list_of_examples = list()
        self.index = 0
        self.num_classes = num_classes
        self.actions_dict = actions_dict
        self.gt_path = gt_path
        self.features_path = features_path
        self.sample_rate = sample_rate

        file_ptr = open(vid, 'r')
        self.list_of_examples = file_ptr.read().split('\n')[:-1]
        file_ptr.close()
    
        if '50salads' in gt_path:
            self.fix_size = 5000
        elif 'breakfast' in gt_path:
            self.fix_size = 2000
        elif 'gtea' in gt_path:
            self.fix_size = 1000
            
        self.mask = torch.ones(self.num_classes, self.fix_size, dtype=torch.float)


    def __getitem__(self, idx):
        
        batch = self.list_of_examples[idx]
        flag = 0
        try:
            batch_input = np.load(self.features_path + batch.split('.')[0] + '_fix' + '.npy')
            batch_target = np.load(self.features_path + batch.split('.')[0] + '_fix_label' + '.npy')
        except Exception as e:
            print('Catch:', e)
            features = np.load(self.features_path + batch.split('.')[0] + '.npy')
            file_ptr = open(self.gt_path + batch, 'r')
            content = file_ptr.read().split('\n')[:-1]
            t11 = time.time()
            classes = np.zeros(min(np.shape(features)[1], len(content)))
            for i in range(len(classes)):
                classes[i] = self.actions_dict[content[i]]
            batch_input = features[:, ::self.sample_rate]
            batch_target = classes[::self.sample_rate]
        

            batch_input = torch.from_numpy(batch_input)
            batch_target = torch.from_numpy(batch_target)

            batch_input = torch.nn.functional.interpolate(batch_input.unsqueeze(0), size=self.fix_size, mode='nearest').squeeze()
            batch_target = torch.nn.functional.interpolate(batch_target.unsqueeze(0).unsqueeze(0), size=self.fix_size, mode='nearest').squeeze().long()

            np.save(self.features_path + batch.split('.')[0] + '_fix', batch_input.numpy())
            np.save(self.features_path + batch.split('.')[0] + '_fix_label', batch_target.numpy())
            flag = 1

        if flag == 0:
            batch_input = torch.from_numpy(batch_input)
            batch_target = torch.from_numpy(batch_target)

        mask = self.mask
        return batch_input, batch_target, mask
    
    def __len__(self):
        return len(self.list_of_examples)

    def collate_fn(self, batch):
        batch_input, batch_target = list(zip(*batch))
        
        length_of_sequences = map(len, batch_target)
        max_length = max(length_of_sequences)
        
        batch_input_tensor = torch.zeros(len(batch_input), np.shape(batch_input[0])[0], max_length, dtype=torch.float)
        batch_target_tensor = torch.ones(len(batch_input), max_length, dtype=torch.long)*(-100)

        mask = torch.zeros(len(batch_input), self.num_classes, max_length, dtype=torch.float)
        for i in range(len(batch_input)):
            batch_input_tensor[i, :, :np.shape(batch_input[i])[1]] = torch.from_numpy(batch_input[i])
            batch_target_tensor[i, :np.shape(batch_target[i])[0]] = torch.from_numpy(batch_target[i])
            mask[i, :, :np.shape(batch_target[i])[0]] = torch.ones(self.num_classes, np.shape(batch_target[i])[0])
        
        return batch_input_tensor, batch_target_tensor, mask


# class SegDataset(Dataset):
#     def __init__(self, num_classes, actions_dict, gt_path, features_path, sample_rate, vid):
#         self.list_of_examples = list()
#         self.index = 0
#         self.num_classes = num_classes
#         self.actions_dict = actions_dict
#         self.gt_path = gt_path
#         self.features_path = features_path
#         self.sample_rate = sample_rate

#         file_ptr = open(vid, 'r')
#         self.list_of_examples = file_ptr.read().split('\n')[:-1]
#         file_ptr.close()

#         self.mask = torch.ones(self.num_classes, 1100, dtype=torch.float)


#     def __getitem__(self, idx):
#         batch = self.list_of_examples[idx]
            
#         features = np.load(self.features_path + batch.split('.')[0] + '.npy')
#         file_ptr = open(self.gt_path + batch, 'r')
#         content = file_ptr.read().split('\n')[:-1]
#         classes = np.zeros(min(np.shape(features)[1], len(content)))
#         for i in range(len(classes)):
#             classes[i] = self.actions_dict[content[i]]
#         batch_input = features[:, ::self.sample_rate]
#         batch_target = classes[::self.sample_rate]
#         return batch_input, batch_target
    
#     def __len__(self):
#         return len(self.list_of_examples)

#     def collate_fn(self, batch):
#         batch_input, batch_target = list(zip(*batch))
        
#         length_of_sequences = map(len, batch_target)
#         max_length = max(length_of_sequences)
        
#         batch_input_tensor = torch.zeros(len(batch_input), np.shape(batch_input[0])[0], max_length, dtype=torch.float)
#         batch_target_tensor = torch.ones(len(batch_input), max_length, dtype=torch.long)*(-100)

#         mask = torch.zeros(len(batch_input), self.num_classes, max_length, dtype=torch.float)
#         for i in range(len(batch_input)):
#             batch_input_tensor[i, :, :np.shape(batch_input[i])[1]] = torch.from_numpy(batch_input[i])
#             batch_target_tensor[i, :np.shape(batch_target[i])[0]] = torch.from_numpy(batch_target[i])
#             mask[i, :, :np.shape(batch_target[i])[0]] = torch.ones(self.num_classes, np.shape(batch_target[i])[0])
        
#         return batch_input_tensor, batch_target_tensor, mask

        


class BatchGenerator(object):
    def __init__(self, num_classes, actions_dict, gt_path, features_path, sample_rate):
        self.list_of_examples = list()
        self.index = 0
        self.num_classes = num_classes
        self.actions_dict = actions_dict
        self.gt_path = gt_path
        self.features_path = features_path
        self.sample_rate = sample_rate

    def reset(self):
        self.index = 0
        random.shuffle(self.list_of_examples)

    def has_next(self):
        if self.index < len(self.list_of_examples):
            return True
        return False

    def read_data(self, vid_list_file):
        file_ptr = open(vid_list_file, 'r')
        self.list_of_examples = file_ptr.read().split('\n')[:-1]
        file_ptr.close()
        random.shuffle(self.list_of_examples)

    def next_batch(self, batch_size):
        batch = self.list_of_examples[self.index:min(self.index + batch_size, len(self.list_of_examples))]
        self.index += batch_size

        batch_input = []
        batch_target = []
        for vid in batch:
            features = np.load(self.features_path + vid.split('.')[0] + '.npy')
            file_ptr = open(self.gt_path + vid, 'r')
            content = file_ptr.read().split('\n')[:-1]
            classes = np.zeros(min(np.shape(features)[1], len(content)))
            for i in range(len(classes)):
                classes[i] = self.actions_dict[content[i]]
            batch_input .append(features[:, ::self.sample_rate])
            batch_target.append(classes[::self.sample_rate])

        length_of_sequences = map(len, batch_target)
        max_length = max(length_of_sequences)
        batch_input_tensor = torch.zeros(len(batch_input), np.shape(batch_input[0])[0], max_length, dtype=torch.float)
        batch_target_tensor = torch.ones(len(batch_input), max_length, dtype=torch.long)*(-100)
        mask = torch.zeros(len(batch_input), self.num_classes, max_length, dtype=torch.float)
        for i in range(len(batch_input)):
            batch_input_tensor[i, :, :np.shape(batch_input[i])[1]] = torch.from_numpy(batch_input[i])
            batch_target_tensor[i, :np.shape(batch_target[i])[0]] = torch.from_numpy(batch_target[i])
            mask[i, :, :np.shape(batch_target[i])[0]] = torch.ones(self.num_classes, np.shape(batch_target[i])[0])

        return batch_input_tensor, batch_target_tensor, mask
