#!/usr/bin/python2.7

import torch
import os
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import copy
import numpy as np
import pickle
import multiprocessing
from logger import Logger
from batch_gen_resize import SegDataset
from torch.utils.data import DataLoader
import ray, time
from eval import f_score
from local_searcher.search_engine import init_config, Searcher, write_to_json
from local_searcher.operators import BaseOperator

num_gpus=2
num_cpus=6

class MultiStageModel(nn.Module):
    def __init__(self, num_stages, num_layers, num_f_maps, dim, num_classes, dilations):
        super(MultiStageModel, self).__init__()
        assert len(dilations) == num_layers * num_stages, "Dilations rates are: %d" % dilations
        self.stage1 = SingleStageModel(num_layers, num_f_maps, dim, num_classes, dilations[:num_layers])
        self.stages = nn.ModuleList([copy.deepcopy(SingleStageModel(num_layers, num_f_maps, num_classes, num_classes, dilations[(s+1)*num_layers:(s+2)*num_layers])) for s in range(num_stages-1)])

    def forward(self, x, mask):
        out = self.stage1(x, mask)
        outputs = out.unsqueeze(0)
        for s in self.stages:
            out = s(F.softmax(out, dim=1) * mask[:, 0:1, :], mask)
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)
        return outputs


class SingleStageModel(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes, dilations):
        super(SingleStageModel, self).__init__()
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
        self.layers = nn.ModuleList([copy.deepcopy(DilatedResidualLayer(dilations[i], num_f_maps, num_f_maps)) for i in range(num_layers)])
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x, mask):
        out = self.conv_1x1(x)
        for layer in self.layers:
            out = layer(out, mask)
        out = self.conv_out(out) * mask[:, 0:1, :]
        return out


class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation, in_channels, out_channels, rates_select=False):
        super(DilatedResidualLayer, self).__init__()
        self.conv_dilated = nn.Conv1d(in_channels, out_channels, 3, padding=dilation, dilation=dilation)
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()
   
    def forward(self, x, mask):
        out = F.relu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return (x + out) * mask[:, 0:1, :]


class GA_solver():
    def __init__(self, num_blocks, num_layers, num_f_maps, dim, num_classes, save_dir, actions_dict, 
                gt_path, features_path, sample_rate, vid_list_file, vid_list_file_test, pop_nums=50, eval_epochs=5, iter_nums=100, multate_rate=0.2):
        self.pops = []

        self.pop_nums = pop_nums
        self.iter_nums = iter_nums
        self.multate_rate = multate_rate

        self.num_blocks = num_blocks
        self.num_layers = num_layers
        self.num_f_maps = num_f_maps 
        self.dim = dim
        self.num_classes = num_classes
        self.num_epochs = eval_epochs

        self.id = 1
        self._EMPTY = 34512

        #dataset
        self.actions_dict, self.gt_path, self.features_path, self.sample_rate, self.vid_list_file, self.vid_list_file_test= \
            actions_dict, gt_path, features_path, sample_rate, vid_list_file, vid_list_file_test

        self.save_dir = save_dir
        self.logger = Logger(os.path.join(save_dir, 'log.txt'))

        ray.init(num_gpus=num_gpus, num_cpus=num_cpus)

    def init_pops(self):
        length = self.num_blocks * self.num_layers
        for _ in range(self.pop_nums):
            self.pops.append(Evol_unit(length, self.id))
            self.id += 1

    def cross(self, rate1, rate2):
        assert len(rate1) == len(rate2)
        length = len(rate1)
        child1 = Evol_unit(length, self.id)
        self.id += 1
        child2 = Evol_unit(length, self.id)
        self.id += 1

        pos1 = int(np.random.randint(0, self.num_blocks * self.num_layers))
        pos2 = int(np.random.randint(0, self.num_blocks * self.num_layers))
        if pos1 > pos2:
            pos1, pos2 = pos2, pos1

        for i in range(length):
            if i >= pos1 and i <= pos2:
                child1[i] = rate2[i]
                child2[i] = rate1[i]
            else:
                child1[i] = rate1[i]
                child2[i] = rate2[i]
        
        return child1, child2
    
    def multate(self, rate):
        length = len(rate)
        child = Evol_unit(length, self.id)
        self.id += 1

        for i in range(length):
            if np.random.rand() < 0.2:
                child[i] = nearest_2(int(np.random.uniform(1, 1024)))
            else:
                child[i] = rate[i]

        return child

        
    def eval_params(self, rate, idx, epochs=None):
        if epochs is not None:
            args = (self.num_blocks, self.num_layers, self.num_f_maps, self.dim, self.num_classes, rate, \
            50, self.actions_dict, self.gt_path, self.features_path, self.sample_rate, self.vid_list_file, self.vid_list_file_test, idx)
        else:
            args = (self.num_blocks, self.num_layers, self.num_f_maps, self.dim, self.num_classes, rate, \
                self.num_epochs, self.actions_dict, self.gt_path, self.features_path, self.sample_rate, self.vid_list_file, self.vid_list_file_test, idx)
        # q = self.queue
        return args
        # self.jobs_pools[pool_idx].apply(run, (args, q))
        # return multiprocessing.Process(target=self.run, args=(args, q))

    def filter(self):
        new_pops = []
        self.pops = sorted(self.pops)
        for i in range(self.pop_nums):
            if self.pops[i].acc < self.pops[0].acc - 10:
                continue
            new_pops.append(self.pops[i])
            self.logger.info('Select unit (id:%d) for next iteration, val_acc=%.4f.' % (self.pops[i].id, self.pops[i].acc))
        if len(new_pops) < self.pop_nums:
            for i in range(len(new_pops), self.pop_nums):
                length = len(self.pops[0])
                new_pops.append(Evol_unit(length, self.id))
                self.id += 1
        return new_pops

    def select(self):
        self.pops = sorted(self.pops)
        percentage = np.zeros(len(self.pops))
        indexes = []
        for i in range(len(percentage)):
            if i == 0:
                percentage[i] = self.pops[i].acc
            else:
                percentage[i] = self.pops[i].acc + percentage[i-1]
        for i in range(len(percentage)):
            rd = np.random.randint(0, int(percentage[-1]))
            for j in range(len(percentage)):
                if percentage[j] > rd:
                    index = j-1
                    break
            indexes.append(index)
        return indexes

    def solve(self):
        self.init_pops()
        for iter in range(self.iter_nums):
            self.logger.info('----Start %dth iteration.----' % (iter+1))
            indexes = self.select()
            for i in range(1, len(indexes)-1, 2):
                child1, child2 = self.cross(self.pops[indexes[i-1]], self.pops[indexes[i]])
                self.pops.append(child1)
                self.pops.append(child2)
                if np.random.rand() < self.multate_rate:
                    self.pops.append(self.multate(child1))
                if np.random.rand() < self.multate_rate:
                    self.pops.append(self.multate(child2))

            regs = []
            for i in range(len(self.pops)):
                if self.pops[i].iter_epochs >= 5:
                    continue
                args = self.eval_params(self.pops[i], i)
                regs.append(run.remote(args))

            results = ray.get(regs)
            for i in range(len(regs)):
                rate, idx = results[i]
                self.pops[idx] = rate    

            self.pops = self.filter()

            for i in range(len(self.pops)):
                model = MultiStageModel(self.num_blocks, self.num_layers, self.num_f_maps, self.dim, self.num_classes, self.pops[i])
                cfg = init_config()
                searcher = Searcher(cfg, model)
                searcher.wrap_model(model, cfg, search_op='Conv1d')
                for name, module in model.named_modules():
                    if isinstance(module, BaseOperator):
                        cfg['model'][name] = module.op_layer.dilation
                write_to_json(cfg, os.path.join(self.save_dir, 'iter%d_pop[%d].json' % (iter+1, i)))

                np.save(os.path.join(self.save_dir, 'iter%d_pop[%d]' % (iter+1, i)), np.array(self.pops[i].rates))
                self.logger.info('SAVE: id=%d, acc=%.4f, f=%.4f, dir:%s' % (self.pops[i].id, self.pops[i].acc, self.pops[i].f, os.path.join(self.save_dir, 'iter%d_pop[%d]' % (iter+1, i))))
            
            if (iter + 1) % 10 == 0:
                regs = []
                for i in range(min(len(self.pops), 8)):
                    args = self.eval_params(self.pops[i], i, 50)
                    regs.append(run.remote(args))
                results = ray.get(regs)
                for i in range(min(len(self.pops), 8)):
                    rate, idx = results[i]
                    self.logger.info('TEST UNIT: id=%d, acc=%.4f, f=%.4f after 50 epochs training, rates save to %s.' % (rate.id, rate.acc50, rate.f50, os.path.join(self.save_dir, 'iter%d_pop[%d]' % (iter+1, i))))
            try:
                os.system('ps -ef | grep ray::IDLE | grep -v grep | cut -c 9-15 | xargs kill -9')
            except Exception as e:
                pass
                

class Evol_unit():
    def __init__(self, length, id=-1):
        rd = np.random.uniform(0, 11, size=length).astype(np.int32)
        self.rates = (2 ** rd).astype(np.int32).tolist()
        self.id = id
        self.acc = 1
        self.acc50 = 0
        self.f = 1
        self.f50 = 0
        self.iter_epochs = 0

    def __setitem__(self, key, value):
        self.rates[key] = value
    
    def __getitem__(self, key):
        return self.rates[key]

    def __len__(self):
        return len(self.rates)
    
    def __lt__(self, other):
        return self.f > other.f


@ray.remote(num_gpus=1, max_calls=0, num_cpus=3)
def run(args):
        print('running...')
        num_blocks, num_layers, num_f_maps, dim, num_classes, rate, num_epochs, actions_dict, gt_path, features_path, sample_rate, vid_list_file, vid_list_file_test, idx = args 
        try:
            dataset_train = SegDataset(num_classes, actions_dict, gt_path, features_path, sample_rate, vid_list_file)
            dataloader_train = DataLoader(dataset=dataset_train, batch_size=8, shuffle=True, num_workers=4, pin_memory=True)#, collate_fn=dataset_train.collate_fn)
            dataset_val = SegDataset(num_classes, actions_dict, gt_path, features_path, sample_rate, vid_list_file_test)
            dataloader_val = DataLoader(dataset=dataset_val, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)#, collate_fn=dataset_val.collate_fn)
        except Exception as e:
            print(e)

        # print('Starting evaluation.')
        model = MultiStageModel(num_blocks, num_layers, num_f_maps, dim, num_classes, rate).cuda()
        ce = nn.CrossEntropyLoss(ignore_index=-100)
        mse = nn.MSELoss(reduction='none')
        optimizer = optim.Adam(model.parameters(), lr=0.0005)
        t = time.time()
        print('rrrr')
        for epoch in range(num_epochs):
            correct = 0
            total = 0

            for (batch_input, batch_target, mask) in dataloader_train:
                batch_input, batch_target, mask = batch_input.cuda(), batch_target.cuda(), mask.cuda()
                optimizer.zero_grad()
                predictions = model(batch_input, mask)

                loss = 0
                for p in predictions:
                    loss += ce(p.transpose(2, 1).contiguous().view(-1, num_classes), batch_target.view(-1))
                    loss += 0.15*torch.mean(torch.clamp(mse(F.log_softmax(p[:, :, 1:], dim=1), F.log_softmax(p.detach()[:, :, :-1], dim=1)), min=0, max=16)*mask[:, :, 1:])

                loss.backward()
                optimizer.step()
        print('time consuming: %.8f' % (time.time() - t))
        model.eval()
        overlap = [.1, .25, .5]
        tp, fp, fn, f1 = np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3)
        for (batch_input, batch_target, mask) in dataloader_val:
            with torch.no_grad():
                batch_input, batch_target, mask = batch_input.cuda(), batch_target.cuda(), mask.cuda()
                predictions = model(batch_input, mask)
                _, predicted = torch.max(predictions[-1].data, 1)
                correct += ((predicted == batch_target).float()*mask[:, 0, :].squeeze(1)).sum().item()
                total += torch.sum(mask[:, 0, :]).item()

                recognition = []
                predicted = predicted.squeeze()
                for i in range(len(predicted)):
                    recognition = np.concatenate((recognition, [list(actions_dict.keys())[list(actions_dict.values()).index(predicted[i].item())]]*sample_rate))
                gt_content = []
                batch_target = batch_target.squeeze()
                for i in range(len(batch_target)):
                    gt_content = np.concatenate((gt_content, [list(actions_dict.keys())[list(actions_dict.values()).index(batch_target[i].item())]]*sample_rate))
                for s in range(len(overlap)):
                    tp1, fp1, fn1 = f_score(recognition, gt_content, overlap[s])
                    tp[s] += tp1
                    fp[s] += fp1
                    fn[s] += fn1
        for s in range(len(overlap)):
            precision = tp[s] / float(tp[s]+fp[s])
            recall = tp[s] / float(tp[s]+fn[s])
            f1[s] = 2.0 * (precision*recall) / (precision+recall+1e-10)
            f1[s] = np.nan_to_num(f1[s])*100
                

        model.train()

        acc = float(correct)/total * 100
        rate.acc = acc
        rate.f = f1[0]
        rate.iter_epochs = epoch + 1
        if epoch > 5:
            rate.acc50 = acc 
            rate.f50 = f1[0]

        print('Evol_unit: ID=%d, val_acc=%.4f, F1@%0.2f=%.4f, F1@%0.2f=%.4f, F1@%0.2f=%.4f.' % (rate.id, rate.acc, overlap[0], f1[0], overlap[1], f1[1], overlap[2], f1[2]))
        return (rate, idx)

def nearest_2(x):
    n2 = [2 ** i for i in range(11)]
    idx = 1
    ab = 2 ** 11
    for n in n2:
        if ab > np.abs(x - n):
            ab = np.abs(x - n)
            idx = n
    return idx    
