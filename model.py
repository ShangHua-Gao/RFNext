#!/usr/bin/python2.7

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import copy
import numpy as np
from eval import edit_score, f_score, read_file

class MultiStageModel(nn.Module):
    def __init__(self, num_stages, num_layers, num_f_maps, dim, num_classes):
        super(MultiStageModel, self).__init__()
        self.stage1 = SingleStageModel(num_layers, num_f_maps, dim, num_classes)
        self.stages = nn.ModuleList([copy.deepcopy(SingleStageModel(num_layers, num_f_maps, num_classes, num_classes)) for s in range(num_stages-1)])

    def forward(self, x, mask):
        out = self.stage1(x, mask)
        outputs = out.unsqueeze(0)
        for s in self.stages:
            out = s(F.softmax(out, dim=1) * mask[:, 0:1, :], mask)
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)
        return outputs


class SingleStageModel(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes):
        super(SingleStageModel, self).__init__()
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
        self.layers = nn.ModuleList([copy.deepcopy(DilatedResidualLayer(2 ** i, num_f_maps, num_f_maps)) for i in range(num_layers)])
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x, mask):
        out = self.conv_1x1(x)
        for layer in self.layers:
            out = layer(out, mask)
        out = self.conv_out(out) * mask[:, 0:1, :]
        return out


class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation, in_channels, out_channels):
        super(DilatedResidualLayer, self).__init__()
        self.conv_dilated = nn.Conv1d(in_channels, out_channels, 3, padding=dilation, dilation=dilation)
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()

    def forward(self, x, mask):
        out = F.relu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return (x + out) * mask[:, 0:1, :]


class Trainer:
    def __init__(self, num_blocks, num_layers, num_f_maps, dim, num_classes):
        self.model = MultiStageModel(num_blocks, num_layers, num_f_maps, dim, num_classes)
        self.ce = nn.CrossEntropyLoss(ignore_index=-100)
        self.mse = nn.MSELoss(reduction='none')
        self.num_classes = num_classes

    def train(self, save_dir, dataloader, num_epochs, batch_size, learning_rate, device, searcher):
        self.model.train()
        self.model.to(device)
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50], gamma=0.1)
        for epoch in range(num_epochs):
            epoch_loss = 0
            correct = 0
            total = 0
            for (batch_input, batch_target, mask) in dataloader:
                batch_input, batch_target, mask = batch_input.to(device), batch_target.to(device), mask.to(device)
                optimizer.zero_grad()
                predictions = self.model(batch_input, mask)

                loss = 0
                for p in predictions:
                    loss += self.ce(p.transpose(2, 1).contiguous().view(-1, self.num_classes), batch_target.view(-1))
                    loss += 0.15*torch.mean(torch.clamp(self.mse(F.log_softmax(p[:, :, 1:], dim=1), F.log_softmax(p.detach()[:, :, :-1], dim=1)), min=0, max=16)*mask[:, :, 1:])

                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()
                

                _, predicted = torch.max(predictions[-1].data, 1)
                correct += ((predicted == batch_target).float()*mask[:, 0, :].squeeze(1)).sum().item()
                total += torch.sum(mask[:, 0, :]).item()

            searcher.step()
            scheduler.step()
            torch.save(self.model.state_dict(), save_dir + "/epoch-" + str(epoch + 1) + ".model")
            torch.save(optimizer.state_dict(), save_dir + "/epoch-" + str(epoch + 1) + ".opt")
            print("[epoch %d]: epoch loss = %f,   acc = %f" % (epoch + 1, epoch_loss / len(dataloader),
                                                               float(correct)/total))

    def predict(self, model_dir, results_dir, features_path, vid_list_file, pretrain, actions_dict, device, sample_rate):
        self.model.eval()
        with torch.no_grad():
            self.model.to(device)
            self.model.load_state_dict(torch.load(pretrain))

            file_ptr = open(vid_list_file, 'r')
            list_of_vids = file_ptr.read().split('\n')[:-1]
            file_ptr.close()

            overlap = [.1, .25, .5]
            tp, fp, fn = np.zeros(3), np.zeros(3), np.zeros(3)
            correct = 0
            total = 0
            edit = 0

            for vid in list_of_vids:
                #print(vid)
                features = np.load(features_path + vid.split('.')[0] + '.npy')
                features = features[:, ::sample_rate]
                input_x = torch.tensor(features, dtype=torch.float)
                input_x.unsqueeze_(0)
                x_size = input_x.shape[-1]
                #input_x = torch.nn.functional.interpolate(input_x, size=2000, mode='linear', align_corners=True)
                input_x = input_x.to(device)
                #print(input_x.shape)
                predictions = self.model(input_x, torch.ones(input_x.size(), device=device))
                #print(predictions.shape)
                predictions = predictions.squeeze()
                #predictions = torch.nn.functional.interpolate(predictions, size=x_size, mode='nearest')
                values, predicted = torch.max(torch.softmax(predictions[-1], dim=0).data, 0)
                values = values.cpu().numpy()
                entropy = -values * np.log(values)

                #predicted = predicted.squeeze()
                recognition = []
                for i in range(len(predicted)):
                    recognition = np.concatenate((recognition, [list(actions_dict.keys())[list(actions_dict.values()).index(predicted[i].item())]]*sample_rate))
                ground_truth_path = './data/' + 'breakfast' + '/groundTruth/'
                gt_file = ground_truth_path + vid 
                gt_content = read_file(gt_file).split('\n')[0:-1]

                f_name = vid.split('/')[-1].split('.')[0]
                np.save(results_dir + '/' + f_name + '.npy', entropy)
                f_ptr = open(results_dir + "/" + f_name, "w")
                f_ptr.write("### Frame level recognition: ###\n")
                f_ptr.write(' '.join(recognition))
                f_ptr.close()

                for i in range(len(gt_content)):
                    total += 1
                    if gt_content[i] == recognition[i]:
                        correct += 1

                edit += edit_score(recognition, gt_content)

                for s in range(len(overlap)):
                    tp1, fp1, fn1 = f_score(recognition, gt_content, overlap[s])
                    tp[s] += tp1
                    fp[s] += fp1 
                    fn[s] += fn1 

            acc_ = (100*float(correct)/total)
            edit_ = (1.0*edit)/len(list_of_vids)
            f = []
            for s in range(len(overlap)):
                precision = tp[s] / float(tp[s] + fp[s])
                recall = tp[s] / float(tp[s]+fn[s])
                f1 = 2.0 * (precision*recall) / (precision + recall)
                f1 = np.nan_to_num(f1)*100
                f.append(f1)

            return acc_, edit_, f
