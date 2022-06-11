import torch
from model import Trainer
from batch_gen import BatchGenerator, SegDataset
from torch.utils.data import DataLoader
import os
import argparse
import random
import numpy as np
from model_global import GA_solver
from local_searcher.search_engine import Searcher, init_config, write_to_json, load_config
from local_searcher.operators import BaseOperator

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = int(random.random() * 10000000)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    parser = argparse.ArgumentParser()
    parser.add_argument('--action', default='train')
    parser.add_argument('--dataset', default="gtea")
    parser.add_argument('--split', default='1')
    parser.add_argument('--tmp', default='./results')
    parser.add_argument('--num_epochs', type=int, default=65)
    parser.add_argument('--config', default='')
    parser.add_argument('--finetune', default=False)
    parser.add_argument('--pretrain', default='')

    args = parser.parse_args()

    num_stages = 4
    num_layers = 10
    num_f_maps = 64
    features_dim = 2048
    args.bs = 8
    args.lr = 0.0005

    # use the full temporal resolution @ 15fps
    sample_rate = 1
    # sample input features @ 15fps instead of 30 fps
    # for 50salads, and up-sample the output to 30 fps
    if args.dataset == "50salads":
        sample_rate = 2

    vid_list_file = "./data/"+args.dataset+"/splits/train.split"+args.split+".bundle"
    vid_list_file_tst = "./data/"+args.dataset+"/splits/test.split"+args.split+".bundle"
    features_path = "./data/"+args.dataset+"/features/"
    gt_path = "./data/"+args.dataset+"/groundTruth/"

    mapping_file = "./data/"+args.dataset+"/mapping.txt"

    model_dir = os.path.join(args.tmp, "./models/", args.dataset, "split_"+args.split)
    results_dir = os.path.join(args.tmp, "results/", args.dataset, "split_"+args.split)
    
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    file_ptr = open(mapping_file, 'r')
    actions = file_ptr.read().split('\n')[:-1]
    file_ptr.close()
    actions_dict = dict()
    for a in actions:
        actions_dict[a.split()[1]] = int(a.split()[0])

    num_classes = len(actions_dict)
    

    if args.action == 'global':
        ga_solver = GA_solver(num_stages, num_layers, num_f_maps, features_dim, num_classes, model_dir, \
                              actions_dict, gt_path, features_path, sample_rate, vid_list_file, vid_list_file_tst, \
                              pop_nums=50, eval_epochs=5, iter_nums=100, multate_rate=0.2)
        ga_solver.solve()
        exit()

    trainer = Trainer(num_stages, num_layers, num_f_maps, features_dim, num_classes)

    
    cfg = load_config(args.config, finetune=args.finetune)
    searcher = Searcher(cfg, trainer.model)
    searcher.set_model(trainer.model, cfg, search_op='Conv1d')
    if args.action == 'train' and not args.finetune:
        searcher.wrap_model(trainer.model, cfg, search_op='Conv1d')
    
    print(trainer.model)

    if args.action == "train":
        dataset = SegDataset(num_classes, actions_dict, gt_path, features_path, sample_rate, vid_list_file)
        dataloader = DataLoader(dataset=dataset, batch_size=args.bs, shuffle=True, num_workers=4, pin_memory=True, collate_fn=dataset.collate_fn)
        trainer.train(model_dir, dataloader, num_epochs=args.num_epochs, batch_size=args.bs, learning_rate=args.lr, device=device, searcher=searcher)
        
    if args.action == "predict":
        acc, edit, f = trainer.predict(model_dir, results_dir, features_path, vid_list_file_tst, args.pretrain, actions_dict, device, sample_rate)
        print('Acc, edit, f-score @(0.1, 0.25, 0.5):', acc, edit, f)

