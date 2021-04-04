# G2L: global to local search

Code for CVPR 2021 paper: 'Global2Local: Efficient Structure Search for Video Action Segmentation'
### Data preparation:
Download data from [data](https://mega.nz/#!O6wXlSTS!wcEoDT4Ctq5HRq_hV-aWeVF1_JB3cacQBQqOLjCIbc8)., which contains the features and the ground truth labels. (~30GB)
Using a soft link to link the data to current path. (ln -s path/to/data ./data)

### Training:
#### Global Search:
```py
CUDA_VISIBLE_DEVICES=XXX pythton main.py --dataset=breakfast --split=1 --action=global --tmp=results_global
```

#### Local Search:
Firstly, a local search processing to find a fine-grid dilations on the global searched structures.
```py
python main.py --dataset=breakfast --split=1 --action=train --config='breakfast_sp1.json' --tmp=results_local
```
Then, training the model from the searched structures.
```py
python main.py --dataset=breakfast --split=1 --action=train --config='path to config (default as 'log/search_config_step30.json')' --tmp=results
```

### Testing:
```py
python main.py --dataset=breakfast --split=1 --action=predict --tmp=results --config=path/to/model_config --pretrain=path/to/model
