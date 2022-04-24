# G2L: global to local search for receptive fields (dilation rates)

A general receptive field searching method for CNN.
The official implemention of the CVPR2021 paper: 'Global2Local: Efficient Structure Search for Video Action Segmentation'

## News
- G2L search helps to achieve the 1st place ([Team Feedback](https://github.com/hlh981029/megcup-feedback)) in [2022 MegCup RAW image denoising](https://studio.brainpp.com/competition/5?name=2022%20MegCup%20%E7%82%BC%E4%B8%B9%E5%A4%A7%E8%B5%9B&tab=rank).

## Introduction
Temporal receptive fields of models play an important
role in action segmentation. Large receptive fields facilitate
the long-term relations among video clips while small receptive fields help capture the local details. Existing methods construct models with hand-designed receptive fields in
layers. Can we effectively search for receptive field combinations to replace hand-designed patterns? To answer this
question, we propose to find better receptive field combinations through a global-to-local search scheme. Our search
scheme exploits both global search to find the coarse combinations and local search to get the refined receptive field
combination patterns further. The global search finds possible coarse combinations other than human-designed patterns. On top of the global search, we propose an expectation guided iterative local search scheme to refine combinations effectively. Our global-to-local search can be plugged
into existing action segmentation methods to achieve stateof-the-art performance.

## Usage

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
python main.py --dataset=breakfast --split=1 --action=train --config='path to config (default as 'log/search_config_step30.json')' --tmp=results --finetune=True
```

### Testing:
```py
python main.py --dataset=breakfast --split=1 --action=predict --tmp=results --config=path/to/model_config --pretrain=path/to/model
```
## Citation
If you find this work or code is helpful in your research, please cite:
```
@inproceedings{gao2021global2local,
  title={Global2Local: Efficient Structure Search for Video Action Segmentation},
  author={Gao, Shang-Hua and Han, Qi and Li, Zhong-Yu and Peng, Pai and Wang, Liang and Cheng, Ming-Ming},
  booktitle=CVPR,
  year={2021}
}
```
## License

The source code is free for research and education use only. Any comercial use should get formal permission first.

## Contact
If you have any questions, feel free to E-mail Shang-Hua Gao (`shgao(at)live.com`) and Qi Han(`hqer(at)foxmail.com`).
