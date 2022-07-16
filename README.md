# RF-Next: Efficient Receptive Field Search for Convolutional Neural Networks

A general receptive field searching method for CNN.**If your network has Conv with kernel larger than 1, RF-Next can further improve your model.**
The official implemention of the TPAMI2022 paper: 'RF-Next: Efficient Receptive Field Search for Convolutional Neural Networks',
CVPR2021 paper: 'Global2Local: Efficient Structure Search for Video Action Segmentation'


## News
- 2022.6.11 Code for mmcv and mmdetection is released. ConvNext, PVT, Res2Net, HRNet are supported. [code](https://github.com/ShangHua-Gao/RF-mmdetection/tree/rfsearch/configs/rfnext)
- 2022.4.24 RF-Next helps to achieve the 1st place ([Team Feedback](https://github.com/hlh981029/megcup-feedback)) in [2022 MegCup RAW image denoising](https://studio.brainpp.com/competition/5?name=2022%20MegCup%20%E7%82%BC%E4%B8%B9%E5%A4%A7%E8%B5%9B&tab=rank).
- 2022.2.10 RF-Next improve the SOTA CNN model ConvNeXt on multiple tasks.
- 2021.1.1 RF-Next for Video Action Segmentation.[code](https://github.com/ShangHua-Gao/RFNext/tree/main/rf-action_segmentation)

## Introduction
Temporal/spatial receptive fields of models play an important role in sequential/spatial tasks. Large receptive fields facilitate long-term relations, while small receptive fields help to capture the local details. Existing methods construct models with hand-designed receptive fields in layers. Can we effectively search for receptive field combinations to replace hand-designed patterns? To answer this question, we propose to find better receptive field combinations through a global-to-local search scheme. Our search scheme exploits both global search to find the coarse combinations and local search to get the refined receptive field combinations further. The global search finds possible coarse combinations other than human-designed patterns. On top of the global search, we propose an
expectation-guided iterative local search scheme to refine combinations effectively. Our RF-Next models, plugging receptive field search to various models, boost the performance on many tasks, e.g., temporal action segmentation, object detection, instance segmentation, and speech synthesis. 
## Applications
- Object detection and Instance segmentation [mmdet](https://github.com/ShangHua-Gao/RF-mmdetection/tree/rfsearch/configs/rfnext)
- Semantic segmentation
- Action segmentation
- Speech synthesis

## Citation
If you find this work or code is helpful in your research, please cite:
```
@article{gao2022rfnext,   
title={RF-Next: Efficient Receptive Field Search for Convolutional Neural Networks},   
author={Gao, Shanghua and Li, Zhong-Yu and Han, Qi and Cheng, Ming-Ming and Wang, Liang},   
journal=TPAMI,   
year={2022} }

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
If you have any questions, feel free to E-mail Shang-Hua Gao (`shgao(at)live.com`)
