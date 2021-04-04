import torch 
import torchvision
import json 
# def gen_config(model, mode='local'):
#     config = init_config()
#     for name, module in model.named_modules():
#         if isinstance(module, BaseOperator):
#             config['model'][name] = module.dilation
#     return config

def init_config():
    config = {}
    config['global'] = {}
    config['model'] = {}
    config['global']['step'] = 0
    config['global']['max_step'] = 31
    config['global']['search_interval'] = 3
    config['global']['exp_rate'] = 0.1
    config['global']['init_alphas'] = 0.01
    config['global']['normlize'] = 'absavg'
    config['global']['mmin'] = 1
    config['global']['mmax'] = 2048
    config['global']['finetune'] = False
    return config 

def load_config(filename, finetune):
    with open(filename, 'r', encoding='utf-8') as f:
        config = json.load(f)
        if finetune:
            config['global']['finetune'] = True
        return config

def write_to_json(dicts, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(dicts, f, indent=4)

def expands_rate(d, config):
    exp_rate = config['exp_rate']

    return [value_crop(int(round((1 - exp_rate) * d)), config['mmin'], config['mmax']), value_crop(d, config['mmin'], config['mmax']), \
            value_crop(int(round((1 + exp_rate) * d)), config['mmin'], config['mmax'])]

def value_crop(d, mind, maxd):
    if d < mind: 
        d = mind 
    elif d > maxd:
        d = maxd
    return d

# model = torchvision.models.resnet18()
# config = gen_config(model)
# print(config)
