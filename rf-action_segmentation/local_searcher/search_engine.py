import os 
import torch 
from local_searcher.operators import Conv1dOp, Conv2dOp, BaseOperator
from local_searcher.utils import init_config, write_to_json, load_config
import logging
logging.basicConfig(format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]', level = logging.NOTSET, datefmt='%Y-%m-%d%I:%M:%S %p')
logger = logging.getLogger('Searcher')
logger.setLevel(logging.INFO)

class Searcher():
    def __init__(self, config, model, logdir='./log'):
        self.config = config    
        self.model = model
        self.logdir = logdir
        os.makedirs(self.logdir, exist_ok=True)
    
    def step(self):
        if self.config['global']['finetune']:
            return
        self.config['global']['step'] += 1
        if (self.config['global']['step']) % self.config['global']['search_interval'] == 0 and (self.config['global']['step']) < self.config['global']['max_step']:
            self.search()
            for name, module in self.model.named_modules():
                if isinstance(module, BaseOperator):
                    self.config['model'][name] = module.op_layer.dilation
            write_to_json(self.config, os.path.join(self.logdir, 'local_search_config_step%d.json' % self.config['global']['step']))
        elif (self.config['global']['step'] + 1) == self.config['global']['max_step']:
            self.search_estimate_only()
    
    def search(self):
        for _, module in self.model.named_modules():
            if isinstance(module, BaseOperator):
                module.estimate()
                module.expand()
    
    def search_estimate_only(self):
        for _, module in self.model.named_modules():
            if isinstance(module, BaseOperator):
                module.estimate()

    def wrap_model(self, model, config, search_op='Conv1d', init_rates=None):
        op = 'torch.nn.' + search_op
        # logger.info("only support kernel size < 5 for now.")
        for name, module in model.named_children():
            if isinstance(module, eval(op)):
                if module.kernel_size[0] > 1 and module.kernel_size[0] < 5:
                    moduleWrap = eval(search_op+'Op')(module, init_rates, config['global'])
                    logger.info('Wrap model %s to %s.' % (str(module), str(moduleWrap)))
                    setattr(model, name, moduleWrap)
            elif isinstance(module, BaseOperator):
                pass
            else:
                self.wrap_model(module, config, search_op, init_rates)

    def set_model(self, model, config, search_op='Conv1d', init_rates=None, prefix=""):
        op = 'torch.nn.' + search_op
        # logger.info("only support kernel size < 5 for now.")
        for name, module in model.named_children():
            if prefix == "":
                fullname = name 
            else:
                fullname = prefix + '.' + name
            if isinstance(module, eval(op)):
                if module.kernel_size[0] > 1 and module.kernel_size[0] < 5:
                    if type(config['model'][fullname]) == type(0):
                        config['model'][fullname] = [config['model'][fullname]]
                    module.dilation = config['model'][fullname]
                    module.padding = config['model'][fullname]
                    setattr(model, name, module)
                    logger.info('Set module %s dilation as: [%d]' % (fullname, module.dilation[0]))
            elif isinstance(module, BaseOperator):
                pass
            else:
                self.set_model(module, config, search_op, init_rates, fullname)
