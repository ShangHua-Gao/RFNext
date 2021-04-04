# dilation_searcher

### Usage
#### Search:
```py
from search_engine import Searcher, init_config
...
model = resnet18()
config = init_config()
searcher = Searcher(config, model)
searcher.wrap_model(model, config, search_op='Conv2d', init_rates=3)

optimizer = SGD(model.parameters())
... ...
loss.backward()
optimizer.step()
searcher.step()

```
#### finetune:
```py
from search_engine import Searcher, init_config
...
model = resnet18()
config = load_config('searched_config.json', finetune=True)
searcher = Searcher(config, model)
searcher.set_model(model, config, search_op='Conv2d')

optimizer = SGD(model.parameters())
... ...
loss.backward()
optimizer.step()
searcher.step()
```

Note:
`find_unused_parameters=True` is required when use `DistributedDataParallel`
```
torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
```
