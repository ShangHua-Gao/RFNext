import torch
import torch.nn as nn
import torch.nn.functional as F
import collections.abc as container_abcs
from itertools import repeat
from timm.models.layers import get_padding


def _ntuple(n):
    def parse(x):
        if isinstance(x, container_abcs.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse


_pair = _ntuple(2)


def value_crop(dilation, min_dilation, max_dilation):
    if min_dilation is not None:
        if dilation < min_dilation:
            dilation = min_dilation
    if max_dilation is not None:
        if dilation > max_dilation:
            dilation = max_dilation
    return dilation


def rf_expand(dilation, expand_rate, num_branches, min_dilation=1, max_dilation=None):
    rate_list = []
    assert num_branches>=2, "number of branches must >=2"
    delta_dilation0 = expand_rate * dilation[0]
    delta_dilation1 = expand_rate * dilation[1]
    for i in range(num_branches):
        rate_list.append(
            tuple([value_crop(
                int(round(dilation[0] - delta_dilation0 + (i) * 2 * delta_dilation0/(num_branches-1))), min_dilation, max_dilation),
                value_crop(
                int(round(dilation[1] - delta_dilation1 + (i) * 2 * delta_dilation1/(num_branches-1))), min_dilation, max_dilation)
            ])
        )

    unique_rate_list = list(set(rate_list))
    unique_rate_list.sort(key=rate_list.index)
    return unique_rate_list


class RFConv2d(nn.Conv2d):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=1,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 padding_mode='zeros',
                 num_branches=3,
                 expand_rate=0.5,
                 min_dilation=1,
                 max_dilation=None,
                 init_weight=0.01,
                 search_interval=1250,
                 max_search_step=0,
                 rf_mode='rfsearch',
                 pretrained=None
                 ):
        if pretrained is not None and rf_mode == 'rfmerge':
            rates = pretrained['rates']
            num_rates = pretrained['num_rates']
            sample_weights = pretrained['sample_weights']
            sample_weights = self.normlize(sample_weights[:num_rates.item()])
            max_dliation_rate = rates[num_rates.item() - 1]
            if isinstance(kernel_size, int):
                kernel_size = [kernel_size, kernel_size]
            if isinstance(stride, int):
                stride = [stride, stride]
            new_kernel_size = (
                kernel_size[0] + (max_dliation_rate[0].item() -
                                  1) * (kernel_size[0] // 2) * 2,
                kernel_size[1] + (max_dliation_rate[1].item() - 1) * (kernel_size[1] // 2) * 2)
            # assign dilation to (1, 1) after merge
            new_dilation = (1, 1)
            new_padding = (
                get_padding(new_kernel_size[0], stride[0], new_dilation[0]),
                get_padding(new_kernel_size[1], stride[1], new_dilation[1]))

            # merge weight of each branch
            old_weight = pretrained['weight']
            new_weight = torch.zeros(
                size=(old_weight.shape[0], old_weight.shape[1],
                      new_kernel_size[0], new_kernel_size[1]),
                dtype=old_weight.dtype)
            for r, rate in enumerate(rates[:num_rates.item()]):
                rate = (rate[0].item(), rate[1].item())
                for i in range(- (kernel_size[0] // 2), kernel_size[0] // 2 + 1):
                    for j in range(- (kernel_size[1] // 2), kernel_size[1] // 2 + 1):
                        new_weight[:, :,
                                   new_kernel_size[0] // 2 - i * rate[0],
                                   new_kernel_size[1] // 2 - j * rate[1]] += \
                            old_weight[:, :, kernel_size[0] // 2 - i,
                                       kernel_size[1] // 2 - j] * sample_weights[r]

            kernel_size = new_kernel_size
            padding = new_padding
            dilation = new_dilation
            pretrained['rates'][0] = torch.FloatTensor([1, 1])
            pretrained['num_rates'] = torch.IntTensor([1])
            pretrained['weight'] = new_weight
            # re-initilize the sample_weights
            pretrained['sample_weights'] = pretrained['sample_weights'] * \
                0.0 + init_weight

        super(RFConv2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode
        )
        self.rf_mode = rf_mode
        self.pretrained = pretrained
        self.num_branches = num_branches
        self.max_dilation = max_dilation
        self.min_dilation = min_dilation
        self.expand_rate = expand_rate
        self.init_weight = init_weight
        self.search_interval = search_interval
        self.max_search_step = max_search_step
        self.sample_weights = nn.Parameter(torch.Tensor(self.num_branches))
        self.register_buffer('counter', torch.zeros(1))
        self.register_buffer('current_search_step', torch.zeros(1))
        self.register_buffer('rates', torch.ones(
            size=(self.num_branches, 2), dtype=torch.int32))
        self.register_buffer('num_rates', torch.ones(1, dtype=torch.int32))
        self.rates[0] = torch.FloatTensor([self.dilation[0], self.dilation[1]])
        self.sample_weights.data.fill_(self.init_weight)

        # rf-next
        if pretrained is not None:
            # load pretrained weights
            msg = self.load_state_dict(pretrained, strict=False)
            assert all([key in ['sample_weights', 'counter', 'current_search_step', 'rates', 'num_rates'] for key in msg.missing_keys]), \
                'Missing keys: {}'.format(msg.missing_keys)
        if self.rf_mode == 'rfsearch':
            self.estimate()
            self.expand()
        elif self.rf_mode == 'rfsingle':
            self.estimate()
            self.max_search_step = 0
            self.sample_weights.requires_grad = False
        elif self.rf_mode == 'rfmultiple':
            self.estimate()
            self.expand()
            # re-initilize the sample_weights
            self.sample_weights.data.fill_(self.init_weight)
            self.max_search_step = 0
        elif self.rf_mode == 'rfmerge':
            self.max_search_step = 0
            self.sample_weights.requires_grad = False
        else:
            raise NotImplementedError()

        if self.rf_mode in ['rfsingle', 'rfmerge']:
            assert self.num_rates.item() == 1

    def _conv_forward_dilation(self, input, dilation_rate):
        if self.padding_mode != 'zeros':
            return F.conv2d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            self.weight, self.bias, self.stride,
                            _pair(0), dilation_rate, self.groups)
        else:
            padding = (
                dilation_rate[0] * (self.kernel_size[0] - 1) // 2, dilation_rate[1] * (self.kernel_size[1] - 1) // 2)
            return F.conv2d(input, self.weight, self.bias, self.stride,
                            padding, dilation_rate, self.groups)

    def normlize(self, w):
        abs_w = torch.abs(w)
        norm_w = abs_w / torch.sum(abs_w)
        return norm_w

    def forward(self, x):
        if self.num_rates.item() == 1:
            return super().forward(x)
        else:
            norm_w = self.normlize(self.sample_weights[:self.num_rates.item()])
            xx = [
                self._conv_forward_dilation(
                    x, (self.rates[i][0].item(), self.rates[i][1].item()))
                * norm_w[i] for i in range(self.num_rates.item())
            ]
        x = xx[0]
        for i in range(1, self.num_rates.item()):
            x += xx[i]
        if self.training:
            self.searcher()
        return x

    def searcher(self):
        self.counter += 1
        if self.counter % self.search_interval == 0 and self.current_search_step < self.max_search_step and self.max_search_step != 0:
            self.counter[0] = 0
            self.current_search_step += 1
            self.estimate()
            self.expand()

    def tensor_to_tuple(self, tensor):
        return tuple([(x[0].item(), x[1].item()) for x in tensor])

    def estimate(self):
        norm_w = self.normlize(self.sample_weights[:self.num_rates.item()])
        print('Estimate dilation {} with weight {}.'.format(
            self.tensor_to_tuple(self.rates[:self.num_rates.item()]), norm_w.detach().cpu().numpy().tolist()))

        sum0, sum1, w_sum = 0, 0, 0
        for i in range(self.num_rates.item()):
            sum0 += norm_w[i].item() * self.rates[i][0].item()
            sum1 += norm_w[i].item() * self.rates[i][1].item()
            w_sum += norm_w[i].item()
        estimated = [value_crop(
            int(round(sum0 / w_sum)),
            self.min_dilation,
            self.max_dilation), value_crop(
            int(round(sum1 / w_sum)),
            self.min_dilation,
            self.max_dilation)]
        self.dilation = tuple(estimated)
        self.padding = (
            get_padding(self.kernel_size[0], self.stride[0], self.dilation[0]),
            get_padding(self.kernel_size[1], self.stride[1], self.dilation[1])
        )
        self.rates[0] = torch.FloatTensor([self.dilation[0], self.dilation[1]])
        self.num_rates[0] = 1
        print('Estimate as {}'.format(self.dilation))

    def expand(self):
        rates = rf_expand(self.dilation, self.expand_rate,
                          self.num_branches,
                          min_dilation=self.min_dilation,
                          max_dilation=self.max_dilation)
        for i, rate in enumerate(rates):
            self.rates[i] = torch.FloatTensor([rate[0], rate[1]])
        self.num_rates[0] = len(rates)
        self.sample_weights.data.fill_(self.init_weight)
        print('Expand as {}'.format(self.rates[:len(rates)].cpu().tolist()))
