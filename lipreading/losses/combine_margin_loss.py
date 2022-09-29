import inspect

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.autograd import Function
from torch.nn import Parameter


class CombineMarginLinear(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 scale=1.0,
                 m1=1.0,
                 m2=0.0,
                 m3=0.0,
                 input_norm=True,
                 support_vector=False,
                 t=0.2,
                 adaptive=False,
                 combine=False,
                 base=1000,
                 gamma=0.000015,
                 power=5,
                 lambda_min=5,
                 calculate_loss=False,
                 **kwargs):
        super(CombineMarginLinear, self).__init__()
        self.input_norm = input_norm
        self.s = scale
        self.m1 = m1
        self.m2 = m2
        self.m3 = m3
        self.support_vector = support_vector
        self.t = t
        self.adaptive = adaptive

        self.combine = combine
        self.base = base
        self.gamma = gamma
        self.power = power
        self.lambda_min = lambda_min

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = Parameter(
            torch.Tensor(self.out_channels, self.in_channels))
        self.weight.data.normal_(0, 0.01)
        self.calculate_loss = calculate_loss
        if self.calculate_loss:
            self.ce = nn.CrossEntropyLoss()
        self._register_state_dict_hook(self._state_dict_hook)
        self.iter = 0

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        if prefix + 'weight' in state_dict:
            weight = state_dict[prefix + 'weight']
            if self.weight is None:
                object.__setattr__(self, '_loaded_weight', weight)
                print('Init CombineMarginLinear weight later')
            else:
                self.weight.data.copy_(weight.cuda())
                print('Load CombineMarginLinear weight successfully.')
        else:
            print('No CombineMarginLinear weight in state_dict.')
        # load iter from state dict
        if prefix + 'iter' in state_dict:
            self.iter = state_dict[prefix + 'iter'].item()
            print(f'[CbMarginMP] resume from iter {self.iter}')

    def _state_dict_hook(self, module, state_dict, prefix, local_metadata):
        # save current iter into state dict
        state_dict[prefix + 'iter'] = torch.LongTensor([self.iter]).cuda()
        print(f'CombineMarginLinear iteration: {self.iter}')

    def forward(self, input, target=None):
        if self.training:
            ex = input.renorm(2, 0, 1e-5).mul(1e5)
            ew = self.weight.renorm(2, 0, 1e-5).mul(1e5)
            cos_theta = torch.mm(ex, ew.t()).clamp(-1.0, 1.0)
            if self.m1 != 1.0 or self.m2 != 0.0 or self.m3 != 0.0:
                target_view = target.view(-1, 1)
                diff = cos_theta.gather(1, target_view)
                output = diff * 1.0
                if self.m1 != 1.0 or self.m2 != 0.0:
                    angle = torch.acos(diff)
                    angle = angle * self.m1 + self.m2
                    k = (angle / 3.14159265).floor()
                    # diff = ((-1.0) ** k) * torch.cos(angle) - 2 * k
                    coef = (k + 1) % 2
                    coef.masked_fill_(coef == 0, -1)
                    diff = coef * torch.cos(angle) - 2 * k
                diff = diff - self.m3

                if self.combine:
                    self.lamb = max(
                        self.lambda_min,
                        self.base / (1 + self.gamma * self.iter)**self.power)
                    output = output - output * (1.0 + 0) / (1 + self.lamb)
                    output = output + diff * (1.0 + 0) / (1 + self.lamb)
                    cos_theta = cos_theta.scatter(1, target_view, output)
                else:
                    cos_theta = cos_theta.scatter(1, target_view, diff)
                if self.support_vector:
                    gt_score = cos_theta.gather(index=target.view(-1, 1),
                                                dim=1)
                    mask = cos_theta > gt_score.view(-1, 1)
                    hard_vector = cos_theta[mask]
                    if self.adaptive:
                        cos_theta[mask] = (self.t + 1.0) * hard_vector + self.t
                    else:
                        cos_theta[mask] = hard_vector + self.t
            self.iter += 1
        else:
            ex = input.renorm(2, 0, 1e-5).mul(1e5)
            ew = self.weight.renorm(2, 0, 1e-5).mul(1e5)
            cos_theta = torch.mm(ex, ew.t()).clamp(-1.0 + 1e-6, 1.0 - 1e-6)

        if self.input_norm:
            output = cos_theta * self.s
        else:
            output = cos_theta * input.norm(2, 1, keepdim=True)
        if self.calculate_loss:
            output = self.ce(output, target)
        return output

    def __repr__(self):
        return ('{name}(in_channels={in_channels}, '
                'out_channels={out_channels}, '
                'scale={s}, m1={m1}, m2={m2}, m3={m3}, '
                'input_norm={input_norm}, '
                'support_vector={support_vector}, t={t}, '
                'adaptive={adaptive}, '
                'base={base}, gamma={gamma}, power={power}, '
                'lambda_min={lambda_min}, combine={combine}, '
                'calculate_loss={calculate_loss})'.format(
                    name=self.__class__.__name__, **self.__dict__))
