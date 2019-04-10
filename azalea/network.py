
import logging

import torch
from torch import nn
from torch.nn import functional as F


def conv3x3(in_chans, out_chans):
    return nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1, bias=False)


def conv1x1(in_chans, out_chans):
    return nn.Conv2d(in_chans, out_chans, kernel_size=1, bias=False)


class Resblock(nn.Module):
    def __init__(self, in_dim, dim):
        super().__init__()
        self.conv1 = conv3x3(in_dim, dim)
        self.bn1 = nn.BatchNorm2d(dim)
        self.conv2 = conv3x3(dim, dim)
        self.bn2 = nn.BatchNorm2d(dim)
        if dim != in_dim:
            self.res_conv = conv1x1(in_dim, dim)
            self.res_bn = nn.BatchNorm2d(dim)
        else:
            self.res_conv = self.res_bn = None
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        y = self.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        # residual connection
        if self.res_conv:
            x = self.res_bn(self.res_conv(x))
        y += x
        y = self.relu(y)
        return y


class Network(nn.Module):
    def __init__(self, board_size, input_dim, num_blocks,
                 base_chans, value_chans, policy_chans):
        super().__init__()
        # input upsampling
        self.conv1 = conv3x3(input_dim, base_chans)
        self.bn1 = nn.BatchNorm2d(base_chans)
        # residual blocks
        blocks = [Resblock(base_chans, base_chans)
                  for i in range(num_blocks)]
        self.resblocks = nn.Sequential(*blocks)
        # value head
        self.value_conv1 = conv1x1(base_chans, value_chans)
        self.value_bn1 = nn.BatchNorm2d(value_chans)
        self.value_fc2 = nn.Linear(value_chans * board_size * board_size, 64)
        self.value_fc3 = nn.Linear(64, 1)
        # policy head
        self.move_conv1 = conv1x1(base_chans, policy_chans)
        self.move_bn1 = nn.BatchNorm2d(policy_chans)
        self.relu = nn.ReLU(inplace=True)

    @property
    def device(self):
        """Get current device of model."""
        return self.conv1.weight.device

    def forward(self, x):
        """
        :param x: Batch of game boards (batch x height x width, int32)
        """
        # upsample
        x = self.relu(self.bn1(self.conv1(x)))
        # residual blocks
        x = self.resblocks(x)
        # value head
        v = self.relu(self.value_bn1(self.value_conv1(x)))
        v = v.view(v.size(0), -1)
        v = self.relu(self.value_fc2(v))
        v = self.value_fc3(v)
        value = torch.tanh(v).squeeze(1)
        # policy head
        p = self.relu(self.move_bn1(self.move_conv1(x)))
        p = p.view(p.size(0), -1)
        return value, p

    def run(self, batch, *, compute_loss=False):
        """
        :param input: batch of board and moves (padded)
        """
        output = self.forward(batch['board'], batch['legal_moves'])
        if compute_loss:
            moves_prob = batch['moves_prob']
            # Eq. (1) without regularization (done with weight_decay)
            value_loss = F.mse_loss(output['value'], batch['reward'])
            moves_loss = -(moves_prob * output['moves_logprob']).sum() \
                / len(moves_prob)
            loss = value_loss.to(moves_loss.device) + moves_loss
            output = dict((k, v.detach()) for k, v in output.items())
            output.update(value_loss=value_loss.item(),
                          moves_loss=moves_loss.item())
            return output, loss
        else:
            output = dict((k, v.detach()) for k, v in output.items())
            return output

    def load(self, modelpath):
        state = torch.load(modelpath)
        self.load_state_dict(state['model'])
        return state['optimizer']

    def save(self, modelpath, optimizer):
        state = {
            'model': self.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        torch.save(state, modelpath)


class HexNetwork(Network):
    def __init__(self, board_size=11, num_blocks=6, base_chans=64):
        super().__init__(board_size, input_dim=4, num_blocks=num_blocks,
                         base_chans=base_chans, value_chans=2, policy_chans=4)
        # tile encoder
        self.encoder = nn.Embedding(3, 4)
        # policy head
        self.move_fc = nn.Linear(4 * board_size * board_size,
                                 board_size * board_size)
        nnet = sum(p.nelement() for p in self.parameters())
        nenc = sum(p.nelement() for p in self.encoder.parameters())
        logging.info('Net params: {}'.format(nnet - nenc))
        logging.info('Embedding params: {}'.format(nenc))

    def forward(self, x, legal_moves):
        """
        legal_moves padded with zeros
        :param x: Batch of game boards (batch x height x width, int32)
        :param legal_moves: Batch of legal moves (batch x MAX_MOVES, int32)
        """
        # piece encoder
        x = self.encoder(x.long())  # batch x height x width x 4
        x = x.permute(0, 3, 1, 2)
        # resnet
        value, p = super().forward(x)
        # policy head
        moves_logit = self.move_fc(p)
        legal_tiles = (legal_moves - 1).clamp(min=0)
        moves_logit = torch.gather(moves_logit, 1, legal_tiles.long())
        # clear padding
        moves_logit.masked_fill_(legal_moves == 0, -99)
        moves_logprob = F.log_softmax(moves_logit, dim=1)
        return dict(value=value, moves_logprob=moves_logprob)
