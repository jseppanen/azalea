
import logging
import numpy as np
import torch
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data.dataloader import default_collate

#from .datagen import Datagen
from .parallel_datagen import ParallelDatagen as Datagen
from .game import import_game
from .policy import Policy
from .replaybuf import ReplaybufLoader
from .monitor import monitor


class PolicyTrainer:
    def __init__(self, config):
        self.reset(config)

    def reset(self, config):
        np.random.seed(config['seed'])
        torch.manual_seed(config['seed'])
        torch.cuda.manual_seed(config['seed'])
        self.device = torch.device(config['device'])
        self.policy = Policy(config)
        self.policy.initialize(config)
        self.policy.net.train()
        self.datagen = Datagen(self.policy, config)
        gamelib = import_game(config['game'])
        self.sampler = ReplaybufLoader(self.datagen,
                                       randomize=gamelib.random_reflect,
                                       board_size=config['board_size'],
                                       replaybuf_size=config['replaybuf_size'],
                                       replaybuf_resample=config['replaybuf_resample'],
                                       batch_size=config['batch_size'],
                                       shuffle=True,
                                       pin_memory=(self.device.type == 'cuda'),
                                       num_workers=4,  # FIXME hardcoded
                                       collate_fn=pad_moves)
        self.optimizer = optim.SGD(self.policy.net.parameters(),
                                   lr=config['lr_initial'],
                                   momentum=config['momentum'],
                                   weight_decay=config['l2_regularization'])
        self.scheduler = lr_scheduler.StepLR(
            self.optimizer,
            step_size=config['lr_decay_steps'],
            gamma=config['lr_decay'])

    def load_state_dict(self, state):
        """Load model state
        """
        # load weights, ignore optimizer
        self.policy.load_state_dict(state['policy'])

    def load_replaybuf_state_dict(self, state):
        self.sampler.load_state_dict(state)

    def state_dict(self):
        """Return model state
        """
        return {
            'policy': self.policy.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }

    def step(self):
        self.policy.net.train()
        input, target = self.sampler.sample()
        for k in input:
            input[k] = input[k].to(self.device)
        for k in target:
            target[k] = target[k].to(self.device)
        self.optimizer.zero_grad()
        output, loss, metrics = self.policy.net.run(input, target)
        loss.backward()
        metrics['loss'] = loss.item()
        for m in metrics:
            monitor.add_scalar('optim/{}'.format(m), metrics[m])
        self.scheduler.step()
        self.optimizer.step()
        monitor.add_scalar('optim/learning_rate',
                           self.optimizer.param_groups[0]['lr'])
        if monitor.step % 5000 == 0:
            for name, param in self.policy.net.named_parameters():
                monitor.add_histogram('param/' + name, param)
                monitor.add_histogram('grad/' + name, param.grad)
        return metrics

    def close(self):
        self.datagen.close()


def pad(vec, size):
    res = np.zeros(size, dtype=vec.dtype)
    res[:len(vec)] = vec
    return res


def pad_moves(batch):
    # pad legal moves with zeros
    maxlen = max(len(x[0]['moves']) for x in batch)
    for input, target in batch:
        input['moves'] = pad(input['moves'], maxlen)
        target['moves_prob'] = pad(target['moves_prob'], maxlen)
    return default_collate(batch)
