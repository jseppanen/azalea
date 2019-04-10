
# https://en.wikipedia.org/wiki/Bradley%E2%80%93Terry_model

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import grad
from scipy.optimize import minimize


def lossgrad(scores, outcomes):

    # prior: player #0 is at 0 elo
    scores[0] = 0.0

    # gamma = 10^(elo/400)
    # convert to natural base
    scores = scores * np.log(10) / 400
    scores = torch.tensor(scores, requires_grad=True)

    loss = torch.tensor(0.0, dtype=torch.float64)

    # likelihoods
    for i, j in outcomes:
        pair = torch.cat([scores[i, None], scores[j, None]])
        logprob = F.log_softmax(pair, 0)
        o = outcomes[i, j]
        loss += int(o[0]) * -logprob[0]
        loss += int(o[2]) * -logprob[1]
        assert o[1] == 0, 'draws not supported'

    g, = grad(loss, (scores,))
    l, g = loss.item(), g.numpy()

    # keep player #0 at 0 elo
    g[0] = 0.0

    #print(l, (g**2).sum() ** .5)
    return l, g


class RankingError(Exception):
    pass


def compute_ranking(num_players, outcomes):
    """Compute Elo ranking from match outcomes.
    Outcomes are represented as a dict, where each pair of player ids
    (p1, p2) maps to triplet of (p1_wins, draws, p1_losses).
    :param outcomes: match results
    :returns: array of Elo scores
    """
    res = minimize(lossgrad, np.zeros(num_players), args=(outcomes,),
                   method='L-BFGS-B', jac=True)
    if not res.success:
        raise RankingError('did not converge')
    #print('loss', res.fun)
    return res.x


if __name__ == '__main__':
    '''
    0 1000
    1 10000
    2 50000
    3 70000
    4 80000
    5 141000
    '''

    # (p1, p2): (p1 wins, draws, p1 losses)
    outcomes = {
        (5, 3): (74, 0, 26),
        (5, 5): (42, 0, 58),
        (4, 3): (66, 0, 34),
        (5, 0): (100, 0, 0),
        (5, 1): (100, 0, 0),
        (4, 2): (74, 0, 26),
    }

    scores = compute_ranking(6, outcomes)
    print(scores)
