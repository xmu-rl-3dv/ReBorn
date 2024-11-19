import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from modules.layers.act_layer import ActivateLayer

class QMixerCentralFF(nn.Module):
    def __init__(self, args):
        super(QMixerCentralFF, self).__init__()

        self.args = args

        self.n_agents = args.n_agents
        self.state_dim = int(np.prod(args.state_shape))

        self.input_dim = self.n_agents * self.args.central_action_embed + self.state_dim
        self.embed_dim = args.central_mixing_embed_dim

        non_lin = nn.ReLU

        self.net = nn.Sequential(nn.Linear(self.input_dim, self.embed_dim),
                                 non_lin(),
                                 ActivateLayer(self.embed_dim, 'CW1'),
                                 nn.Linear(self.embed_dim, self.embed_dim),
                                 non_lin(),
                                 ActivateLayer(self.embed_dim, 'CW2'),
                                 nn.Linear(self.embed_dim, self.embed_dim),
                                 non_lin(),
                                 ActivateLayer(self.embed_dim, 'CW3'),
                                 nn.Linear(self.embed_dim, 1))

        # V(s) instead of a bias for the last layers
        self.V = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim),
                               non_lin(),
                               ActivateLayer(self.embed_dim, 'CV1'),
                               nn.Linear(self.embed_dim, 1))

    def forward(self, agent_qs, states):
        bs = agent_qs.size(0)
        states = states.reshape(-1, self.state_dim)
        agent_qs = agent_qs.reshape(-1, self.n_agents * self.args.central_action_embed)

        inputs = th.cat([states, agent_qs], dim=1)

        advs = self.net(inputs)
        vs = self.V(states)

        y = advs + vs

        q_tot = y.view(bs, -1, 1)
        return q_tot
