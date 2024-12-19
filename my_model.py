import torch
import torch.nn as nn
import torch.nn.functional as F

class QMix(nn.Module):
    def __init__(self, state_dim, n_agents, action_shape, embed_dim=64, hypernet_embed=64, abs=True):
        super(QMix, self).__init__()

        self.n_agents = n_agents
        self.state_dim = state_dim
        self.action_shape = action_shape

        self.embed_dim = embed_dim
        self.hypernet_embed = hypernet_embed
        self.abs = abs

        # Hypernetwork for generating weights of the first layer
        self.hyper_w_1 = nn.Sequential(
            nn.Linear(self.state_dim, self.hypernet_embed),
            nn.ReLU(),
            nn.Linear(self.hypernet_embed, self.n_agents * self.embed_dim)
        )

        # Hypernetwork for generating biases of the first layer
        self.hyper_b_1 = nn.Linear(self.state_dim, self.embed_dim)

        # Hypernetwork for generating weights of the second layer (now outputs for all actions)
        self.hyper_w_final = nn.Sequential(
            nn.Linear(self.state_dim, self.hypernet_embed),
            nn.ReLU(),
            nn.Linear(self.hypernet_embed, self.embed_dim * self.action_shape)  # Changed output dimension
        )

        # V(s) instead of a bias for the last layers
        self.V = nn.Sequential(
            nn.Linear(self.state_dim, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, self.action_shape) # Changed output dimension
        )

    def forward(self, agent_qs, states):
        """
        Compute q_tot for all actions from the given inputs.
        :param agent_qs: (torch.Tensor) q value inputs into network [batch_size, n_agents, action_shape]
        :param states: (torch.Tensor) state observation [batch_size, state_dim]
        :return q_tot: (torch.Tensor) return q-total [batch_size, action_shape]
        """
        bs = agent_qs.size(0)
        states = states.reshape(-1, self.state_dim)
        agent_qs = agent_qs.view(-1, 1, self.n_agents)

        # First layer
        w1 = self.hyper_w_1(states).abs() if self.abs else self.hyper_w_1(states)
        b1 = self.hyper_b_1(states)
        w1 = w1.view(-1, self.n_agents, self.embed_dim)
        b1 = b1.view(-1, 1, self.embed_dim)
        hidden = F.elu(torch.bmm(agent_qs, w1) + b1)

        # Second layer
        w_final = self.hyper_w_final(states).abs() if self.abs else self.hyper_w_final(states)
        w_final = w_final.view(-1, self.embed_dim, self.action_shape)  # Changed shape

        # State-dependent bias
        v = self.V(states).view(-1, 1, self.action_shape)  # Changed shape

        # Compute final output
        y = torch.bmm(hidden, w_final) + v  # Changed

        # Reshape and return
        q_tot = y.view(bs, self.action_shape)  # Changed shape
        return q_tot