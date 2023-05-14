from discriminators.base import Discriminator
from networks.base import Network
import torch
import torch.nn as nn

class SQIL(Discriminator):
    def __init__(self, writer, device, state_dim, action_dim, discriminator_args):
        super(SQIL, self).__init__()
        self.writer = writer
        self.device = device
        self.args = discriminator_args
        #lambda is not used
    def forward(self,x):
        pass

    def get_reward(self,*value):
        return torch.tensor(0)

    def train_network(self, brain, n_epi, agent_s, agent_a, agent_next_s, agent_done_mask, expert_s, expert_a, expert_next_s, expert_done_mask):
        #print("agent", agent_a.shape)
        #print("expert", expert_a.shape)
        expert_s = expert_s.cuda()
        expert_a = expert_a.cuda()
        states = torch.cat((agent_s, expert_s),0)
        #print(agent_a.shape)
        #print(expert_a.shape)
        actions = torch.cat((agent_a, expert_a),0)

        agent_r = torch.zeros((agent_s.shape[0],1)).to(self.device)
        expert_r = torch.ones((expert_s.shape[0],1)).to(self.device)
        rewards = torch.cat((agent_r, expert_r),0)
        expert_next_s = expert_next_s.cuda()
        expert_done_mask = expert_done_mask.cuda()
        next_states = torch.cat((agent_next_s, expert_next_s),0)
        done_masks = torch.cat((agent_done_mask, expert_done_mask),0)
        
        brain.train_network(self.writer, n_epi, states, actions, rewards, next_states, done_masks)
