import torch
import torch.nn as nn 
import numpy as np
from typing import List, Optional

class Space_embedding(nn.Module):
    def __init__(self,args):
        super(Space_embedding,self).__init__()
        self.hidden_size = args.hidden_size

        self.modes_dense = nn.Sequential(
            nn.Linear(self.hidden_size,2*self.hidden_size),
            nn.ReLU(),
            nn.LayerNorm(2*self.hidden_size),
            nn.Linear(2*self.hidden_size,self.hidden_size)
            )


    def forward(self,spito_emd):
        batch_size,hidden_size = spito_emd.shape
        traj_emb = self.modes_dense(spito_emd).view(batch_size,-1)#[B,N,H]
        return traj_emb#[B,N,H]

class Temp_embedding(nn.Module):
    def __init__(self,args,num_nodes=64,step=20) :
        super(Temp_embedding,self).__init__()
        self.num_nodes = num_nodes
        self.hidden_size = args.hidden_size
        self.modes_dense = nn.Sequential(
            nn.Linear(self.hidden_size,2*self.hidden_size),
            nn.ReLU(),
            nn.LayerNorm(2*self.hidden_size),
            nn.Linear(2*self.hidden_size,self.hidden_size)
            )

    def forward(self,space_emd):#[B,T,H]
        batch_size,_ =space_emd.shape 
        # space_emd = space_emd.reshape(batch_size,-1)#[B,step*hidden_size]
        space_emd = self.modes_dense(space_emd).reshape(batch_size,-1)#[B,H]
        return space_emd#[B,H]

class mlp(nn.Module):
    def __init__(self,args,pred_horizon = 30) -> None:
        super(mlp,self).__init__()
        self.hidden_size = args.hidden_size
        self.mlp = nn.Sequential(
            nn.Linear(4,args.hidden_size),
            nn.ReLU(),
            nn.Linear(args.hidden_size,args.hidden_size)    #全连接升维 relu激活 全连接
        )
        # self.position_embedding == nn.Parameter(torch.randn(1,1,pred_horizon,args.hidden_size))
    
    def forward(self,feat):
        feat = self.mlp(feat)
        return feat
    
class SingleInputEmbedding(nn.Module):

    def __init__(self,
                 in_channel: int,
                 out_channel: int) -> None:
        super(SingleInputEmbedding, self).__init__()
        self.embed = nn.Sequential(
            nn.Linear(in_channel, out_channel),
            nn.LayerNorm(out_channel),
            nn.ReLU(inplace=True),
            nn.Linear(out_channel, out_channel),
            nn.LayerNorm(out_channel),
            nn.ReLU(inplace=True),
            nn.Linear(out_channel, out_channel),
            nn.LayerNorm(out_channel))
        self.apply(init_weights) #权重初始化

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embed(x)


class MultipleInputEmbedding(nn.Module):

    def __init__(self,
                 in_channels: List[int],
                 out_channel: int) -> None:
        super(MultipleInputEmbedding, self).__init__()
        self.module_list = nn.ModuleList(
            [nn.Sequential(nn.Linear(in_channel, out_channel),
                           nn.LayerNorm(out_channel),
                           nn.ReLU(inplace=True),
                           nn.Linear(out_channel, out_channel))
             for in_channel in in_channels])
        self.aggr_embed = nn.Sequential(
            nn.LayerNorm(out_channel),
            nn.ReLU(inplace=True),
            nn.Linear(out_channel, out_channel),
            nn.LayerNorm(out_channel))
        self.apply(init_weights)

    def forward(self,
                continuous_inputs: List[torch.Tensor],
                categorical_inputs: Optional[List[torch.Tensor]] = None) -> torch.Tensor:
        for i in range(len(self.module_list)):
            continuous_inputs[i] = self.module_list[i](continuous_inputs[i])
        output = torch.stack(continuous_inputs).sum(dim=0)
        if categorical_inputs is not None:
            output += torch.stack(categorical_inputs).sum(dim=0)
        return self.aggr_embed(output)

def init_weights(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        fan_in = m.in_channels / m.groups
        fan_out = m.out_channels / m.groups
        bound = (6.0 / (fan_in + fan_out)) ** 0.5
        nn.init.uniform_(m.weight, -bound, bound)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, mean=0.0, std=0.02)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.MultiheadAttention):
        if m.in_proj_weight is not None:
            fan_in = m.embed_dim
            fan_out = m.embed_dim
            bound = (6.0 / (fan_in + fan_out)) ** 0.5
            nn.init.uniform_(m.in_proj_weight, -bound, bound)
        else:
            nn.init.xavier_uniform_(m.q_proj_weight)
            nn.init.xavier_uniform_(m.k_proj_weight)
            nn.init.xavier_uniform_(m.v_proj_weight)
        if m.in_proj_bias is not None:
            nn.init.zeros_(m.in_proj_bias)
        nn.init.xavier_uniform_(m.out_proj.weight)
        if m.out_proj.bias is not None:
            nn.init.zeros_(m.out_proj.bias)
        if m.bias_k is not None:
            nn.init.normal_(m.bias_k, mean=0.0, std=0.02)
        if m.bias_v is not None:
            nn.init.normal_(m.bias_v, mean=0.0, std=0.02)
    elif isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                for ih in param.chunk(4, 0):
                    nn.init.xavier_uniform_(ih)
            elif 'weight_hh' in name:
                for hh in param.chunk(4, 0):
                    nn.init.orthogonal_(hh)
            elif 'weight_hr' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias_ih' in name:
                nn.init.zeros_(param)
            elif 'bias_hh' in name:
                nn.init.zeros_(param)
                nn.init.ones_(param.chunk(4, 0)[1])
    elif isinstance(m, nn.GRU):
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                for ih in param.chunk(3, 0):
                    nn.init.xavier_uniform_(ih)
            elif 'weight_hh' in name:
                for hh in param.chunk(3, 0):
                    nn.init.orthogonal_(hh)
            elif 'bias_ih' in name:
                nn.init.zeros_(param)
            elif 'bias_hh' in name:
                nn.init.zeros_(param)
                
class emb(nn.Module):
    def __init__(self,args) -> None:
        super(emb,self).__init__()
        self.args = args
        self.hidden_size = args.hidden_size
        self.lm_head = nn.Linear(args.d_model, args.vocab_size, bias=False)
        self.mlp = nn.Sequential(
            nn.Linear(4,args.hidden_size),
            nn.ReLU(),
            nn.Linear(args.hidden_size,args.hidden_size)    #全连接升维 relu激活 全连接
        )
        # self.position_embedding == nn.Parameter(torch.randn(1,1,pred_horizon,args.hidden_size))
    
    def forward(self,input,batch):
        # rotation, origin = batch["rotation"], batch["origin"]

        # Extract the number of agents in each sample of the current batch
        agents_per_sample = batch['agents_per_sample'] #数有几个agent
        gnn_in = [ input[i][:agents_per_sample[i]]  for i in range(len(agents_per_sample)) ] #B个[A,T,H]

        gnn_in = torch.cat(gnn_in,dim=0)
        x = gnn_in
        logits = self.lm_head(x)
        out  =  logits[:,-1,:] 
        max_agents = max(agents_per_sample)
        padded_att_in = torch.zeros((len(agents_per_sample), max_agents, self.hidden_size),device=input.device) #[B,A,H]
        mask = torch.arange(max_agents) < torch.tensor(agents_per_sample)[:, None] #[B,A]将 torch.arange(max_agents) 中的每个元素与 agents_per_sample 中每个样本对应的代理数量进行比较。如果 torch.arange(max_agents) 中的索引小于或等于某个样本的代理数量，则相应位置的结果为 True，否则为 False
        padded_att_in[mask] = out#[B,A,H] bool索引
        output = torch.stack([ x[0] for x in padded_att_in],)
        # max_agents = max(agents_per_sample) 
        # output = torch.zeros((len(agents_per_sample), max_agents, logits.shape[1],self.hidden_size),device=logits[0].device) #0张量[B,A,T,H]
        # mask = torch.arange(max_agents) < torch.tensor(agents_per_sample)[:, None] #[B,A]
        # output[mask] = logits
        return output
 