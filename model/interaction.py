from typing import Optional,List,Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F
import pytorch_lightning as pl
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj
from torch_geometric.typing import OptTensor
from torch_geometric.typing import Size
from torch_geometric.utils import softmax
from torch_geometric.utils import subgraph
from torch_geometric.utils import remove_self_loops
from torch_geometric.nn import conv
from torch_geometric.nn.pool import radius_graph
from torch_geometric.utils import from_scipy_sparse_matrix
from torch_geometric.data import Data
from torch_geometric.data import Batch
from torch.nn.utils.rnn import pad_sequence
from scipy import sparse
import numpy as np  
from model.layer import MultyHeadAttn,att
from model.embedding import SingleInputEmbedding,MultipleInputEmbedding,init_weights
from torch_scatter import scatter
# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


# class Spito_inter(nn.Module):
#     def __init__(self,args):
#         super(Spito_inter,self).__init__()
#         self.args = args
#         self.agent_gnn = AgentGnn(self.args)

#         self.hidden_size = args.hidden_size
#     def forward(self, input ,batch):
#     # Set batch norm to eval mode in order to prevent updates on the running means,
#     # if the weights are frozen


#         displ, centers = batch["displ"], batch["centers"]
#         # rotation, origin = batch["rotation"], batch["origin"]

#         # Extract the number of agents in each sample of the current batch
#         agents_per_sample = [x.shape[0] for x in displ] #数有几个agent

#         # Convert the list of tensors to tensors
#         # displ_cat = torch.cat(displ, dim=0)
#         centers_cat = torch.cat(centers, dim=0) #将所有agent的中心拼到一起 [agent_all,F]
#         reshapeinput = input

#         gnn_in = [ input[i][:agents_per_sample[i]]  for i in range(len(agents_per_sample)) ] #B个[A,T,H] 

#         gnn_in = torch.cat(gnn_in,dim=0) #把B个拼成一个 [total_agent,T,H]
#         gnn_input  =  gnn_in[:,-1,:] #最后一秒所有agent的特征 [total_agent,H]
#         # out_encoder_lstm = self.encoder_lstm(displ_cat, agents_per_sample)
#         out_agent_gnn = self.agent_gnn(gnn_input, centers_cat, agents_per_sample)#[total_agent,H]
#         # out_self_attention = torch.stack([x[0] for x in out_self_attention])
#         max_agents = max(agents_per_sample)
#         padded_att_in = torch.zeros((len(agents_per_sample), max_agents, self.hidden_size),device=out_agent_gnn[0].device) #[B,A,H]
#         mask = torch.arange(max_agents) < torch.tensor(agents_per_sample)[:, None] #[B,A]将 torch.arange(max_agents) 中的每个元素与 agents_per_sample 中每个样本对应的代理数量进行比较。如果 torch.arange(max_agents) 中的索引小于或等于某个样本的代理数量，则相应位置的结果为 True，否则为 False
#         padded_att_in[mask] = out_agent_gnn#[B,A,H] bool索引
#         gnn_out = torch.stack([ x[0] for x in padded_att_in],)#[B,H] 
#         return gnn_out


# class AgentGnn(nn.Module):  #Crystal-GCN 空间交互
#     def __init__(self, args):
#         super(AgentGnn, self).__init__()
#         self.args = args
#         self.latent_size = args.hidden_size
#         self.gcn1 = conv.CGConv(self.latent_size, dim=2, batch_norm=True)
#         self.gcn2 = conv.CGConv(self.latent_size, dim=2, batch_norm=True)   #两层gcn
#     def forward(self, gnn_in, centers, agents_per_sample):
#         # gnn_in is a batch and has the shape (batch_size, number_of_agents, latent_size)

#         x, edge_index = gnn_in, self.build_fully_connected_edge_idx(
#             agents_per_sample).to(gnn_in.device)
#         edge_attr = self.build_edge_attr(edge_index, centers).to(gnn_in.device)

#         edge_attr = torch.tensor(edge_attr,dtype=torch.float32) 
#         x = F.relu(self.gcn1(x, edge_index, edge_attr))
#         # x = F.relu(self.gcn2(x, edge_index, edge_attr))
#         gnn_out = F.relu(self.gcn2(x, edge_index, edge_attr))

#         return gnn_out

#     def build_fully_connected_edge_idx(self, agents_per_sample):
#         edge_index = []

#         # In the for loop one subgraph is built (no self edges!)
#         # The subgraph gets offsetted and the full graph over all samples in the batch
#         # gets appended with the offsetted subgrah
#         offset = 0
#         for i in range(len(agents_per_sample)):

#             num_nodes = agents_per_sample[i]

#             adj_matrix = torch.ones((num_nodes, num_nodes))
#             adj_matrix = adj_matrix.fill_diagonal_(0)

#             sparse_matrix = sparse.csr_matrix(adj_matrix.numpy())
#             edge_index_subgraph, _ = from_scipy_sparse_matrix(sparse_matrix)

#             # Offset the list
#             edge_index_subgraph = torch.Tensor(
#                 np.asarray(edge_index_subgraph) + offset)
#             offset += agents_per_sample[i]

#             edge_index.append(edge_index_subgraph)

#         # Concat the single subgraphs into one
#         edge_index = torch.LongTensor(np.column_stack(edge_index))
#         return edge_index
    
#     def build_edge_attr(self, edge_index, data):
#         edge_attr = torch.zeros((edge_index.shape[-1], 2), dtype=torch.float)

#         rows, cols = edge_index
#         # goal - origin
#         edge_attr = data[cols] - data[rows]

#         return edge_attr
    
class Temp_inter(nn.Module):
    def __init__(self,args):
        super(Temp_inter,self).__init__()
        self.hidden_size = args.hidden_size
        self.mha = MultiheadSelfAttention(args)
        # self.lstm = nn.LSTM()
    def forward(self,mha_in,batch):
        mha_in = torch.stack([x[0] for x in mha_in])#[B,T,H]
        out = self.mha(mha_in)#[B,T,H]
        
        return out[:,-1,:]#[B,H]最后一帧

class MultiheadSelfAttention(nn.Module):
    def __init__(self, args):
        super(MultiheadSelfAttention, self).__init__()
        self.args = args

        self.multihead_attention = nn.MultiheadAttention(self.args.hidden_size, 8)

    def forward(self, att_in):
        padded_att_in_swapped = torch.swapaxes(att_in,0,1)#[T,B,H]
        padded_att_in_swapped, _ = self.multihead_attention(
                padded_att_in_swapped, padded_att_in_swapped, padded_att_in_swapped)
        att_out = torch.swapaxes(padded_att_in_swapped,0,1)#[B,T,H]
        return att_out

class Spito_inter(nn.Module):
    def __init__(self,args):
        super(Spito_inter,self).__init__()
        self.args = args
        self.hidden_size = args.hidden_size
        self.sgnn1 = SGNNLayer(self.args)
        # self.sgnn2 = SGNNLayer(self.args)
        # self.sgnn3 = SGNNLayer(self.args)
        
        self.layer_norm = nn.LayerNorm(self.hidden_size)
        self.residual_proj = nn.Linear(self.hidden_size,self.hidden_size)
        self.grad_clip = 1.0

    def forward(self,input,data):
        B, A, T, _ = input.shape
        sgnn_input = input.view(B * A, T,input.shape[-1])[:,-1,:] #[B*A,H]
        x = self.sgnn1(sgnn_input,data)
        # identity = x
        
        # x = self.sgnn2(x,data)
        # x = x + identity
        # x = self.layer_norm(x)
        
        # identity = x
        # x = self.sgnn3(x,data)
        # x = x + identity
        # x = self.layer_norm(x)
        # torch.nn.utils.clip_grad_norm_(x,self.grad_clip)
        
        sgnn_out = x.view(B, A,  x.shape[-1])
        out = sgnn_out[:,0]
        return out


class SGNNLayer(nn.Module):
    def __init__(self, args,dropout_rate=0.1):
        super(SGNNLayer, self).__init__()
        self.hidden_size = args.hidden_size
        
        self.ln1 = nn.LayerNorm(self.hidden_size)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.sgnn = SGNN(args)
        self.ln2 = nn.LayerNorm(self.hidden_size)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.gelu = nn.GELU()
        self.scale = nn.Parameter(torch.ones(1))
    def forward(self,input,data):
        x = self.ln1(input)
        x = self.dropout1(x)
        x = self.sgnn(x,data)
        x = self.ln2(x)
        x = self.dropout2(x)             
        x = self.gelu(x)
        
        x = x*self.scale
        
        return x

class SGNN(nn.Module):
    def __init__(self,args, n_layer=1, p_step=4, s_dim=128, hidden_dim=128, activation=nn.SiLU(), cutoff=30, gravity_axis=None): 
        super(SGNN, self).__init__()
        self.cutoff = cutoff
        # initialize the networks
        self.embedding = nn.Linear(s_dim, hidden_dim)
        self.embedding1 = nn.Linear(2 * hidden_dim, hidden_dim)
        self.embedding2 = nn.Linear(2 * hidden_dim, hidden_dim)
        self.obj_g_p = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                activation,
                nn.Linear(hidden_dim, 1),
            )
        self.particle_g_p = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                activation,
                nn.Linear(hidden_dim, 1),
            )
        self.gravity_axis = gravity_axis
        self.n_relation = -1
        self.eta = 0.001
        self.local_interaction = SGNNMessagePassingNetwork(n_layer=n_layer, p_step=p_step,
                                                           node_f_dim=3, node_s_dim=hidden_dim,
                                                           edge_f_dim=1, edge_s_dim=0, hidden_dim=hidden_dim,
                                                           activation=activation, edge_readout=True)
        self.object_message_passing = SGNNMessagePassingNetwork(n_layer=n_layer, p_step=p_step,
                                                                node_f_dim=1 if self.gravity_axis is None else 2,
                                                                node_s_dim=hidden_dim,
                                                                edge_f_dim=2, edge_s_dim=hidden_dim, hidden_dim=hidden_dim,
                                                                activation=activation)
        self.object_to_particle = SGNNMessagePassingNetwork(n_layer=n_layer, p_step=p_step,
                                                            node_f_dim=4 if self.gravity_axis is None else 6,
                                                            node_s_dim=hidden_dim,
                                                            edge_f_dim=1, edge_s_dim=0, hidden_dim=hidden_dim,
                                                            activation=activation)

    def build_graph(self, x_p,agents_per_sample):
        edge_index_list = []
        offset = 0
        for num_agents in agents_per_sample:
            edge_index = radius_graph(x_p[offset:offset + num_agents], r=self.cutoff, loop=False)  # 每个场景的图
            edge_index[0] += offset  # 给每个场景的边索引加上偏移，确保每个场景的节点索引不冲突
            edge_index[1] += offset
            edge_index_list.append(edge_index)
            offset += num_agents
        edge_index = torch.cat(edge_index_list, dim=1)  # [2, total_edges]
        self.n_relation = edge_index.shape[1]  # 总的边数
        return edge_index



    def forward(self,  h_p, data):
        # try:
        #     h_p[x_p[..., 1] < 0.1, -1] = 1
        # except:
        #     pass
        agents_per_sample = [x.shape[0] for x in data['displ']]
        s_p = self.embedding(h_p)  # [B, H]
        x_p = pad_sequence(data['x'], batch_first=True).to(s_p.device) #[B,MAXA.T.H]
        v_p = pad_sequence(data['v'], batch_first=True).to(s_p.device)
        obj_id = pad_sequence_with_value(data['ids'], batch_first=True).to(s_p.device) #[B,MAXA]
        B, A, T, _ = x_p.shape
        obj_id = obj_id.view(-1)
        # s_p = s_p.view(B * A, T,s_p.shape[-1])[:,-1,:]
        x_p = x_p.view(B * A, T, 2)[:,-1,:]  # [B * A, T, 2]
        v_p = v_p.view(B * A, T, 2)[:,-1,:]  # [B * A, T, 2]

        f_o = scatter(torch.stack((x_p, v_p), dim=-1), obj_id, dim=0, reduce='mean').to(torch.float32)  # [N_obj, 3, x]
        s_o = scatter(s_p, obj_id, dim=0) * self.eta  # [B * N_obj, H]


        edge_index = self.build_graph(x_p, agents_per_sample) 
        edge_index_inner_mask = obj_id[edge_index[0]] == obj_id[edge_index[1]]
        edge_index_inter_mask = obj_id[edge_index[0]] != obj_id[edge_index[1]]
        edge_index_inner = edge_index[..., edge_index_inner_mask]  # [2, M_in]
        edge_index_inter = edge_index[..., edge_index_inter_mask]  # [2, M_out]

        f_p = torch.stack((x_p, v_p), dim=-1)  # [B*A, 2, 2] -> position and velocity combined
        f_p = torch.cat((f_p - f_o[obj_id], v_p.unsqueeze(-1)), dim=-1).to(torch.float32)  # [B*A, 2, 3] 
        s_p = torch.cat((s_o[obj_id], s_p), dim=-1)  # [B* A, 2H]
        s_p = self.embedding1(s_p)# [B* A, H]
        # Edge attributes (for the interactions between particles in the batch)
        edge_attr_inter_f = (x_p[edge_index_inter[0]] - x_p[edge_index_inter[1]]).unsqueeze(-1)  # [M_out, 2, 1]
        edge_attr_f, edge_attr_s = self.local_interaction(f_p, s_p, edge_index_inter, edge_attr_inter_f)

        # if self.gravity_axis is not None:
        #         g_o = torch.zeros_like(f_o)[..., 0]  # [N_obj, 2] (gravity effect on 2D plane)
        #         g_o[..., self.gravity_axis] = 1
        #         g_o = g_o * self.obj_g_p(s_o)
        #         f_o = torch.cat((f_o, g_o.unsqueeze(-1)), dim=-1)  # [N_obj, 2, 1]


        num_obj = torch.unique(obj_id).shape[0]  # N_obj
        # num_obj = torch.tensor(len(obj_id))
        edge_index_o = get_fully_connected(num_obj, device=obj_id.device, loop=True)  # [2, M_obj]
        edge_mapping = obj_id[edge_index_inter[0]] * num_obj + obj_id[edge_index_inter[1]]  # [M_out]
        edge_attr_o_f = scatter(edge_attr_f, edge_mapping, dim=0, reduce='mean', dim_size=num_obj ** 2)  # [M_obj, 2, 1]
        edge_attr_o_s = scatter(edge_attr_s, edge_mapping, dim=0, reduce='mean', dim_size=num_obj ** 2)  # [M_obj, H]
        edge_pseudo = torch.ones(edge_attr_s.shape[0],device = edge_mapping.device) # [M_, 1]
        count = scatter(edge_pseudo, edge_mapping, dim=0, reduce='sum', dim_size=num_obj ** 2)
        mask = count > 0
        edge_index_o, edge_attr_o_f, edge_attr_o_s = edge_index_o[..., mask], edge_attr_o_f[mask], edge_attr_o_s[mask]       
        f_o_, s_o_ = self.object_message_passing(f_o[..., 1:], s_o, edge_index_o, edge_attr_o_f, edge_attr_o_s)

        edge_attr_inner_f = (x_p[edge_index_inner[0]] - x_p[edge_index_inner[1]]).unsqueeze(-1)  # [M_in, 2, 1]
        f_p_ = torch.cat((f_o_[obj_id], f_p), dim=-1)  # [B*A,2,3]
        s_p_ = torch.cat((s_o_[obj_id], s_p), dim=-1)  # [B * A, 2H]
        s_p_ = self.embedding2(s_p_)  #[B * A, H]


        f_p_, s_p_ = self.object_to_particle(f_p_, s_p_, edge_index_inner, edge_attr_inner_f)
        # h_p_new = torch.cat((s_p_, s_o_), dim=-1)  
        # h_p_new = self.embedding2(s_p_)  #[B*A,H]
        
        # h_p_new = s_p_.view(B, A,  s_p_.shape[-1])
        # out = h_p_new[:,0]
        # v_out = self.predictor(x_p, f_p_, s_p_, obj_id, obj_type=None, num_obj)  # [N, 3]
        return s_p_ #[B*A,H]
    
def get_fully_connected(num_obj, device, loop=True):
    row = torch.arange(num_obj, dtype=torch.long, device=device)
    col = torch.arange(num_obj, dtype=torch.long, device=device)
    row = row.view(-1, 1).repeat(1, num_obj).view(-1)
    col = col.repeat(num_obj)
    edge_index = torch.stack([row, col], dim=0)
    if not loop:
        edge_index = remove_self_loops(edge_index)
    return edge_index

def pad_sequence_with_value(sequences, batch_first=False):
    # 步骤1: 使用 pad_sequence 填充序列
    padded_sequences = pad_sequence(sequences, batch_first=batch_first, padding_value=-1)

    # 步骤2: 获取所有现有的 trackid（除去填充值-1）
    all_ids = padded_sequences[padded_sequences != -1]  # 去除填充部分
    unique_ids = torch.unique(all_ids)  # 获取所有唯一的 ids

    # 步骤3: 找到最小的未使用的整数
    min_unused_value = 0
    while min_unused_value in unique_ids:
        min_unused_value += 1
    
    # 步骤4: 将填充部分替换为最小未使用的整数
    padded_sequences[padded_sequences == -1] = min_unused_value

    return padded_sequences

class BaseMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, activation, residual=False, last_act=False, flat=False):
        super(BaseMLP, self).__init__()
        self.residual = residual
        if flat:
            activation = nn.Tanh()
            hidden_dim = 4 * hidden_dim
        if residual:
            assert output_dim == input_dim
        if last_act:
            self.mlp = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                activation,
                nn.Linear(hidden_dim, hidden_dim),
                activation,
                nn.Linear(hidden_dim, output_dim),
                activation
            )
        else:
            self.mlp = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                activation,
                nn.Linear(hidden_dim, hidden_dim),
                activation,
                nn.Linear(hidden_dim, output_dim)
            )

    def forward(self, x):
        return self.mlp(x) if not self.residual else self.mlp(x) + x


class SGNNMessagePassingLayer(nn.Module):
    def __init__(self, node_f_dim, node_s_dim, edge_f_dim, edge_s_dim, hidden_dim, activation):
        super(SGNNMessagePassingLayer, self).__init__()
        self.node_f_dim, self.node_s_dim, self.edge_f_dim, self.edge_s_dim = node_f_dim, node_s_dim, edge_f_dim, edge_s_dim
        self.hidden_dim = hidden_dim
        self.activation = activation
        self.net = BaseMLP(input_dim=(node_f_dim * 2 + edge_f_dim) ** 2 + node_s_dim * 2 + edge_s_dim,
                           hidden_dim=hidden_dim,
                           output_dim=(node_f_dim * 2 + edge_f_dim) * node_f_dim + node_s_dim,
                           activation=activation,
                           residual=False,
                           last_act=False,
                           flat=False)
        self.self_net = BaseMLP(input_dim=(node_f_dim * 2) ** 2 + node_s_dim * 2,
                                hidden_dim=hidden_dim,
                                output_dim=node_f_dim * 2 * node_f_dim + node_s_dim,
                                activation=activation,
                                residual=False,
                                last_act=False,
                                flat=False)

    def forward(self, f, s, edge_index, edge_f=None, edge_s=None):
        if edge_index.shape[1] == 0:
            f_c, s_c = torch.zeros_like(f), torch.zeros_like(s)
        else:
            _f = torch.cat((f[edge_index[0]], f[edge_index[1]]), dim=-1)
            if edge_f is not None:
                _f = torch.cat((_f, edge_f), dim=-1)  # [M, 3, 2F+Fe]
            _s = torch.cat((s[edge_index[0]], s[edge_index[1]]), dim=-1)
            if edge_s is not None:
                _s = torch.cat((_s, edge_s), dim=-1)  # [M, 2S]
            _f_T = _f.transpose(-1, -2)
            f2s = torch.einsum('bij,bjk->bik', _f_T, _f)  # [M, (2F+Fe), (2F+Fe)]
            f2s = f2s.reshape(f2s.shape[0], -1)  # [M, (2F+Fe)*(2F+Fe)]
            f2s = F.normalize(f2s, p=2, dim=-1)
            f2s = torch.cat((f2s, _s), dim=-1)  # [M, (2F+Fe)*(2F+Fe)+2S+Se]
            c = self.net(f2s)  # [M, (2F+Fe)*F+H]
            # c = scatter(c, edge_index[0], dim=0, reduce='mean', dim_size=f.shape[0])  # [N, (2F+Fe)*F+H]
            f_c, s_c = c[..., :-self.hidden_dim], c[..., -self.hidden_dim:]  # [M, (2F+Fe)*F], [M, H]
            f_c = f_c.reshape(f_c.shape[0], _f.shape[-1], -1)  # [M, 2F+Fe, F]
            f_c = torch.einsum('bij,bjk->bik', _f, f_c)  # [M, 3, F]
            f_c = scatter(f_c, edge_index[0], dim=0, reduce='mean', dim_size=f.shape[0])  # [N, 3, F]
            s_c = scatter(s_c, edge_index[0], dim=0, reduce='mean', dim_size=f.shape[0])  # [N, H]
        # aggregate f_c and f
        temp_f = torch.cat((f, f_c), dim=-1)  # [N, 3, 2F]
        temp_f_T = temp_f.transpose(-1, -2)  # [N, 2F, 3]
        temp_f2s = torch.einsum('bij,bjk->bik', temp_f_T, temp_f)  # [N, 2F, 2F]
        temp_f2s = temp_f2s.reshape(temp_f2s.shape[0], -1)  # [N, 2F*2F]
        temp_f2s = F.normalize(temp_f2s, p=2, dim=-1)
        temp_f2s = torch.cat((temp_f2s, s, s_c), dim=-1)  # [N, 2F*2F+2S]
        temp_c = self.self_net(temp_f2s)  # [N, 2F*F+H]
        temp_f_c, temp_s_c = temp_c[..., :-self.hidden_dim], temp_c[..., -self.hidden_dim:]  # [N, 2F*F], [N, H]
        temp_f_c = temp_f_c.reshape(temp_f_c.shape[0], temp_f.shape[-1], -1)  # [N, 2F, F]
        temp_f_c = torch.einsum('bij,bjk->bik', temp_f, temp_f_c)  # [N, 3, F]
        f_out = temp_f_c
        s_out = temp_s_c
        return f_out, s_out
    

class SGNNEdgeReadoutLayer(nn.Module):
    def __init__(self, node_f_dim, node_s_dim, edge_f_dim, edge_s_dim, hidden_dim, activation, output_f_dim):
        super(SGNNEdgeReadoutLayer, self).__init__()
        self.node_f_dim, self.node_s_dim, self.edge_f_dim, self.edge_s_dim = node_f_dim, node_s_dim, edge_f_dim, edge_s_dim
        self.hidden_dim = hidden_dim
        self.output_f_dim = output_f_dim
        self.net = BaseMLP(input_dim=(node_f_dim * 2 + edge_f_dim) ** 2 + node_s_dim * 2 + edge_s_dim,
                           hidden_dim=hidden_dim,
                           output_dim=(node_f_dim * 2 + edge_f_dim) * output_f_dim + node_s_dim,
                           activation=activation,
                           residual=False,
                           last_act=False,
                           flat=False)

    def forward(self, f, s, edge_index, edge_f=None, edge_s=None):
        if edge_index.shape[1] == 0:
            f_c, s_c = torch.zeros(0, 3, self.output_f_dim).to(f.device), torch.zeros(0, self.node_s_dim).to(s.device)
            return f_c, s_c
        _f = torch.cat((f[edge_index[0]], f[edge_index[1]]), dim=-1)
        if edge_f is not None:
            _f = torch.cat((_f, edge_f), dim=-1)  # [M, 3, 2F+Fe]
        _s = torch.cat((s[edge_index[0]], s[edge_index[1]]), dim=-1)
        if edge_s is not None:
            _s = torch.cat((_s, edge_s), dim=-1)  # [M, 2S]
        _f_T = _f.transpose(-1, -2)
        f2s = torch.einsum('bij,bjk->bik', _f_T, _f)  # [M, (2F+Fe), (2F+Fe)]
        f2s = f2s.reshape(f2s.shape[0], -1)  # [M, (2F+Fe)*(2F+Fe)]
        f2s = F.normalize(f2s, p=2, dim=-1)
        f2s = torch.cat((f2s, _s), dim=-1)  # [M, (2F+Fe)*(2F+Fe)+2S+Se]
        c = self.net(f2s)  # [M, (2F+Fe)*F_out+H]
        f_c, s_c = c[..., :-self.hidden_dim], c[..., -self.hidden_dim:]  # [M, (2F+Fe)*F_out], [M, H]
        f_c = f_c.reshape(f_c.shape[0], _f.shape[-1], -1)  # [M, 2F+Fe, F_out]
        f_c = torch.einsum('bij,bjk->bik', _f, f_c)  # [M, 3, F_out]
        return f_c, s_c


class SGNNMessagePassingNetwork(nn.Module):
    def __init__(self, n_layer, p_step, node_f_dim, node_s_dim, edge_f_dim, edge_s_dim, hidden_dim,
                 activation, edge_readout=False):
        super(SGNNMessagePassingNetwork, self).__init__()
        self.networks = nn.ModuleList()
        self.n_layer = n_layer
        self.p_step = p_step
        for i in range(self.n_layer):
            self.networks.append(SGNNMessagePassingLayer(node_f_dim, node_s_dim, edge_f_dim,
                                                           edge_s_dim, hidden_dim, activation))
        self.edge_readout = edge_readout
        if edge_readout:
            self.readout = SGNNEdgeReadoutLayer(node_f_dim, node_s_dim, edge_f_dim, edge_s_dim, hidden_dim,
                                                  activation, output_f_dim=2)

    def forward(self, f, s, edge_index, edge_f=None, edge_s=None):
        for i in range(self.p_step):
            f, s = self.networks[0](f, s, edge_index, edge_f, edge_s)
        if self.edge_readout:  # edge-level readout
            fe, se = self.readout(f, s, edge_index, edge_f, edge_s)
            return fe, se
        else:  # node-level output
            return f, s
#/home/zzh/anaconda3/pkgs/libstdcxx-devel_linux-64-11.2.0-h1234567_1/x86_64-conda-linux-gnu/lib64/libstdc++.so.6.0.29