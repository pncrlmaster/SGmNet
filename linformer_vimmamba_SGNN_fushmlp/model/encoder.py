from __future__ import annotations
import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from einops import rearrange, repeat, einsum
from typing import Union

import torch 
import torch.nn as nn
import numpy as np
from model.layer import att  
# from layer import att   

from .layer import EncoderLstm
from .vim_mamba import init_weights, create_block
from timm.models.layers import DropPath, to_2tuple
try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


class Temp_encoder(nn.Module):  #时间自注意力 关注哪些车更加重要
    def __init__(self,args,sub_layers = 2) :
        super(Temp_encoder,self).__init__() #调用父类（nn.Module）的初始化方法
        
        self.hidden_size = args.hidden_size 
        self.temp_encoder = nn.ModuleList(
            [att(d_model=args.hidden_size) for _ in range(sub_layers)] #建立一个sub_layers层的注意力层
        )

    
    def forward(self,input,batch): #forward传播
        time_mask = batch['time_mask']
        temp_out = input
        for layer in self.temp_encoder:
            temp_out = layer(temp_out,temp_out,temp_out,time_mask) #QKV都是temp_out time_mask具体做什么作用？
        return temp_out



class Spito_encoder(nn.Module): #空间自注意力 关注哪些时间更加重要
    def __init__(self,args,sub_layers = 2) :
        super(Spito_encoder,self).__init__()
        self.hidden_size = args.hidden_size

        self.spito_encoder = nn.ModuleList(
            [att(d_model=args.hidden_size) for _ in range(sub_layers)]
        )      
    
    def forward(self,input,batch):
        space_mask = batch['agent_mask']
        spito_out = input.permute(0,2,1,3)  #交换21顺序 为什么要交换
        for layer in self.spito_encoder:
            spito_out = layer(spito_out,spito_out,spito_out,space_mask) 
        return spito_out.permute(0,2,1,3)


class Temp_mamba(nn.Module):
    def __init__(self,args):
        super(Temp_mamba,self).__init__()
        embed_dim = args.hidden_size
        drop_path=0.2
        
        self.hist_embed_mlp = nn.Sequential(
            nn.Linear(args.hidden_size, args.hidden_size),
            nn.GELU(),
            nn.Linear(args.hidden_size, embed_dim),
        )

        # Agent Encoding Mamba
        self.hist_embed_mamba = nn.ModuleList(  
            [
                create_block(  
                    d_model=embed_dim,
                    layer_idx=i,
                    drop_path=0.2,  
                    bimamba=False,  
                    rms_norm=True,  
                )
                for i in range(4)
            ]
        )
        self.norm_f = RMSNorm(embed_dim, eps=1e-5)
        self.drop_path = DropPath(drop_path)

        # self.pos_embed = nn.Sequential(
        #     nn.Linear(4, embed_dim),
        #     nn.GELU(),
        #     nn.Linear(embed_dim, embed_dim),
        # )
    


    def forward(self,mha_in,batch):
        # unidirectional mamba
        # actor_feat = self.hist_embed_mlp(hist_feat[hist_feat_key_valid].contiguous())
        hist_valid_mask = batch["mamba_mask"]
        hist_key_valid_mask = hist_valid_mask.any(-1)
        # hist_feat = torch.cat(
        #     [
        #         data["x_positions_diff"],
        #         data["x_velocity_diff"][..., None],
        #         hist_valid_mask[..., None],
        #     ],
        #     dim=-1,
        # )

        B, N, L, D = mha_in.shape
        hist_feat = mha_in.view(B * N, L, D)
        hist_feat_key_valid = hist_key_valid_mask.view(B * N)
        actor_feat = self.hist_embed_mlp(hist_feat[hist_feat_key_valid].contiguous())
        residual = None
        for blk_mamba in self.hist_embed_mamba:
            actor_feat, residual = blk_mamba(actor_feat, residual)
        fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
        actor_feat = fused_add_norm_fn(
            self.drop_path(actor_feat),
            self.norm_f.weight,
            self.norm_f.bias,
            eps=self.norm_f.eps,
            residual=residual,
            prenorm=False,
            residual_in_fp32=True  
        )

        # actor_feat = actor_feat[:, -1]
        actor_feat_tmp = torch.zeros(
            B * N, actor_feat.shape[-2],actor_feat.shape[-1], device=actor_feat.device
        )
        actor_feat_tmp[hist_feat_key_valid] = actor_feat
        actor_feat = actor_feat_tmp.view(B, N, actor_feat.shape[-2], actor_feat.shape[-1])
        # out = torch.stack([x[0] for x in actor_feat])#[B,20,H]
        ag_feat = torch.stack([x[0] for x in actor_feat])
        
        return ag_feat[:,-1,:]





# 
            
      
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from einops import rearrange
# from typing import Union
# # 定义模型参数

# # class ModelArgs:
# #     d_model: int  # 模型的隐藏层维度
# #     n_layer: int  # 模型的层数
# #     vocab_size: int  # 词汇表的大小
# #     d_state: int = 16  # 状态空间的维度，默认为16
# #     expand: int = 2  # 扩展因子，默认为2
# #     dt_rank: Union[int, str] = 'auto'  # 输入依赖步长Δ的秩，'auto'表示自动设置
# #     d_conv: int = 4  # 卷积核的维度，默认为4
# #     pad_vocab_size_multiple: int = 8  # 词汇表大小的最小公倍数，默认为8
# #     conv_bias: bool = True  # 卷积层是否使用偏置项
# #     bias: bool = False  # 其他层（如线性层）是否使用偏置项

# #     def __post_init__(self):
# #         self.d_inner = int(self.expand * self.d_model)  # 计算内部维度
# #         if self.dt_rank == 'auto':
# #             self.dt_rank = math.ceil(self.d_model / 16)  # 自动计算Δ的秩
# #         if self.vocab_size % self.pad_vocab_size_multiple != 0:
# #             self.vocab_size += (self.pad_vocab_size_multiple - self.vocab_size % self.pad_vocab_size_multiple)  # 调整vocab_size

# # Mamba模型的残差块
# class ResidualBlock(nn.Module):
#     def __init__(self, args):
#         super().__init__()
#         self.args = args
#         self.mixer = MambaBlock(args)  # Mamba块
#         self.norm = RMSNorm(args.d_model)  # RMS归一化

#     def forward(self, x):
#         # x: shape (b, l, d) (batch size, sequence length, hidden dim)
#         output = self.mixer(self.norm(x)) + x  # Norm -> Mamba -> Add
#         return output

# # Mamba模型
# class Mamba(nn.Module):
#     def __init__(self, args):
#         super().__init__()
#         self.args = args
#         self.embedding = nn.Linear(args.vocab_size, args.d_model)  # 词嵌入层
#         self.layers = nn.ModuleList([ResidualBlock(args) for _ in range(args.n_layer)])  # n_layer个ResidualBlock
#         self.norm_f = RMSNorm(args.d_model)  # RMS归一化
#         self.lm_head = nn.Linear(args.d_model, args.vocab_size, bias=False)  # 输出层
#         self.lm_head.weight = self.embedding.weight  # 权重共享

#     def forward(self, input_ids,batch):
#         displ = batch["displ"]
#         # rotation, origin = batch["rotation"], batch["origin"]

#         # Extract the number of agents in each sample of the current batch
#         agents_per_sample = [x.shape[0] for x in displ] #数有几个agent

#         # Convert the list of tensors to tensors
#         # displ_cat = torch.cat(displ, dim=0)
#         # centers_cat = torch.cat(centers, dim=0) #将所有agent的中心拼到一起 [agent_all,F]


#         gnn_in = [ input_ids[i][:agents_per_sample[i]]  for i in range(len(agents_per_sample)) ] #B个[A,T,H]

#         gnn_in = torch.cat(gnn_in,dim=0)
#         x = gnn_in
#         x = self.embedding(x)  # (b, l, d) 生成词嵌入
#         for layer in self.layers:
#             x = layer(x)  # (b, l, d)
#         x = self.norm_f(x)  # (b, l, d)
#         logits = self.lm_head(x)  # (b, l, vocab_size)
#         max_agents = max(agents_per_sample) 
#         output = torch.zeros((len(agents_per_sample), max_agents, logits.shape[1],logits.shape[2]),device=logits[0].device) #0张量[B,A,T,H]
#         mask = torch.arange(max_agents) < torch.tensor(agents_per_sample)[:, None] #[B,A]
#         output[mask] = logits
#         return output

# # Mamba模型的核心块
# class MambaBlock(nn.Module):
#     def __init__(self, args ):
#         super().__init__()
#         self.args = args
#         self.in_proj = nn.Linear(args.d_model, 2 * args.d_inner)  # 输入投影层
#         self.conv1d = nn.Conv1d(args.d_inner, args.d_inner, kernel_size=args.d_conv, bias=args.conv_bias)  # 深度卷积层
#         self.out_proj = nn.Linear(args.d_inner, args.d_model)  # 输出投影层

#     def forward(self, x):
#         # x: shape (b, l, d) (batch size, sequence length, hidden dim)
#         x_and_res = self.in_proj(x)  # (b, l, 2 * d_in)
#         (x, res) = x_and_res.split(split_size=[self.args.d_inner, self.args.d_inner], dim=-1)  # 分割为x和res
#         x = rearrange(x, 'b l d_in -> b d_in l')  # 调整x的形状
#         x = self.conv1d(x)[:, :, :x.size(1)]  # (b, d_in, l) 应用深度卷积
#         x = rearrange(x, 'b d_in l -> b l d_in')  # 再次调整x的形状
#         x = F.silu(x)  # 应用SiLU激活函数
#         y = self.ssm(x)  # 运行状态空间模型
#         y = y * F.silu(res)  # 将res的SiLU激活结果与y相乘
#         output = self.out_proj(y)  # (b, l, d)
#         return output

#     def ssm(self, x):
#         # 这里简化了状态空间模型的实现，实际中会更复杂
#         # x: shape (b, l, d_in) (batch size, sequence length, hidden dim)
#         # 这里只是一个示例，实际的SSM实现会涉及更多的矩阵操作
#         return x  # 简化版，直接返回输入x

# # RMS归一化层
# class RMSNorm(nn.Module):
#     def __init__(self, d_model: int, eps: float = 1e-5):
#         super().__init__()
#         self.eps = eps
#         self.weight = nn.Parameter(torch.ones(d_model))

#     def forward(self, x):
#         output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
#         return output


# @dataclass
# class ModelArgs:
#     d_model : int
#     n_layer: int
#     vocab_size: int
#     d_state: int = 16
#     expand: int = 2
#     dt_rank: Union[int, str] = 'auto'
#     d_conv: int = 4 
#     pad_vocab_size_multiple: int = 8
#     conv_bias: bool = True
#     bias: bool = False
    
#     def __post_init__(self):
#         self.d_inner = int(self.expand * self.d_model)
        
#         if self.dt_rank == 'auto':
#             self.dt_rank = math.ceil(self.d_model / 16)
            
#         if self.vocab_size % self.pad_vocab_size_multiple != 0:
#             self.vocab_size += (self.pad_vocab_size_multiple
#                                 - self.vocab_size % self.pad_vocab_size_multiple) 
# class Mamba(nn.Module):
#     def __init__(self, args,n_layer = 4): 
#         """Full Mamba model."""
#         super().__init__()
#         self.args = args
#         self.hidden_size = args.hidden_size 

#         # self.embedding = nn.Embedding(args.vocab_size, self.hidden_size)  
#         self.layers = nn.ModuleList([ResidualBlock(args) for _ in range(n_layer)]) #这些残差块可以被视为对输入数据进行多次变换的层，用于捕捉输入序列中的时空特征
#         self.norm_f = RMSNorm(args.d_model)

#         self.lm_head = nn.Linear(args.d_model, args.vocab_size, bias=False)
#         # self.lm_head.weight = self.embedding.weight  # Tie output projection to embedding weights. 权重绑定
#         #                                              # See "Weight Tying" paper


#     def forward(self, input, batch):
#         """
#         Args:
#             input_ids (long tensor): shape (b, l)    (See Glossary at top for definitions of b, l, d_in, n...)
    
#         Returns:
#             logits: shape (b, l, vocab_size)

#         Official Implementation:
#             class MambaLMHeadModel, https://github.com/state-spaces/mamba/blob/main/mamba_ssm/models/mixer_seq_simple.py#L173

#         """
#         # x = self.embedding(input_ids)
#         displ = batch["displ"]
#         # rotation, origin = batch["rotation"], batch["origin"]

#         # Extract the number of agents in each sample of the current batch
#         agents_per_sample = [x.shape[0] for x in displ] #数有几个agent

#         # Convert the list of tensors to tensors
#         # displ_cat = torch.cat(displ, dim=0)
#         # centers_cat = torch.cat(centers, dim=0) #将所有agent的中心拼到一起 [agent_all,F]


#         gnn_in = [ input[i][:agents_per_sample[i]]  for i in range(len(agents_per_sample)) ] #B个[A,T,H]

#         gnn_in = torch.cat(gnn_in,dim=0)
#         x = gnn_in
#         for layer in self.layers:
#             x = layer(x)
            
#         x = self.norm_f(x) #[total_agent,T,H]
#         logits = self.lm_head(x) #[total_agent,T,vocal_size]
#         max_agents = max(agents_per_sample) 
#         output = torch.zeros((len(agents_per_sample), max_agents, logits.shape[1],self.hidden_size),device=logits[0].device) #0张量[B,A,T,H]
#         mask = torch.arange(max_agents) < torch.tensor(agents_per_sample)[:, None] #[B,A]
#         output[mask] = logits
#         return output

    
#     # @staticmethod
#     # def from_pretrained(pretrained_model_name: str):
#     #     """Load pretrained weights from HuggingFace into model.
    
#     #     Args:
#     #         pretrained_model_name: One of
#     #             * 'state-spaces/mamba-2.8b-slimpj'
#     #             * 'state-spaces/mamba-2.8b'
#     #             * 'state-spaces/mamba-1.4b'
#     #             * 'state-spaces/mamba-790m'
#     #             * 'state-spaces/mamba-370m'
#     #             * 'state-spaces/mamba-130m'
                            
#     #     Returns:
#     #         model: Mamba model with weights loaded
    
#     #     """
#     #     from transformers.utils import WEIGHTS_NAME, CONFIG_NAME
#     #     from transformers.utils.hub import cached_file
        
#     #     def load_config_hf(model_name):
#     #         resolved_archive_file = cached_file(model_name, CONFIG_NAME,
#     #                                             _raise_exceptions_for_missing_entries=False)
#     #         return json.load(open(resolved_archive_file))
        
        
#     #     def load_state_dict_hf(model_name, device=None, dtype=None):
#     #         resolved_archive_file = cached_file(model_name, WEIGHTS_NAME,
#     #                                             _raise_exceptions_for_missing_entries=False)
#     #         return torch.load(resolved_archive_file, weights_only=True, map_location='cpu', mmap=True)
        
#     #     config_data = load_config_hf(pretrained_model_name)
#     #     args = ModelArgs(
#     #         d_model=config_data['d_model'],
#     #         n_layer=config_data['n_layer'],
#     #         vocab_size=config_data['vocab_size']
#     #     )
#     #     model = Mamba(args)
        
#     #     state_dict = load_state_dict_hf(pretrained_model_name)
#     #     new_state_dict = {}
#     #     for key in state_dict:
#     #         new_key = key.replace('backbone.', '')
#     #         new_state_dict[new_key] = state_dict[key]
#     #     model.load_state_dict(new_state_dict)
        
#     #     return model


# class ResidualBlock(nn.Module): #为Mamba Block 添加 normalization 和 残差连接 
#     def __init__(self, args):
#         """Simple block wrapping Mamba block with normalization and residual connection."""
#         super().__init__()
#         self.args = args
#         self.mixer = MambaBlock(args)
#         self.norm = RMSNorm(args.hidden_size)
        

#     def forward(self, x):
#         """
#         Args:
#             x: shape (b, l, d)    (See Glossary at top for definitions of b, l, d_in, n...)
    
#         Returns:
#             output: shape (b, l, d)

#         Official Implementation:
#             Block.forward(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py#L297
            
#             Note: the official repo chains residual blocks that look like
#                 [Add -> Norm -> Mamba] -> [Add -> Norm -> Mamba] -> [Add -> Norm -> Mamba] -> ...
#             where the first Add is a no-op. This is purely for performance reasons as this
#             allows them to fuse the Add->Norm.

#             We instead implement our blocks as the more familiar, simpler, and numerically equivalent
#                 [Norm -> Mamba -> Add] -> [Norm -> Mamba -> Add] -> [Norm -> Mamba -> Add] -> ....
            
#         """
#         output = self.mixer(self.norm(x)) + x

#         return output
            

# class MambaBlock(nn.Module):
#     def __init__(self, args):
#         """A single Mamba block, as described in Figure 3 in Section 3.4 in the Mamba paper [1]."""
#         super().__init__()
#         self.args = args
#         args.d_model = args.hidden_size
#         self.in_proj = nn.Linear(args.hidden_size, args.hidden_size * 4, bias=False)

#         self.conv1d = nn.Conv1d(
#             in_channels=args.d_inner,
#             out_channels=args.d_inner,
#             bias=args.conv_bias,
#             kernel_size=args.d_conv,
#             groups=args.d_inner,
#             padding=args.d_conv - 1,
#         )

#         # x_proj takes in `x` and outputs the input-specific Δ, B, C
#         self.x_proj = nn.Linear(args.d_inner, args.dt_rank + args.d_state * 2, bias=False)
        
#         # dt_proj projects Δ from dt_rank to d_in
#         self.dt_proj = nn.Linear(args.dt_rank, args.d_inner, bias=True)

#         A = repeat(torch.arange(1, args.d_state + 1), 'n -> d n', d=args.d_inner)
#         self.A_log = nn.Parameter(torch.log(A))
#         self.D = nn.Parameter(torch.ones(args.d_inner))
#         self.out_proj = nn.Linear(args.d_inner, args.d_model, bias=args.bias)
        

#     def forward(self, x):
#         """Mamba block forward. This looks the same as Figure 3 in Section 3.4 in the Mamba paper [1].
    
#         Args:
#             x: shape (b, l, d)    (See Glossary at top for definitions of b, l, d_in, n...)
    
#         Returns:
#             output: shape (b, l, d)
        
#         Official Implementation:
#             class Mamba, https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py#L119
#             mamba_inner_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L311
            
#         """
#         (b, l, d) = x.shape
        
#         x_and_res = self.in_proj(x)  # shape (b, l, 2 * d_in)
#         (x, res) = x_and_res.split(split_size=[self.args.d_inner, self.args.d_inner], dim=-1)

#         x = rearrange(x, 'b l d_in -> b d_in l')
#         x = self.conv1d(x)[:, :, :l]
#         x = rearrange(x, 'b d_in l -> b l d_in')
        
#         x = F.silu(x)

#         y = self.ssm(x)
        
#         y = y * F.silu(res)
        
#         output = self.out_proj(y)

#         return output

    
#     def ssm(self, x): #状态空间模型
#         """Runs the SSM. See:
#             - Algorithm 2 in Section 3.2 in the Mamba paper [1]
#             - run_SSM(A, B, C, u) in The Annotated S4 [2]

#         Args:
#             x: shape (b, l, d_in)    (See Glossary at top for definitions of b, l, d_in, n...)
    
#         Returns:
#             output: shape (b, l, d_in)

#         Official Implementation:
#             mamba_inner_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L311
            
#         """
#         (d_in, n) = self.A_log.shape

#         # Compute ∆ A B C D, the state space parameters.
#         #     A, D are input independent (see Mamba paper [1] Section 3.5.2 "Interpretation of A" for why A isn't selective)
#         #     ∆, B, C are input-dependent (this is a key difference between Mamba and the linear time invariant S4,
#         #                                  and is why Mamba is called **selective** state spaces)
        
#         A = -torch.exp(self.A_log.float())  # shape (d_in, n)
#         D = self.D.float()

#         x_dbl = self.x_proj(x)  # (b, l, dt_rank + 2*n)
        
#         (delta, B, C) = x_dbl.split(split_size=[self.args.dt_rank, n, n], dim=-1)  # delta: (b, l, dt_rank). B, C: (b, l, n)
#         delta = F.softplus(self.dt_proj(delta))  # (b, l, d_in)
        
#         y = self.selective_scan(x, delta, A, B, C, D)  # This is similar to run_SSM(A, B, C, u) in The Annotated S4 [2]
        
#         return y

    
#     def selective_scan(self, u, delta, A, B, C, D): #选择性扫描算法
#         """Does selective scan algorithm. See:
#             - Section 2 State Space Models in the Mamba paper [1]
#             - Algorithm 2 in Section 3.2 in the Mamba paper [1]
#             - run_SSM(A, B, C, u) in The Annotated S4 [2]

#         This is the classic discrete state space formula:
#             x(t + 1) = Ax(t) + Bu(t)
#             y(t)     = Cx(t) + Du(t)
#         except B and C (and the step size delta, which is used for discretization) are dependent on the input x(t).
    
#         Args:
#             u: shape (b, l, d_in)    (See Glossary at top for definitions of b, l, d_in, n...)
#             delta: shape (b, l, d_in)
#             A: shape (d_in, n)
#             B: shape (b, l, n)
#             C: shape (b, l, n)
#             D: shape (d_in,)
    
#         Returns:
#             output: shape (b, l, d_in)
    
#         Official Implementation:
#             selective_scan_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L86
#             Note: I refactored some parts out of `selective_scan_ref` out, so the functionality doesn't match exactly.
            
#         """
#         (b, l, d_in) = u.shape
#         n = A.shape[1]
        
#         # Discretize continuous parameters (A, B)
#         # - A is discretized using zero-order hold (ZOH) discretization (see Section 2 Equation 4 in the Mamba paper [1])
#         # - B is discretized using a simplified Euler discretization instead of ZOH. From a discussion with authors:
#         # A 使用零阶保持法(zero-order hold, ZOH)进行离散化 (see Section 2 Equation 4 in the Mamba paper [1])
#         # B 则使用一种简化的Euler方法进行离散化
#         # B没有使用ZOH的原因，作者解释如下: "A is the more important term and the performance doesn't change much with the simplification on B"
#         deltaA = torch.exp(einsum(delta, A, 'b l d_in, d_in n -> b l d_in n'))
#         deltaB_u = einsum(delta, B, u, 'b l d_in, b l n, b l d_in -> b l d_in n')
        
#         # Perform selective scan (see scan_SSM() in The Annotated S4 [2])
#         # Note that the below is sequential, while the official implementation does a much faster parallel scan that
#         # is additionally hardware-aware (like FlashAttention).
#         x = torch.zeros((b, d_in, n), device=deltaA.device)
#         ys = []    
#         for i in range(l):
#             x = deltaA[:, i] * x + deltaB_u[:, i]
#             y = einsum(x, C[:, i, :], 'b d_in n, b n -> b d_in')
#             ys.append(y)
#         y = torch.stack(ys, dim=1)  # shape (b, l, d_in)
        
#         y = y + u * D
    
#         return y


# class RMSNorm(nn.Module):   #均方根归一化
#     def __init__(self,
#                  d_model: int,
#                  eps: float = 1e-5):
#         super().__init__()
#         self.eps = eps
#         self.weight = nn.Parameter(torch.ones(d_model))


#     def forward(self, x):
#         output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

#         return output

# class Temp_encoder(nn.Module):
#     def __init__(self, args):
#         super(Temp_encoder, self).__init__()
#         self.args = args

#         self.input_size = 3
#         self.hidden_size = args.hidden_size
#         self.num_layers = 1
#         self.tmp_att = att(self.hidden_size)
#         self.lstm = nn.LSTM(
#             input_size=self.input_size,
#             hidden_size=self.hidden_size,
#             num_layers=self.num_layers,
#             batch_first=True,
#         )

#     def forward(self,batch):
#         displ = batch['displ']
#         lstm_in = torch.cat(displ,dim=0)
#         # lstm_in are all agents over all samples in the current batch
#         # Format for LSTM has to be has to be (batch_size, timeseries_length, latent_size), because batch_first=True

#         # Initialize the hidden state.
#         # lstm_in.shape[0] corresponds to the number of all agents in the current batch
#         lstm_hidden_state = torch.randn(
#             self.num_layers, lstm_in.shape[0], self.hidden_size, device=lstm_in.device)
#         lstm_cell_state = torch.randn(
#             self.num_layers, lstm_in.shape[0], self.hidden_size, device=lstm_in.device)
#         lstm_hidden = (lstm_hidden_state, lstm_cell_state)

#         lstm_out, lstm_hidden = self.lstm(lstm_in, lstm_hidden)

#         # lstm_out is the hidden state over all time steps from the last LSTM layer
#         # In this case, only the features of the last time step are used
#         return lstm_out[:, -1, :]