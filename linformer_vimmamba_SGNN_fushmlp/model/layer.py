import torch
import torch.nn as nn
import math
import numpy as np


class att(nn.Module):   #注意力层
    def __init__(self, d_model, dropout=0.2):
        super().__init__()
        
        self.q_lin = nn.Linear(d_model, d_model)
        self.k_lin = nn.Linear(d_model, d_model)
        self.v_lin = nn.Linear(d_model, d_model)    #通过线性层将输入向量转换为QKV
        
        self.layernorm = nn.LayerNorm(d_model)  #归一化
        self.scale = d_model

        self.ff_mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),    #前馈网络
        )

        self.ff_prenorm = nn.LayerNorm(d_model)
        self.ff_postnorm = nn.LayerNorm(d_model)    #前后分别layernorm稳定训练过程

    @classmethod
    def masked_softmax(self, x, mask):
        if mask is not None: #[B,A,T]

            return nn.functional.softmax(torch.masked_fill(x, ~mask.bool(), -1e12), dim=-1) * mask.float() #false的话将那个数改为极小数再做SOFTMAX 
        else:
            return nn.functional.softmax(x, dim=-1)

    def forward(self, q, k, v, mask=None):
        """
        q: [b, a, t, h][2,20,20,64]
        k: [b, a, t, h]
        v: [b, a, t, h]
        attn_mask: [b, a, t, t]
        """
        query = self.q_lin(q)
        key = self.k_lin(k)
        value = self.v_lin(v)
        
        scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.scale) #qk/scale
        attention_weights = self.masked_softmax(scores, mask)
        x = torch.matmul(attention_weights, value)
        x = q + self.layernorm(x)
        x = x + self.ff_postnorm(self.ff_mlp(self.ff_prenorm(x)))
        return x

class MultyHeadAttn(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.2):
        super().__init__()
        assert d_model % num_heads == 0
        self.q_lin = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Dropout(dropout),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        self.k_lin = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Dropout(dropout),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        self.v_lin = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Dropout(dropout),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        self.layernorm = nn.LayerNorm(d_model // num_heads)
        self.scaled = d_model // num_heads
        self.num_heads = num_heads

    @classmethod
    def masked_softmax(cls, x, mask):
        if mask is not None:
            return nn.functional.softmax(torch.masked_fill(x, ~mask.bool().unsqueeze(dim=-1), -np.inf), dim=-1) * (1- mask.float())
        else:
            return nn.functional.softmax(x, dim=-1)
    def forward(self, q, k, v, mask=None):
        """
        q: [b, q, h]
        k: [b, k, h]
        v: [b. k, h]
        attn_mask: [b, q, k]
        """
        query = self.q_lin(q)
        key = self.k_lin(k)
        value = self.v_lin(v)

        # k_size = k.size(1)
        # q_size = q.size(1)

        # query = query.view(-1, q_size, self.scaled)
        # key = key.view(-1, k_size, self.scaled)
        # value = value.view(-1, k_size, self.scaled)

        scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.scaled)
        attention_weights = self.masked_softmax(scores, torch.repeat_interleave(mask, self.num_heads, 0))
        x = torch.matmul(attention_weights, value)
        x = self.layernorm(x)
        # x = x.view(-1, q_size, self.scaled * self.num_heads)
        output = q + x
        return output
    
class LSTM(nn.Module):
    def __init__(self, args,input_dim=4, num_layers=2, output_dim=128):
        """
        初始化LSTM网络。
        :param input_dim: 输入特征维度
        :param hidden_dim: LSTM的隐藏状态维度
        :param num_layers: LSTM层数
        :param output_dim: 输出特征维度
        """
        super(LSTM, self).__init__()

        self.input_dim = input_dim
        hidden_dim = args.hidden_size
        self.num_layers = num_layers
        self.output_dim = output_dim

        # LSTM层
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)

        # 一个线性层将LSTM的输出映射到目标输出维度
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        前向传播
        :param x: 输入数据形状 [B, A, T, input_dim]
        :return: 输出数据形状 [B, A, T, output_dim]
        """
        # 转换输入形状为 [B * A, T, input_dim]，因为LSTM需要处理 [batch_size, time_steps, input_dim]
        B, A, T, input_dim = x.size()
        x = x.view(B * A, T, input_dim)

        # LSTM的输入是 [B * A, T, input_dim]
        lstm_out, (hn, cn) = self.lstm(x) 

        # LSTM的输出形状为 [B * A, T, hidden_dim]
        # 将LSTM的输出传递到一个全连接层，输出形状为 [B * A, T, output_dim]
        out = self.fc(lstm_out)

        # 变回原来的形状 [B, A, T, output_dim]
        out = out.view(B, A, T, self.output_dim)

        return out


    
class EncoderLstm(nn.Module):
    def __init__(self, args):
        super(EncoderLstm, self).__init__()
        self.args = args

        self.input_size = 3
        self.hidden_size = args.latent_size
        self.num_layers = 1

        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
        )

    def forward(self, lstm_in, agents_per_sample):
        # lstm_in are all agents over all samples in the current batch
        # Format for LSTM has to be has to be (batch_size, timeseries_length, latent_size), because batch_first=True

        # Initialize the hidden state.
        # lstm_in.shape[0] corresponds to the number of all agents in the current batch
        lstm_hidden_state = torch.randn(
            self.num_layers, lstm_in.shape[0], self.hidden_size, device=lstm_in.device)
        lstm_cell_state = torch.randn(
            self.num_layers, lstm_in.shape[0], self.hidden_size, device=lstm_in.device)
        lstm_hidden = (lstm_hidden_state, lstm_cell_state)

        lstm_out, lstm_hidden = self.lstm(lstm_in, lstm_hidden)

        # lstm_out is the hidden state over all time steps from the last LSTM layer
        # In this case, only the features of the last time step are used
        return lstm_out[:, -1, :]
    
class CrossAttentionLayer(nn.Module):
    def __init__(self, args):
        super(CrossAttentionLayer, self).__init__()
        self.input_dim = args.hidden_size
        self.ffn_dim = self.input_dim * 2
        # 线性变换层
        self.query_proj = nn.Linear(self.input_dim, self.input_dim)
        self.key_proj = nn.Linear(self.input_dim, self.input_dim)
        self.value_proj = nn.Linear(self.input_dim, self.input_dim)
        
        # 前馈神经网络（FFN）
        self.ffn = nn.Sequential(
            nn.Linear(self.input_dim, self.ffn_dim),
            nn.ReLU(),
            nn.Linear(self.ffn_dim, self.input_dim)
        )
        
        # 最终线性变换
        self.final_proj = nn.Linear(self.input_dim, self.input_dim)
        
        # Softmax归一化
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query, key, value):
        """
        query: 空间特征 (spito_out), 形状 [B, 128]
        key: 时间特征 (temp_out), 形状 [B, 128]
        value: 时间特征 (temp_out), 形状 [B, 128]
        """
        # 线性变换
        Q = self.query_proj(query)  # [B, 128]
        K = self.key_proj(key)      # [B, 128]
        V = self.value_proj(value)  # [B, 128]

        # 计算点积，得到相关性矩阵
        attn_scores = torch.matmul(Q, K.transpose(-1, -2))  # [B, B]
        
        # 归一化相关性图
        attn_weights = self.softmax(attn_scores)  # [B, B]
        
        # 用相关性权重加权V
        weighted_V = torch.matmul(attn_weights, V)  # [B, 128]
        
        # 通过线性层将加权结果恢复到原始形状
        weighted_V = self.final_proj(weighted_V)  # [B, 128]
        
        # 与V进行残差连接
        T = weighted_V + V  # [B, 128]
        
        # 通过前馈神经网络（FFN）进行进一步的非线性变换
        T_transformed = self.ffn(T)  # [B, 128]
        
        # 与T再进行残差连接
        final_output = T_transformed + T  # [B, 128]

        return final_output