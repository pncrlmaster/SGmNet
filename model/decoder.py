import torch 
import torch.nn as nn
import pickle
import argparse
import sys 
# sys.path.append('/home/lxy/programs/lxy_new/model')
# from main_model import MFnet
# from layer import att
from .layer import att
from .position_encoding_utils import gen_sineembed_for_position


# path = '/home/zzh/hsti/att_vimmamba_SGNN/dataset/kmeans_center.pkl' 
# with open(path,'rb') as f:
#     center = pickle.load(f) #[64,2] 64个聚类中心
# center = torch.from_numpy(center) #numpy数组变pytorch张量
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# parser = argparse.ArgumentParser()
# parser = MFnet.init_args(parser)

#     # parser.add_argument('--train_args_path', type=str, default="scene_model_v2.json")
# args = parser.parse_args()



# class Decoder(nn.Module):
#     def __init__(self,args,num_nodes=64,pred_horizon=30):
#         super().__init__()
#         self.center =center
#         self.pred_horizon = 30  #预测30帧
#         self.num_nodes = num_nodes  #预测64条轨迹
#         self.hidden_size = args.hidden_size
#         self.batch_size = 5
#         self.sub_layers = 3
#         # self.modes_dense = nn.Linear(self.hidden_size,num_nodes*self.self.hidden_size)

#         self.to_propose_goal = nn.Sequential(
#             nn.Linear(self.hidden_size, self.hidden_size),
#             nn.LayerNorm(self.hidden_size),
#             nn.ReLU(inplace=True),
#             nn.Linear(self.hidden_size, 2)
#         )


#         self.goal_dense = nn.Sequential(
#             nn.Linear(2, self.hidden_size),
#             nn.LayerNorm(self.hidden_size),
#             nn.ReLU(),
#             nn.Linear(self.hidden_size, self.hidden_size * 2),
#             nn.LayerNorm(self.hidden_size * 2),
#             nn.ReLU(),
#             nn.Linear(self.hidden_size * 2, self.hidden_size)
#         )

#         self.goal_fusion = nn.Sequential(
#             nn.Linear(2 * self.hidden_size, self.hidden_size),
#             nn.LayerNorm(self.hidden_size),
#             nn.ReLU(),
#             nn.Linear(self.hidden_size, self.hidden_size),
#             nn.LayerNorm(self.hidden_size),
#             nn.ReLU()
#         )
#         self.to_pred_traj = nn.Sequential(
#             nn.Linear(self.hidden_size, self.hidden_size * 2),
#             nn.LayerNorm(self.hidden_size * 2),
#             nn.GELU(),
#             nn.Linear(self.hidden_size * 2, self.hidden_size),
#             nn.GELU(),
#             nn.Linear(self.hidden_size, self.hidden_size),
#             nn.GELU(),
#             nn.Linear(self.hidden_size,self.pred_horizon*2),
#         )
        
#         self.to_pi = nn.Sequential(
#             nn.Linear(self.hidden_size, self.hidden_size),
#             nn.LayerNorm(self.hidden_size),
#             nn.ReLU(inplace=True),
#             nn.Linear(self.hidden_size, 1),
#         )
        
#         self.to_cen = nn.Sequential(
#             nn.Linear(self.hidden_size,self.hidden_size),
#             nn.GELU(),
#             nn.Linear(self.hidden_size,self.hidden_size),
#         )

#         self.to_fusion = nn.Sequential(
#             nn.Linear(self.hidden_size,self.hidden_size),
#             nn.LayerNorm(self.hidden_size),
#             nn.ReLU(),
#             nn.Linear(self.hidden_size, self.hidden_size),
#             nn.ReLU(),
#             nn.Linear(self.hidden_size, self.hidden_size),
            
#         )
#         self.query = nn.Parameter(torch.randn(self.num_nodes,self.hidden_size))

#         self.att = nn.ModuleList(att(d_model=self.hidden_size) for _ in range(self.sub_layers))

    
    
#     def forward(self,traj_emb):
#         batch_size,self.hidden_size = traj_emb.shape
#         traj_emb = traj_emb.unsqueeze(1).repeat(1,64,1).float()#[B,64,128] unsqueeze插入一维 repeat按维度重复，第二维复制64次
#         # traj_emb = self.mod(traj_emb).view(batch_size,self.num_nodes,-1)
#         #tra_emd [B,N,H][B,64,128]
#         center = self.center.to(device=traj_emb.device) #聚类中心
#         cen = center.unsqueeze(0).repeat(batch_size,1,1)#[B,64,2]
#         pos = gen_sineembed_for_position(cen).float()#[B,64,128] xy做正余弦嵌入
#         pos_emb = self.to_cen(pos)#[B,64,128]

#         trajc_emb = self.to_fusion(traj_emb+pos_emb)#[B,64,128] query是可学习参数
#         q,k,v = (self.query+trajc_emb),(self.query+trajc_emb),self.query 
#         for i in range(self.sub_layers):
#             q = self.att[i](q,k,v) + traj_emb  #[B,64,128]
#             k = q
        
#         scores = self.to_pi(q).reshape(batch_size,-1) #[B,64]
#         pred_traj = self.to_pred_traj(q).reshape(batch_size,self.num_nodes,self.pred_horizon,-1) #[B,64,T,F]
#         pred_trajs_final, pred_scores_finals,_= self.batch_nms(pred_traj,scores)#[B,numnodes,T,F],[B,numnodes]
#         propse_loc = pred_trajs_final[:,:,-1,:] #最后一秒位置
#         return pred_trajs_final,pred_scores_finals,propse_loc

#     def batch_nms(self,pred_trajs, pred_scores, dist_thresh=3, num_ret_modes=6): #非极大值抑制 主要目标是基于距离阈值和得分排序，从给定的一批预测轨迹中选择一定数量的高分且不重叠的轨迹
#         """

#         Args:
#             pred_trajs (batch_size, num_modes, num_timestamps, 7)
#             pred_scores (batch_size, num_modes):
#             dist_thresh (float):
#             num_ret_modes (int, optional): Defaults to 6.

#         Returns:
#             ret_trajs (batch_size, num_ret_modes, num_timestamps, 5)
#             ret_scores (batch_size, num_ret_modes)
#             ret_idxs (batch_size, num_ret_modes)
#         """
#         batch_size, num_modes, num_timestamps, num_feat_dim = pred_trajs.shape

#         sorted_idxs = pred_scores.argsort(dim=-1, descending=True) #降序排序(预测模式？)[B,64]
#         bs_idxs_full = torch.arange(batch_size).type_as(sorted_idxs)[:, None].repeat(1, num_modes) #[B,64]
#         sorted_pred_scores = pred_scores[bs_idxs_full, sorted_idxs] #[B,64]
#         sorted_pred_trajs = pred_trajs[bs_idxs_full, sorted_idxs]  # (batch_size, num_modes, num_timestamps, f) 按模式得分排序的得分张量和轨迹张量
#         sorted_pred_goals = sorted_pred_trajs[:, :, -1, :]  # (batch_size, num_modes, f) #预测目标点

#         dist = (sorted_pred_goals[:, :, None, 0:2] - sorted_pred_goals[:, None, :, 0:2]).norm(dim=-1) #[B,64,64]计算所有预测目标点之间的欧氏距离，形成一个距离矩阵
#         point_cover_mask = (dist < dist_thresh) #距离太近的目标点会覆盖

#         point_val = sorted_pred_scores.clone()  # (batch_size, N)
#         point_val_selected = torch.zeros_like(point_val)  # (batch_size, N)

#         ret_idxs = sorted_idxs.new_zeros(batch_size, num_ret_modes).long() #[B,num_ret_modes(6)]
#         ret_trajs = sorted_pred_trajs.new_zeros(batch_size, num_ret_modes, num_timestamps, num_feat_dim) #[B,6,30,2]
#         ret_scores = sorted_pred_trajs.new_zeros(batch_size, num_ret_modes) 
#         bs_idxs = torch.arange(batch_size).type_as(ret_idxs) #[2]

#         for k in range(num_ret_modes): #选num_ret_modes个轨迹
#             cur_idx = point_val.argmax(dim=-1) # (batch_size)未标记的最高分的索引
#             ret_idxs[:, k] = cur_idx #索引储存

#             new_cover_mask = point_cover_mask[bs_idxs, cur_idx]  # (batch_size, N)
#             point_val = point_val * (~new_cover_mask).float()  # (batch_size, N) 根据新的覆盖掩码更新得分张量，降低被覆盖预测的得分
#             point_val_selected[bs_idxs, cur_idx] = -1
#             point_val += point_val_selected

#             ret_trajs[:, k] = sorted_pred_trajs[bs_idxs, cur_idx]
#             ret_scores[:, k] = sorted_pred_scores[bs_idxs, cur_idx]

#         bs_idxs = torch.arange(batch_size).type_as(sorted_idxs)[:, None].repeat(1, num_ret_modes) #[B,num_ret_modes]

#         ret_idxs = sorted_idxs[bs_idxs, ret_idxs] #[B,num_ret_modes] []中的两个数组形状要匹配
#         return ret_trajs, ret_scores, ret_idxs

class DecoderV2(nn.Module):
    def __init__(self, args,hidden_size=128, num_modes=6, sub_layers=3, pred_horizon=30):
        super().__init__()
        self.sub_layers = sub_layers
        self.pred_horizon = pred_horizon
        self.num_modes = num_modes
        self.modes_dense = nn.Linear(hidden_size, hidden_size * num_modes)
        # self.mode2lane_attn = nn.ModuleList(
        #     [Attn(d_model=hidden_size) for _ in range(sub_layers)]
        # )
        # self.model2mode_attn = nn.ModuleList(
        #     [Attn(d_model=hidden_size) for _ in range(sub_layers)]
        # )
        
        self.to_pi = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, 1),
        )
        
        self.to_pred_traj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.LayerNorm(hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, pred_horizon * 2),
        )
        
        self.to_propose_goal = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, 2)
        )
        
        self.goal_dense = nn.Sequential(
            nn.Linear(2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size * 2),
            nn.LayerNorm(hidden_size * 2),
            nn.ReLU(),
            nn.Linear(hidden_size * 2, hidden_size)
        )
        
        self.goal_fusion = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU()
        )
        
        self.goal_attn = nn.ModuleList([att(d_model=hidden_size) for _ in range(sub_layers)])
        
    def forward(self, traj_emb, data):
        batch_size,  hidden_size = traj_emb.shape
        traj_emb = self.modes_dense(traj_emb).view(batch_size,  self.num_modes, -1) #[B,nodes,H]
        # traj_emb = traj_emb.permute(0, 2, 1, 3).reshape(batch_size, -1, hidden_size)
        # agent_masks = data['mamba_mask']
        # agent_masks = torch.stack([x[:,0] for x in agent_masks])
        # masks = agent_masks.any(dim=-1).repeat(1, self.num_modes)
        # a2a_mask = masks.unsqueeze(-2) & masks.unsqueeze(-1)
        # a2l_masks = a2l_masks.repeat(1, self.num_modes, 1)
        
        # a2a_masks = self.get_self_attn_mask(agent_masks)
        # for i in range(self.sub_layers):
        #     traj_emb = self.mode2lane_attn[i](traj_emb, lane_emb, lane_emb, a2l_masks)
        #     traj_emb = self.model2mode_attn[i](traj_emb, traj_emb, traj_emb, a2a_masks)
        
        # query = query.reshape(batch_size, self.num_modes, num_agent, -1)
        propose_goals = self.to_propose_goal(traj_emb)
        propose_goals_emb = self.goal_dense(propose_goals)  #
        
        traj_emb = self.goal_fusion(torch.cat([traj_emb, propose_goals_emb], dim=-1))
        # traj_emb = self.goal_fusion(torch.cat([traj_emb, propose_goals], dim=-1))   # 
        traj_emb = traj_emb.reshape(batch_size, self.num_modes,  -1)
        
        # a2a_mask = agent_masks.any(dim=-1).unsqueeze(-1) & agent_masks.any(dim=-1).unsqueeze(-2)
        
        # pred_trajs = []
        # for step in range(self.num_recurrent_steps):
        #     for layer in self.goal_attn:
        #         traj_emb = layer(traj_emb, traj_emb, traj_emb, a2a_mask.unsqueeze(1))
        #     pred_trajs.append(self.to_pred_traj(traj_emb))

        # for layer in self.goal_attn:
        #     traj_emb = layer(traj_emb, traj_emb, traj_emb, a2a_mask)  
            
        pi = self.to_pi(traj_emb)
        # pred_traj = torch.cat(pred_trajs, dim=-1).reshape(batch_size, self.num_modes, num_agent, self.pred_horizon, 2)
        pred_traj = self.to_pred_traj(traj_emb).reshape(batch_size, self.num_modes, self.pred_horizon, 2)
        
        return pred_traj, pi, propose_goals.reshape(batch_size, self.num_modes,  -1)

    def get_self_attn_mask(self, masks):
        masks = masks.any(dim=-1).repeat(1, self.num_modes)
        masks = masks.unsqueeze(-2) & masks.unsqueeze(-1)
        return masks
