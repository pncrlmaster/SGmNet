import torch 
import torch.nn as nn

class loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.cls_loss = torch.nn.CrossEntropyLoss() #交叉熵损失函数
    
    def forward(self,pred,pi,propose_loc,gts):
        """
        pred: [batch_size, num_modes,pred_horizon, 2]
        gts :[batch_size,num_agrent,pred_horizon, 2]
        location: [batch_size,pred_horizon, 2]
        masks: [batch_size, pred_horizon]
        pi: [batch_size, num_modes,  1]  
        propose_loc: [batch_size, num_modes,2]
        """

        batch_size, num_modes, pred_horizon, _= pred.shape
        pi = pi.reshape(-1,num_modes) #[B,num_modes]
        location = torch.stack([x[0] for x in gts])#[B,Pred_horizon,2]

        l2_norm = torch.norm(propose_loc-location[:,-1:,:],p=2,dim=-1) #l2范数 [B,numnode]
        best_mode = l2_norm.argmin(dim=-1) #选最小的

        best_traj = pred[torch.arange(pred.size(0)), best_mode] #[B,T,F]

        propose_loss = l2_norm[torch.arange(pred.size(0)), best_mode].mean() #[就一个数]为每个样本找到最佳的预测位置，并计算其与真实位置之间的L2距离,最后对所有l2距离求平均
        ade_loss = torch.norm(best_traj - location, p=2, dim=-1).mean() #每个时间步的平均误差
        fde_loss = torch.norm(best_traj[:, -1, :] - location[:, -1, :], p=2, dim=-1).mean() #最终点的误差

        # cls_loss1 = torch.nn.CrossEntropyLoss()

        cls_loss = self.cls_loss(pi,best_mode.to(torch.long)) #交叉熵损失


        return propose_loss,ade_loss,fde_loss,cls_loss,best_traj