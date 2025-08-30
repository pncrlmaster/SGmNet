# loss function for train the model
import torch
import torch.nn as nn
import torch.nn.functional as F


class SceneLossWithProb(nn.Module):
    def __init__(self, reduction):
        super().__init__()
        self.cls_loss = torch.nn.CrossEntropyLoss()
        if reduction in ["mean", "sum", "none"]:
            self.reduction = reduction
        else:
            raise NotImplementedError("[VectorLoss]: The reduction has not been implemented!")

    def forward(self,pred,pi,propse_loc,gts):
        
        """
        pred: [batch_size, num_modes, A, pred_horizon, 2]
        location: [batch_size, A, pred_horizon, 2]
        masks: [batch_size, A, pred_horizon]
        pi: [batch_size, num_modes, A, 1]  
        propose_loc: [batch_size, num_modes, A, 2]
        """
        
        batch_size, num_modes, pred_horizon, _= pred.shape
        pi = pi.reshape(-1,num_modes)
        location = torch.stack([x[0] for x in gts])#[B,Pred_horizon,2]

        l2_norm = torch.norm(propse_loc-location[:,-1:,:],p=2,dim=-1)
        best_mode = l2_norm.argmin(dim=-1)
        
        best_traj = pred[torch.arange(pred.size(0)), best_mode]

        propose_loss = l2_norm[torch.arange(pred.size(0)), best_mode].mean()
        ade_loss = torch.norm(best_traj - location, p=2, dim=-1).mean()
        fde_loss = torch.norm(best_traj[:, -1, :] - location[:, -1, :], p=2, dim=-1).mean()


        cls_loss = cls_loss(pi,best_mode.to(torch.long))

        return propose_loss, ade_loss, fde_loss, cls_loss