import numpy as np
import os
from torch.nn import functional as F
import torch
from torch import nn
import pytorch_lightning as pl
from torch_geometric.utils import from_scipy_sparse_matrix
from scipy import sparse
import sys
from typing import Union
sys.path.append('/home/zzh/hsti/att_vimmamba_SGNN_mlp')
from model.encoder import Temp_mamba,Temp_encoder,Spito_encoder
# from encoder import Temp_encoder,Spito_encoder
from linformer import Linformer
from model.interaction import Spito_inter,Temp_inter,SGNN
# from model.spito_encoder import Spito_encoder
# from model.space_encoder import Space_encoder

from model.layer import LSTM,CrossAttentionLayer
from model.decoder import DecoderV2
from model.fusion import Fusion_gate,Fusion_gate2
from model.embedding import Space_embedding , Temp_embedding,mlp,emb

file_path = os.path.abspath(__file__) #获取当前脚本的绝对路径 abspath将相对路径转换为绝对路径 __file__当前脚本文件的路径名
root_path = os.path.dirname(os.path.dirname(file_path)) #上两级目录（整目录） dirname上一级


class MFnet(pl.LightningModule):
    def __init__(self,args) :
        super(MFnet,self).__init__()

        self.args = args    #参数
        self.mlp = mlp(args)    #全连接激活全连接三层的mlp
        # self.lstm = LSTM(args)
        self.temp_encoder = Linformer(args.hidden_size,args.historical_steps,3)
        # self.temp_encoder = Temp_encoder(args)  #时间自注意力
        self.spito_encoder = Spito_encoder(args)    #空间自注意力
        self.fushin_gate   = Fusion_gate(args)  #融合
        self.emb = emb(args) #降[B,H]
        self.mamba = Temp_mamba(args)
        self.spito = Spito_inter(args)
        # self.spito_inter = Spito_inter(args) #空间交互 Crystal-GCN
        self.temp_inter = Temp_inter(args) #时间交互 MHA
        self.fushin_gate2 = Fusion_gate2(args)
        self.decoder = DecoderV2(args)
        self.temp_emb = Temp_embedding(args)
        
        # self.space_emd = Space_embedding(args)
        self.is_frozen = False

    @staticmethod
    def init_args(parent_parser):
        parser_dataset = parent_parser.add_argument_group("dataset")    #为dataset添加参数
        parser_dataset.add_argument(
            "--train_split", type=str, default=os.path.join(
                '/home/zzh/dataset/minitest', 'train','data' ))   #训练集路径
        parser_dataset.add_argument(
            "--val_split", type=str, default=os.path.join(
                '/home/zzh/dataset/minitest', 'val','data'))  #验证集路径
        # parser_dataset.add_argument(
            # "--test_split", type=str, default=os.path.join(    
            #     '/home/zzh/hsti/lxy_new/dataset', 'test_obs',))
        # parser_dataset.add_argument(
        #     "--test_split", type=str, default=os.path.join(
        #         '/home/zzh/hsti/lxy_new/dataset', "test_obs"))
        parser_dataset.add_argument(
            "--train_split_pre", type=str, default=os.path.join(
                root_path, "dataset", "processed_data", "train_pre.pkl"))
        parser_dataset.add_argument(
            "--val_split_pre", type=str, default=os.path.join(
                root_path, "dataset", "processed_data", "val_pre.pkl"))
        parser_dataset.add_argument(
            "--test_split_pre", type=str, default=os.path.join(
                root_path, "dataset", "processed_data", "test_pre.pkl"))    #预处理后的训练集验证集测试集路径
        parser_dataset.add_argument(
            "--reduce_dataset_size", type=int, default=0)
        parser_dataset.add_argument(
            "--use_preprocessed", type=bool, default=False)
        parser_dataset.add_argument(
            "--align_image_with_target_x", type=bool, default=True)

        parser_training = parent_parser.add_argument_group("training")  #训练参数
        parser_training.add_argument("--epochs", type=int, default=80)
        parser_training.add_argument(
            "--lr", type=int, default=1e-4)
        # parser_training.add_argument(
        #     "--lr_step_epochs", type=list, default=[5, 10, 20])
        parser_training.add_argument("--wd", type=float, default=0.01)
        parser_training.add_argument("--batch_size", type=int, default=64)
        parser_training.add_argument("--val_batch_size", type=int, default=64)
        parser_training.add_argument("--workers", type=int, default=0)
        parser_training.add_argument("--val_workers", type=int, default=0)
        # parser_training.add_argument("--gpus", type=int, default=1)
        parser_training.add_argument('--hidden_size',type=int, default=128)
        parser_training.add_argument('--d_model', type=int, default=128)
        parser_training.add_argument('--save_path',type=str,default= '/home/zzh/hsti/att_vimmamba_SGNN_mlp/lightning_logs/version_3')
        parser_training.add_argument('--gpu_id', type=str, default='0')
        parser_training.add_argument('--n_layer', type=int, default=5)
        parser_training.add_argument('--vocab_size', type=int, default=128)
        parser_training.add_argument('--d_state', type=int, default=16)
        parser_training.add_argument('--expand', type=int, default=2)
        parser_training.add_argument('--dt_rank', type=Union[int,str], default=2)
        parser_training.add_argument('--d_conv', type=int, default=1)
        parser_training.add_argument('--pad_vocab_size_multiple', type=int, default=8) 
        parser_training.add_argument('--conv_bias', type=bool, default=True)
        parser_training.add_argument('--bias', type=bool, default=False)
        parser_training.add_argument('--d_inner', type=int, default=128)
        parser_model = parent_parser.add_argument_group("model")    #模型参数
        parser_model.add_argument('--use_cuda',type=bool,default = True)
        # parser_model.add_argument("--latent_size", type=int, default=128)
        parser_model.add_argument('--multy_gpu_type',type=str, default = 'single_gpu')
        parser_model.add_argument("--num_preds", type=int, default=30)  #预测帧数
        parser_model.add_argument('--modelsave_dir',type=str ,default='/home/zzh/hsti/att_vimmamba_SGNN_mlp/save_model/version_3') 
        # parser_model.add_argument("--mod_steps", type=list, default=[1, 5])
        # parser_model.add_argument("--mod_freeze_epoch", type=int, default=36)
        parser_model.add_argument('--historical_steps', type=int, default=20)
        parser_model.add_argument('--future_steps', type=int, default=30)
        parser_model.add_argument('--num_modes', type=int, default=6)
        parser_model.add_argument('--rotate', type=bool, default=True)
        parser_model.add_argument('--node_dim', type=int, default=2)
        parser_model.add_argument('--edge_dim', type=int, default=2)
        parser_model.add_argument('--embed_dim', type=int,default=64)
        parser_model.add_argument('--num_heads', type=int, default=8)
        parser_model.add_argument('--dropout', type=float, default=0.1)
        parser_model.add_argument('--num_temporal_layers', type=int, default=4)
        parser_model.add_argument('--num_AA_layers', type=int, default=3)
        parser_model.add_argument('--local_radius', type=float, default=50)
        parser_model.add_argument('--parallel', type=bool, default=False)
        parser_model.add_argument('--weight_decay', type=float, default=1e-4)
        parser_model.add_argument('--T_max', type=int, default=64)

        return parent_parser
    
    def forward(self,batch):
        mlp_out = self.mlp(batch['feat'])   #首先将特征进行mlp升维 [B,A,T,H]->[B,A,T,128]
        # lstm_out = self.lstm(batch['feat']) #[B,A,T,128]
        spito_out = self.spito_encoder(mlp_out,batch) #空间自注意力 特征为绝对坐标和位移向量 
        temp_out  = self.temp_encoder(mlp_out.view(-1,mlp_out.shape[-2],mlp_out.shape[-1])) #时间自注意力
        spito = spito_out + mlp_out 
        temp = temp_out.view(mlp_out.shape) + mlp_out
        fusion_out = self.fushin_gate(spito,temp) #融合 并可以通过权重z调节两个注意力的占比
        mamba_out = self.mamba(fusion_out,batch) #时间交互 [B,A,T,H]
        tmp_out = self.temp_emb(mamba_out)#[B,H]

        sgnn_out = self.spito(fusion_out,batch) #空间交互

        # aainter_out2  = self.spito(aainter_out,batch)
        # aa_out = self.emb(sgnn_out,batch)
        # aainter_out2 = self.spito(aainter_out,batch)
        # gnn_out = self.spito_inter(encoder_out,batch)#[B,H] #Crystal-GCN空间交互
        # mha_out = self.temp_inter(encoder_out,batch)#[B,H] #多头注意力时间交互 只有最后一帧特征
        
        # spc_out = self.space_emd(aa_out)#[B,H]
 

        decoder_in = self.fushin_gate2(tmp_out,sgnn_out)
        # decoder_in = mha_out

        pred_traj,pi,propose_loc =  self.decoder(decoder_in,batch)

        return pred_traj,pi,propose_loc
    # def freeze(self):
    #     for param in self.parameters():
    #         param.requires_grad = False

    #     self.decoder_residual.unfreeze_layers()

    #     self.is_frozen = True

    # def loss(self,pred,pi,propse_loc,gts):
    #     """
    #     pred: [batch_size, num_modes,pred_horizon, 2]
    #     gts :[batch_size,num_agrent,pred_horizon, 2]
    #     location: [batch_size,pred_horizon, 2]
    #     masks: [batch_size, pred_horizon]
    #     pi: [batch_size, num_modes,  1]  
    #     propose_loc: [batch_size, num_modes,2]
    #     """

    #     batch_size, num_modes, pred_horizon, _= pred.shape
    #     pi = pi.reshape(-1,num_modes)
    #     location = torch.stack([x[0] for x in gts])#[B,Pred_horizon,2]

    #     l2_norm = torch.norm(propse_loc-location[:,-1:,:],p=2,dim=-1)
    #     best_mode = l2_norm.argmin(dim=-1)

    #     best_traj = pred[torch.arange(pred.size(0)), best_mode]

    #     propose_loss = l2_norm[torch.arange(pred.size(0)), best_mode].mean()
    #     ade_loss = torch.norm(best_traj - location, p=2, dim=-1).mean()
    #     fde_loss = torch.norm(best_traj[:, -1, :] - location[:, -1, :], p=2, dim=-1).mean()

    #     cls_loss1 = torch.nn.CrossEntropyLoss()

    #     cls_loss = cls_loss1(pi,best_mode.to(torch.long))

    #     loss = propose_loss+ade_loss+fde_loss+cls_loss
    #     return loss,best_traj

    # def configure_optimizers(self):
    #     if self.current_epoch == self.args.mod_freeze_epoch:
    #         optimizer = torch.optim.Adam(
    #             filter(lambda p: p.requires_grad, self.parameters()), weight_decay=self.args.wd)
    #     else:
    #         optimizer = torch.optim.Adam(
    #             self.parameters(), weight_decay=self.args.wd)
    #     return optimizer

    # def on_train_epoch_start(self):
    #     for single_param in self.optimizers().param_groups:
    #         single_param["lr"] = self.get_lr(self.current_epoch)

    # def training_step(self, train_batch, batch_idx):
    #     pred_traj,pi,propose_loc = self.forward(train_batch)
    #     loss,_ = self.loss(pred_traj,pi, propose_loc,train_batch["fut_trajs"])
    #     self.log("loss_train", loss / len(pred_traj))
    #     return loss
    
    # def get_lr(self, epoch):
    #     lr_index = 0
    #     for lr_epoch in self.args.lr_step_epochs:#[5,10,20] [5e-2, 1e-3, 5e-4, 1e-4]
    #         if epoch < lr_epoch:
    #             break
    #         lr_index += 1
    #     return self.args.lr_values[lr_index]

    # def validation_step(self, val_batch, batch_idx):
    #     pred_traj,pi,propose_loc = self.forward(val_batch)
    #     loss,best_traj= self.loss(pred_traj, pi, propose_loc,val_batch["fut_trajs"])
    #     self.log("loss_val", loss / len(pred_traj))
    #     location = torch.stack([x[0] for x in val_batch['gt']])

    #     pred = best_traj.detach().cpu().numpy()
    #     gt = location.detach().cpu().numpy()
    #     # pred = best_traj
    #     # gt =location
    #     return pred, gt

    # def validation_epoch_end(self, validation_outputs):
    #     # Extract predictions
    #     pred = [out[0] for out in validation_outputs]
    #     pred = np.concatenate(pred, 0)
    #     gt = [out[1] for out in validation_outputs]
    #     gt = np.concatenate(gt, 0)
    #     # ade1, fde1, ade, fde = self.calc_prediction_metrics(pred, gt)
    #     ade, fde = self.calc_prediction_metrics(pred, gt)
    #     # self.log("ade1_val", ade1, prog_bar=True)
    #     # self.log("fde1_val", fde1, prog_bar=True)
    #     self.log("ade_val", ade, prog_bar=True)
    #     self.log("fde_val", fde, prog_bar=True)

    # def calc_prediction_metrics(self, preds, gts):

    #     fde = np.linalg.norm(preds[:,-1,:]-gts[:,-1,:],ord=2,axis=-1).mean()
    #     ade = np.linalg.norm(preds-gts,ord=2,axis=-1).mean()        
    #     return  ade, fde
    