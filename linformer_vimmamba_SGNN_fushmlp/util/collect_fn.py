import numpy as np
import torch
import argparse
from model.main_model import MFnet
from torch.nn.utils.rnn import pad_sequence

parser = argparse.ArgumentParser()
parser = MFnet.init_args(parser)
   
 # parser.add_argument('--train_args_path', type=str, default="scene_model_v2.json")
args = parser.parse_args()

def recursive_numpy_to_tensor(np_data): #numpy转tensor
    """
    Recursively convert lists of numpy arrays, tuples of numpy arrays
    or dictionary of numpy arrays to torch tensors

    Args:
        np_data: Numpy data to convert

    Returns:
        Converted torch tensor
    """
    if isinstance(np_data, np.ndarray):
        np_data = torch.from_numpy(np_data)
    elif isinstance(np_data, tuple):
        np_data = [recursive_numpy_to_tensor(x) for x in np_data]
    elif isinstance(np_data, list):
        np_data = [recursive_numpy_to_tensor(x) for x in np_data]
    elif isinstance(np_data, dict):
        for key in np_data:
            np_data[key] = recursive_numpy_to_tensor(np_data[key])
    return np_data
def collate_fn_dict(in_batch):
    """Custom collate_fn that returns a dictionary of lists

    Args:
        in_batch: Batch containing a list of dictionaries

    Returns:
        Batch containing a dictionary of lists
    """
    in_batch = recursive_numpy_to_tensor(in_batch) #将所有np转换为tensor in_batch是个batchsize长度的list
    out_batch = dict()
    for key in in_batch[0]:
        out_batch[key] = [x[key] for x in in_batch] #合并相同的键 将不同文件合并成一个数据
    batch_size = len(in_batch) 
    max_agent = 0
    for x in in_batch:
        max_agent = max(max_agent,x['displ'].shape[0])  #看数据的数量看最多有几个ag


    pad_feats = np.zeros([batch_size,max_agent,20,4],dtype=np.float32)  
    # positions = np.zeros([batch_size,max_agent,50,2],dtype=np.float32)
    # padding_masks = np.zeros([batch_size,max_agent,50],np.bool_)
    pad_pre_masks = np.zeros([batch_size,max_agent,20],np.bool_)
    # pad_fut_masks = np.zeros([batch_size,max_agent,30],np.bool_)
    # data['pre_traj_masks'] [B,A,20] ;data['fut_traj_masks'] [B,A,30]
    

    for i ,data in enumerate(in_batch):
        data['feat'] = np.stack(data['feat']) #堆叠到一起 np
        # data['positions'] = np.stack(data['positions'])
        num_agent = data['displ'].shape[0]
        pad_feats[i,:num_agent] = data['feat']
        # positions[i,:num_agent] = data['positions'] #同一个样本为什么两个A不一样 因为自己的预处理不考虑预测开始点没有坐标的车
        pad_pre_masks[i,:num_agent] = data['pre_traj_masks'] 
        # padding_masks[i,:num_agent] = data['padding_mask']
        # pad_fut_masks[i,:num_agent] = data['fut_traj_masks']  #用pad_feats都拿进来 

    pre_masks = torch.from_numpy(pad_pre_masks)#  [B,Agent,20]
    # padding_masks = torch.from_numpy(padding_masks) #[b,a,50]
    out_batch['mamba_mask'] = pre_masks
    out_batch['agent_mask'] = (pre_masks.permute(0,2,1).unsqueeze(-1))&(pre_masks.permute(0,2,1).unsqueeze(-2)) #permute改顺序(0,1,2)改到(0,2,1) unsqueeze分别在最后一个维度和倒数第二个维度添加一个新轴生成形状为 (batch_size, num_agents, sequence_length, 1) 和 (batch_size, num_agents, 1, sequence_length) (B, T, A, A)
    out_batch['time_mask'] = pre_masks.unsqueeze(-1) & pre_masks.unsqueeze(-2) #[B,A,T,20] (B, A, T, T)
#分别用来代理间的关系 每个代理内的时间步之间的关系
    out_batch['feat'] = torch.from_numpy(pad_feats) 
    if args.use_cuda == True:
        # for key,value in out_batch:
        #     out_batch[key] = value.cuda()
        out_batch['feat'] = out_batch['feat'].cuda() #cpu to gpu
        out_batch['agent_mask'] = out_batch['agent_mask'].cuda()
        out_batch['time_mask'] = out_batch['time_mask'].cuda()
        out_batch["displ"] = [x.cuda() for x in out_batch['displ']] #这个就是列表
        out_batch["centers"] = [x.cuda() for x in out_batch['centers']]
        out_batch["fut_trajs"] = [x.cuda() for x in out_batch['fut_trajs']]
        # out_batch['padding_masks'] = padding_masks.cuda() #改成带Batch的
    # out_batch['pad_fut_masks'] = torch.from_numpy(pad_fut_masks)

    return out_batch 


    