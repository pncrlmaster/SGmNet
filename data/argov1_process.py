import torch
import numpy  as np
import pandas as pd
import pickle
import glob
from itertools import permutations
import os
import copy
from argoverse.data_loading.argoverse_forecasting_loader import ArgoverseForecastingLoader
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import Dataset,DataLoader
from argoverse.map_representation.map_api import ArgoverseMap
from multiprocessing import Pool
import sys
import logging
import argparse
import multiprocessing
import uuid
sys.path.append('/home/zzh/hsti/att_vimmamba_SGNN_mlp')
from model.main_model import MFnet

os.umask(0)

root_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root_path)

log_dir = os.path.dirname(os.path.abspath(__file__))
logging.getLogger("pytorch_lightning").setLevel(logging.INFO)

parser = argparse.ArgumentParser()
parser = MFnet.init_args(parser)

parser.add_argument("--n_cpus", type=int, default=multiprocessing.cpu_count())
parser.add_argument("--chunksize", type=int, default=5)

class Argodataset(Dataset):
    def __init__(self,split,save_path,args):
        self.savepath = save_path
        self.split = split
        self.obs_horizon = 20

        if args.use_preprocessed:   #use_preprocessed为true的话用save_path下的预处理过的数据 false的话从split下面的csv里面构建
            with open(save_path, 'rb') as f:
                self.data = pickle.load(f)
                pass
        else:
            self.files = sorted(glob.glob(f"{split}/*.csv"))
            if args.reduce_dataset_size > 0:    #需要减小数据集的话减小
                self.files = self.files[:args.reduce_dataset_size]

    def __getitem__(self, idx):
        data = self.process(idx)
        
        return data
    
    def __len__(self):
        return len(os.listdir(self.split))

    def process(self,idx):
        data = self.get_features(self.files[idx])
        data = self.get_mask(data) #记录信息缺失
        return data


    def get_features(self,file_name):   #提取特征
        # avl = ArgoverseForecastingLoader(self.split)
        df = pd.read_csv(file_name)
        city = df["CITY_NAME"].values[0]
        argo_id = int(Path(file_name).stem) #Path(file_name).stem的作用是从文件路径file_name中提取文件名

        agt_ts = np.sort(np.unique(df["TIMESTAMP"].values))#记录50个时间戳
        mapping = dict()
        for i, ts in enumerate(agt_ts): #时间排序
            mapping[ts] = i
        trajs = np.concatenate((
            df.X.to_numpy().reshape(-1, 1),
            df.Y.to_numpy().reshape(-1, 1)), 1)
        

        steps = [mapping[x] for x in df["TIMESTAMP"].values]#记录步长
        steps = np.asarray(steps, np.int64)
        objs = df.groupby(["TRACK_ID", "OBJECT_TYPE"]).groups   #根据TRACK_ID和OBJECT_TYPE这两列的值分组体现有几辆车 .groups返回一个字典
        keys = list(objs.keys())
        obj_type = [x[1] for x in keys]
        agnt_key = keys.pop(obj_type.index("AGENT")) #把obj_type中的agent拿出来
        av_key = keys.pop(obj_type.index("AV")) #AV拿出来
        keys = [agnt_key, av_key] + keys#把Agent和AV的信息放到1 2位
        # num_nodes = len(objs)


        res_trajs,ctx_steps  = [],[]    #存着所有车的轨迹 时间
        idlist = []
        for key in keys:
            idcs = objs[key] #取出索引 代表一辆车在数据中所占的所有行
            tt = trajs[idcs]#取出每辆车的轨迹点
            ts = steps[idcs]#取出每辆车的运动步
            
            
            rt = np.zeros((50, 3))
            
            if 19 not in ts: #在不在起始点
                continue
            idlist.append(key[0])
            rt[ts, :2] = tt
            rt[ts, 2] = 1.0 #有轨迹的点填充到rt前两列 并在第三列标记为1
            res_trajs.append(rt) #[A,50,3]
            ctx_steps.append(ts)
        num_nodes = len(res_trajs)
        edge_index = torch.LongTensor(list(permutations(range(num_nodes), 2))).t().contiguous()
        res_trajs = np.asarray(res_trajs, np.float32)#转换为np数组
        res_gt = res_trajs[:, 20:].copy()#取出标签 需要预测的后三秒
        # positions = res_trajs[:,:,:2]
        origin = res_trajs[0, 19, :2].copy()#取出agent的最后观察点的位置坐标 
        av_df = df[df['OBJECT_TYPE'] == 'AV'].iloc
        origins = torch.tensor([av_df[19]['X'], av_df[19]['Y']], dtype=torch.float)
        av_heading_vector = origins - torch.tensor([av_df[18]['X'], av_df[18]['Y']], dtype=torch.float)
        thetas = torch.atan2(av_heading_vector[1], av_heading_vector[0])
        pre = res_trajs[0, 19, :2] - res_trajs[0, 18, :2]#航向角
        theta = np.arctan2(pre[1], pre[0])
        rotation = np.asarray([[np.cos(theta), -np.sin(theta)],
                             [np.sin(theta), np.cos(theta)]], np.float32)   #旋转参数

        res_trajs[:, :, :2] = np.dot(res_trajs[:, :, :2] - origin, rotation) #归一化
        res_trajs[np.where(res_trajs[:, :, 2] == 0)] = 0 #如果最后一维标志维度为0，坐标也记为0 
        
        res_fut_trajs = res_trajs[:, 20:].copy()#后30个为预测轨迹
        res_trajs = res_trajs[:, :20].copy()#前20个为历史轨迹坐标点

        res = np.zeros((res_trajs.shape[0],res_trajs.shape[1]-1,res_trajs.shape[2]))
        v = np.zeros((res_trajs.shape[0],res_trajs.shape[1],res_trajs.shape[2]-1))
        car_feats = []  
        for i in range(len(res)):
            
            # Replace  0 in first dimension with 2
            diff = res_trajs[i, 1:, :2] - res_trajs[i, :-1, :2]#得到轨迹向量[19,2] 两点之间位移向量
            position = res_trajs[i,:,:2]# 绝对位置坐标[20，2]
            feat = np.concatenate([np.zeros([1,2]),diff],axis=0) #接一个0向量代表第一秒位移 凑够20
            feats = np.concatenate([feat,position],axis=-1) #特征和绝对位置拼接凑特征向量

            car_feats.append(feats)
            # Sliding window (size=2) with the valid flag
            valid = np.convolve(res_trajs[i, :, 2], np.ones(2), "valid")#判断20个轨迹点是否有效，相邻有效的轨迹点得到的向量也有效（==2）
            # Valid entries have the sum=2 (they get flag=1=valid), unvalid entries have the sum=1 or sum=2 (they get flag=0)
            valid = np.select(
                [valid == 2, valid == 1, valid == 0], [1, 0, 0], valid)#np.select([条件1，条件2，...]，[操作1，操作2，...])当条件为True,执行对应操作 valid表示没有对应条件保持原值

            res[i, :, :2] = diff
            res[i, :, 2] = valid
            v[i, :, : ] = feat
            # v[i,res[i,:,2] == 0] = 0
            # Set zeroes everywhere, where third dimension is = 0 (invalid)
            res[i, res[i, :, 2] == 0] = 0 #如果flag==0则代表该轨迹向量无效
        padding_mask = torch.ones(num_nodes, 50, dtype=torch.bool)
        bos_mask = torch.zeros(num_nodes, 20, dtype=torch.bool)
        # timestamps = list(np.sort(df['TIMESTAMP'].unique()))
        timestamps = list(agt_ts)        
        edge_index = torch.LongTensor(list(permutations(range(num_nodes), 2))).t().contiguous()
        x = torch.zeros(num_nodes, 50, 2, dtype=torch.float)
        padding_mask = torch.ones(num_nodes, 50, dtype=torch.bool)
        historical_timestamps = timestamps[: 20]
        historical_df = df[df['TIMESTAMP'].isin(historical_timestamps)]
        actor_ids = idlist
        id = [uuid.UUID(x).fields[-1] for x in idlist]
        trackid = torch.tensor(id,dtype=torch.long)
        rotate_mat = torch.tensor([[torch.cos(thetas), -torch.sin(thetas)],
                                [torch.sin(thetas), torch.cos(thetas)]])
        rotate_angles = torch.zeros(num_nodes, dtype=torch.float)
        for actor_id, actor_df in df[df['TRACK_ID'].isin(actor_ids)].groupby('TRACK_ID'):
            node_idx = actor_ids.index(actor_id)
            node_steps = [timestamps.index(timestamp) for timestamp in actor_df['TIMESTAMP']]
            padding_mask[node_idx, node_steps] = False
            if padding_mask[node_idx, 19]:  # make no predictions for actors that are unseen at the current time step
                padding_mask[node_idx, 20:] = True
            xy = torch.from_numpy(np.stack([actor_df['X'].values, actor_df['Y'].values], axis=-1)).float()
            x[node_idx, node_steps] = torch.matmul(xy - origin, rotate_mat)
            node_historical_steps = list(filter(lambda node_step: node_step < 20, node_steps))
            if len(node_historical_steps) > 1:  # calculate the heading of the actor (approximately)
                heading_vector = x[node_idx, node_historical_steps[-1]] - x[node_idx, node_historical_steps[-2]]
                rotate_angles[node_idx] = torch.atan2(heading_vector[1], heading_vector[0])
            else:  # make no predictions for the actor if the number of valid time steps is less than 2
                padding_mask[node_idx, 20:] = True
        
        bos_mask[:, 0] = ~padding_mask[:, 0]
        bos_mask[:, 1: 20] = padding_mask[:, : 19] & ~padding_mask[:, 1: 20]
        positions = x.clone()
        
        x[:, 20:] = torch.where((padding_mask[:, 19].unsqueeze(-1) | padding_mask[:, 20:]).unsqueeze(-1),
                                torch.zeros(num_nodes, 30, 2),
                                x[:, 20:] - x[:, 19].unsqueeze(-2))
        x[:, 1: 20] = torch.where((padding_mask[:, : 19] | padding_mask[:, 1: 20]).unsqueeze(-1),
                                torch.zeros(num_nodes, 19, 2),
                                x[:, 1: 20] - x[:, : 19])
        x[:, 0] = torch.zeros(num_nodes, 2)
        id_pd = pd.Series(id)
        ids = pd.factorize(id_pd)[0]
        answer = dict()
        answer['feat'] =np.stack(car_feats)
        answer["argo_id"] = argo_id
        answer["city"] = city
        answer["trajs"] = res_trajs[:,:,:2]
        answer["fut_trajs"] = res_fut_trajs[: ,:,:2]
        answer['steps'] = ctx_steps
        answer["gt"] = res_gt[:, :, :2]
        answer['padding_mask'] = padding_mask
        answer['bos_mask'] = bos_mask
        answer["displ"], answer["centers"] = np.float32(res), res[:, -1, :2] #车辆的轨迹向量特征，第20步的位置坐标
        answer['v'] = v
        # answer['displ'] = np.stack(answer['displ'])
        answer["origin"] = origin
        # We already return the inverse transformation matrix
        answer["rotation"] = np.linalg.inv(rotation) #计算逆矩阵
        answer['num_nodes'] = num_nodes
        answer['edge_index'] = edge_index ## [2,NxN-1]
        answer['positions'] = positions #len(B),每一个是[A,50,2] 为什么变成list了
        answer['x'] = x[:, : 20] #[N, 20, 2]
        answer['rotate_angles'] = rotate_angles
        answer['trackid'] = trackid
        answer['ids'] = ids
        return answer
    
    def get_mask(self,data):
        pre_traj_masks = []
        # paddings = []
        for step in data['steps']:
            traj_mask = np.zeros(50,np.bool_)
            for i in step:
                traj_mask[i] = True
            pre_traj_mask = traj_mask[:20]
            # padding = traj_mask

            pre_traj_masks.append(pre_traj_mask)
            # paddings.append(padding)
        data['pre_traj_masks'] = np.stack(pre_traj_masks) #记录车辆前20步中哪些信息缺失 [A,20]
        # data['padding_mask'] = np.stack(paddings)
        # data['fut_traj_masks'] = fut_traj_masks #记录车辆后30步中哪些信息缺失 [A,30]
        
        return data

    @staticmethod
    def save(dataset, n_cpus, chunksize):
        with Pool(n_cpus) as p:
            preprocessed = list(tqdm(p.imap(dataset.__getitem__, [
                            *range(len(dataset))], chunksize), total=len(dataset)))
        result = []
        for x in preprocessed:    
            if len(x['steps'][0]) == 100 :
                result.append(x)
        os.makedirs(os.path.dirname(dataset.savepath), exist_ok=True)
        with open(dataset.savepath, 'wb') as f:
            pickle.dump(result, f)
            
if __name__ == '__main__':
    args = parser.parse_args()

    train_dataset = Argodataset(args.train_split,args.train_split_pre,args)
    val_dataset = Argodataset(args.val_split,args.val_split_pre,args)
    # test_dataset = Argodataset(args.test_split,args.test_split_pre,args)

    Argodataset.save(train_dataset,args.n_cpus, args.chunksize)
    Argodataset.save(val_dataset,args.n_cpus, args.chunksize)
    # Argodataset.save(test_dataset,args.n_cpus, args.chunksize)
