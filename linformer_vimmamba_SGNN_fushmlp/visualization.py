import torch
from model.main_model import MFnet
from trainer.scene_trainer import SceneTrainer
import argparse
from data.argov1_process import Argodataset
from util.loss import loss
from util.collect_fn import collate_fn_dict
from torch.utils.data import DataLoader
from argoverse.map_representation.map_api import ArgoverseMap
from tqdm import tqdm
import copy
import numpy as np
import  matplotlib.pyplot as plt


loss_fn = loss()

if __name__ == "__main__":

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    parser = argparse.ArgumentParser()
    parser = MFnet.init_args(parser)
    args = parser.parse_args()
    am = ArgoverseMap()


    model = MFnet(args)
    model  = model.to(device)
    model.load_state_dict(torch.load('/home/lxy/programs/lxy_new/save_model/best_DistributedDataParallel.pth'))
    eval_dataset = Argodataset(args.val_split, args.val_split_pre, args)
    #dataloader
    eval_dataloader = DataLoader(eval_dataset,batch_size=1,collate_fn=collate_fn_dict)
    lanes = dict()
    for i,data in (enumerate(eval_dataloader)):
###画出轨迹线
        pred_traj,pi,propose_loc = model(data)
        _,_, _, _,best_traj= loss_fn(pred_traj,pi,propose_loc,data["fut_trajs"])
        fut_trajs =data['fut_trajs'][0].cpu()
        plt.figure(dpi=600)
        plt.plot(data['trajs'][0][0][:,0],data['trajs'][0][0][:,1],color='black')#历史轨迹
        plt.plot(fut_trajs[0][:,0],fut_trajs[0][:,1],color='red')#未来真实轨迹
        for pd_traj in pred_traj[0]:
            plt.plot(pd_traj[:,0].cpu().detach().numpy(),pd_traj[:,1].cpu().detach().numpy(),color ='blue') #多模态未来轨迹
        plt.plot(best_traj[0][:,0].cpu().detach().numpy(),best_traj[0][:,1].cpu().detach().numpy(),color='green')#最好的轨迹
        # plt.show()
###### 画地图
        lane_ids = am.get_lane_ids_in_xy_bbox(data['origin'][0][0],data['origin'][0][1],data['city'][0],75)
        lane_ids = copy.deepcopy(lane_ids)
        for lane_id in lane_ids:
            lane = am.city_lane_centerlines_dict[data['city'][0]][lane_id]
            lane = copy.deepcopy(lane)
            centerline = np.dot((lane.centerline - np.array(data['origin'][0])),np.array(data['rotation'][0]))
            polygon = am.get_lane_segment_polygon(lane_id, data['city'][0])
            polygon = copy.deepcopy(polygon)
            lane.centerline = centerline
            lane.polygon = np.dot((polygon[:,:2] - np.array(data['origin'][0])),data['rotation'][0])
            lanes[lane_id] = lane
        lanes_keys = list(lanes.keys())
        for laneid in (lanes_keys):
            centeor= lanes[laneid].centerline
            polygon = lanes[laneid].polygon
            plt.plot(centeor[:,0],centeor[:,1],linewidth=0.1,color='grey')
            plt.plot(polygon[:,0],polygon[:,1],linewidth=0.1,color='grey')
        plt.show()
        pass
# res_trajs[:, :, :2] = np.dot(res_trajs[:, :, :2] - origin, rotation)