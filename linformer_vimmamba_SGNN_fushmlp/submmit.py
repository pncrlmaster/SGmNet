import torch
from model.main_model import MFnet
from trainer.scene_trainer import SceneTrainer
import argparse
from data.argov1_process import Argodataset
from util.loss import loss
from util.collect_fn import collate_fn_dict
from torch.utils.data import DataLoader
from tqdm import tqdm
from argoverse.evaluation.competition_util import generate_forecasting_h5
import os


loss_fn = loss()

if __name__ == "__main__":

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    parser = argparse.ArgumentParser()
    parser = MFnet.init_args(parser)
    args = parser.parse_args()

    model = MFnet(args)
    model  = model.to(device)
    model.load_state_dict(torch.load('/home/zzh/hsti/submmit/att_mamba_SGNN*2_mlp/best_DistributedDataParallel.pth'))

    test_dataset = Argodataset(args.val_split, args.val_split_pre, args)

    test_dataloader = DataLoader(test_dataset,batch_size=1,collate_fn=collate_fn_dict)
    files = os.listdir('/home/zzh/dataset/argoverse1/test_obs')
    seq_id = [int(x[:-4]) for x in files]
    output_all = {}
    for i , data  in enumerate(tqdm(test_dataloader)):
        pred_traj,pi,propose_loc = model(data)
        origin = data['origin'][0].to(device)
        rotation = data['rotation'][0].to(device)
        post_pred = (torch.matmul(pred_traj,rotation) + origin ).view(6,30,2)
        output_all[seq_id[i]] = post_pred


    out_path = '/home/zzh/hsti/submmit/att_mamba_SGNN*2_mlp'
    generate_forecasting_h5(output_all,out_path)
        
        