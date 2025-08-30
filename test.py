import torch
from model.main_model import MFnet
from trainer.scene_trainer import SceneTrainer
import argparse
from data.argov1_process import Argodataset
from test_loss import loss
from util.collect_fn import collate_fn_dict
from torch.utils.data import DataLoader
from tqdm import tqdm

loss_fn = loss()

if __name__ == "__main__":

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    parser = argparse.ArgumentParser()
    parser = MFnet.init_args(parser)
    args = parser.parse_args()

    model = MFnet(args)
    model  = model.to(device)
    model.load_state_dict(torch.load('/home/ubuntu/code/zhuzihan/hsti/att_vimmamba_SGNN_decoder/save_model/version_2/best_DistributedDataParallel.pth'))
    eval_dataset = Argodataset(args.val_split, args.val_split_pre, args)
    #dataloader
    eval_dataloader = DataLoader(eval_dataset,batch_size=32,collate_fn=collate_fn_dict)
    ade = 0
    fde = 0
    mr = 0
    with torch.no_grad():
        for i,data in enumerate(tqdm(eval_dataloader)):
            pred_traj,pi,propose_loc = model(data)
            propose_loss, ade_loss, fde_loss, cls_loss,_,mr_loss= loss_fn(pred_traj,pi,propose_loc,data["fut_trajs"])
            ade += ade_loss
            fde += fde_loss
            mr += mr_loss

        mr = mr/len(eval_dataloader)
        ade = ade/len(eval_dataset)
        fde = fde/len(eval_dataset)

    print("[result: ade: {:.5e}; fde: {:.5e}; MR: {:.5e}".format(ade.item(),fde.item(),mr))
