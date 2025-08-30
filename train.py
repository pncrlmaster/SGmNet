import os
from torch.optim import Adam, AdamW
import numpy as np
import argparse
import json
from util.loss import loss
from util.collect_fn import collate_fn_dict
# from pathlib import Path
# import sys
# sys.path.append(str(Path(__file__).resolve().parents[1]))
from trainer.scene_trainer import SceneTrainer
# from scene_model.scene_model import SceneModel
from tensorboardX import SummaryWriter

from model.main_model import MFnet

from data.argov1_process import Argodataset
# from trainer.optim_schedule import ScheduledOptim

from trainer.loss import SceneLossWithProb
# from trainer.scene_trainer_v2 import SceneTrainer

import torch.optim.lr_scheduler as lr_scheduler

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = MFnet.init_args(parser)  #参数初始化设定

    
    # parser.add_argument('--train_args_path', type=str, default="scene_model_v2.json")
    args = parser.parse_args()

    
    writer = SummaryWriter('/home/zzh/hsti/att_vimmamba_SGNN_mlp/tensorboard/version_3')

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    # loss_fun = SceneLossWithProb(reduction="none")

    # min_loss = np.inf
    # train_args = json.load(open(args.train_args_path, "r"))

    model = MFnet(args)
    

    model_params = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.2fM" % ((model_params)/1e6))  #计算所有参数的数量并以百万计
    
    loss_fun = loss()
    optim = AdamW(model.parameters(),   #优化器
                  lr= args.lr , 
                #   betas=train_args['optimizer']['betas'],  
                  weight_decay= args.wd
                  )
    
    # scheduler = lr_scheduler.ReduceLROnPlateau(optim, mode='min', factor=0.1, patience=5)
    # scheduler = lr_scheduler.StepLR(optim, step_size=2, gamma=0.5)
    # scheduler = lr_scheduler.ExponentialLR(optim, gamma=0.95)
    scheduler = lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs, eta_min=1e-9)  #学习率调度器 余弦退火动态调整学习率 参数：用optim中的学习率 T_max为整轮对整个训练过程调整学习率 学习率最小为eta_min
    

    train_datasets = Argodataset(args.train_split, args.train_split_pre, args)  #训练数据集
    # train_datasets = Argo2Dataset([os.path.join(train_args['train_data_directory'], file_name) for file_name in os.listdir(train_args['train_data_directory'])])
    eval_datasets = Argodataset(args.val_split, args.val_split_pre, args) #验证集
    scene_trainer = SceneTrainer(
        model=model,  
        optimizer=optim, 
        loss_fun=loss_fun, 
        train_dataset=train_datasets, 
        eval_dataset=eval_datasets, 
        test_dataset=None, 
        optm_schedule=scheduler, 
        use_cuda= args.use_cuda, 
        multy_gpu_type= args.multy_gpu_type, 
        checkpoint=None,
        checkpoint_saving_dir=args.save_path, 
        saving_dir=args.modelsave_dir, 
        epochs=args.epochs,
        batch_size=args.batch_size,
        load_path=None,
        collate_fn=collate_fn_dict,
        writer=writer
    )
    
    scene_trainer.do_train()
    #tensorboard --logdir=/home/zzh/hsti/lxy_new/tensorboard/version_2
    #python -m torch.distributed.launch --nproc_per_node=2 --use_env train.py


#修改：tensorboard path ddp/single_gpu gpu_id main_model中的四个路径 batch_size 预处理pkl路径 decoder的聚类路径 scene_trainer,trainer的localrank注释
    #numnodes
    #version_0 0.86
    #version_1 SGNN*3
    #version_2 SGNNimprove 0.85