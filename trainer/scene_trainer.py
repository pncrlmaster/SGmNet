from trainer.trainer import Trainer
import torch
from tqdm import tqdm


class SceneTrainer(Trainer):
    def __init__(self, model,  optimizer, loss_fun, train_dataset, eval_dataset, test_dataset, optm_schedule, use_cuda:bool, multy_gpu_type:str, checkpoint=None, checkpoint_saving_dir:str='', saving_dir:str='', epochs:int =40, load_path:str = "", batch_size:int=128, collate_fn=None,writer=None):
        super().__init__(model, 
                    optimizer,
                    loss_fun,
                    train_dataset,
                    eval_dataset,
                    test_dataset,
                    optm_schedule,
                    use_cuda,
                    multy_gpu_type,
                    checkpoint,
                    checkpoint_saving_dir,
                    saving_dir,
                    epochs,
                    load_path,
                    batch_size, 
                    collate_fn,
                    writer)

    
    @torch.no_grad()
    def eval(self, epoch):
        total_loss = 0.0
        num_points = 0
        self.model.eval()
        
        for i, data in enumerate(tqdm(self.eval_loader)):

            pred_traj,pi,propose_loc = self.model(data)

            propose_loss, ade_loss, fde_loss, cls_loss,_ = self.loss_fun(pred_traj,pi,propose_loc,data["fut_trajs"])
            # local_rank = torch.distributed.get_rank()
            # if local_rank == 0:
            self.writer.add_scalar("ade_loss", ade_loss.item(), epoch*1609+i )
            self.writer.add_scalar("fde_loss", fde_loss.item(), epoch*1609+i )
            
            loss = ade_loss + cls_loss + fde_loss + propose_loss
            points = self.eval_loader.batch_size
            num_points += points
            total_loss += loss.item()
            lr =self.optimizer.state_dict()['param_groups'][0]['lr']
            print("[Info:eval_Ep_{}_iter_{}: loss: {:.5e};ade: {:.5e}; fde: {:.5e};avg_loss: {:.5e};lr:{:.5e}]".format(epoch, 
                                                                                    i, 
                                                                                    loss.item(),
                                                                                    ade_loss.item(),
                                                                                    fde_loss.item(),
                                                                                    total_loss/num_points,
                                                                                    lr))
        torch.cuda.empty_cache()
        return  total_loss /num_points
    
    def train(self, epoch):
        if self.multy_gpu_type == 'ddp':
            self.train_sample.set_epoch(epoch)
        total_loss = 0.0
        num_points = 0
        self.model.train()
        
        for i, data in enumerate(tqdm(self.train_dataloader)):
            self.optimizer.zero_grad()
            pred_traj,pi,propose_loc = self.model(data)
            propose_loss, ade_loss, fde_loss, cls_loss,_= self.loss_fun(pred_traj,pi,propose_loc,data["fut_trajs"])
            # local_rank = torch.distributed.get_rank()
            # if local_rank == 0:
                # self.writer.add_scalar("ade_loss", ade_loss.item(),epoch)
                # self.writer.add_scalar("fde_loss", fde_loss.item(),epoch)
                
            loss = ade_loss + cls_loss + fde_loss + propose_loss
            loss.backward()
            self.optimizer.step()
            points = self.train_dataloader.batch_size
            num_points +=points
            total_loss += loss.item()
            lr =self.optimizer.state_dict()['param_groups'][0]['lr']
            print("[Info:eval_Ep_{}_iter_{}: loss: {:.5e};ade: {:.5e}; fde: {:.5e};avg_loss: {:.5e};lr:{:.5e}]".format( 
                                                                                    epoch,
                                                                                    i, 
                                                                                    loss.item(),
                                                                                    ade_loss.item(),
                                                                                    fde_loss.item(),
                                                                                    total_loss/num_points,
                                                                                    lr))
        return  total_loss/num_points
            
    def do_train(self):
        super().do_train()