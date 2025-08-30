import torch 
import torch.nn as nn
from model.layer import att,CrossAttentionLayer


class Fusion_gate(nn.Module):
    def __init__(self,args):
        super(Fusion_gate,self).__init__()
        # self.args = args
        # self.sub_layers = 3
        # self.m = nn.Parameter(torch.randn(args.hidden_size))
        # self.cross_att = att(d_model=args.hidden_size)
        self.fusion1 = nn.Linear(args.hidden_size,args.hidden_size)
        self.fusion2 = nn.Linear(args.hidden_size,args.hidden_size)
        self.sig = nn.Sigmoid()


    def forward(self,input1,input2):# input[B,N,H]
        z = self.sig(self.fusion1(input1)+self.fusion2(input2))
        fusion_out = z*input1 + (1-z)*input2 #调整空间和时间注意力的占比权重
        # batch_size,num_nodes,_ = input1.shape
        # fusion_out = self.m.view(1,1,self.args.hidden_size).repeat(batch_size,num_nodes,1)#[B,N,H]
        # for i in range(self.sub_layers):
        #     fusion_out = self.cross_att(fusion_out,input1,input1)
        #     fusion_out = self.cross_att(fusion_out,input2,input2)
        return fusion_out
    
class Fusion_gate2(nn.Module):
    def __init__(self,args):
        super(Fusion_gate2,self).__init__()
        self.cross_att = CrossAttentionLayer(args)
        # self.args = args
        self.sub_layers = 3
        # self.m = nn.Parameter(torch.randn(args.hidden_size))
        # self.cross_att = att(d_model=args.hidden_size)
        self.fusion1 = nn.Linear(args.hidden_size,args.hidden_size)
        self.fusion2 = nn.Linear(args.hidden_size,args.hidden_size)
        self.sig = nn.Sigmoid()


    def forward(self,temp_input,spi_input):
        for i in range(self.sub_layers):
            spi_out = self.cross_att(temp_input,spi_input,spi_input) + spi_input
            temp_out = self.cross_att(spi_input,temp_input,temp_input) + temp_input
        z = self.sig(self.fusion1(spi_out)+self.fusion2(temp_out))
        fusion_out = z*spi_out + (1-z)*temp_out #调整空间和时间注意力的占比权重
        return fusion_out