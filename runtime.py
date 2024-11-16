from gpt_modeling import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import re
import os
import torch.distributed as dist
import datetime



def init_model(model_config):
    model=[]
    for i,j in model_config.items():
        if i=="em_tokn":
            mdl=nn.Embedding(j[0], j[1])
        elif i=="em_pos":
            mdl=nn.Embedding(j[0], j[1])
        elif i=="ln":
            mdl=nn.LayerNorm(j[0])
        elif i=="lm_head":
            mdl=nn.Linear(j[0],j[1])
        elif (re.search("decoder",i)).group()=="decoder": # 可能会报错
            mdl=Block(j[0],j[1])
        model.append(mdl)
    return model

def is_Transfer(iter):
    if iter==5:
        return True,[[2,1]]
    elif iter==10:
        return True,[[1,2],[3,2]]
    else:
        return False,[]

class Stage:
    def __init__(self,ID,model,model_idx,learning_rate,device,batch_size):
        self.stage_ID=ID
        self.device=device
        self.model_idx=model_idx
        self.is_training=True
        self.sub_model= [model[i] for i in model_idx] # model.type:list
        self.optimizer_list= [optim.Adam(model[i].parameters(), lr=learning_rate) for i in model_idx]
        self.out_Y=[]
        self.out_X=[]
        self.out_x=torch.zeros(batch_size,128,128).to(device)
        self.grad_y=torch.zeros(batch_size,128,128).to(device)
        self.lossi=[]

    def to(self,device):
        for layer in self.sub_model:
            layer.to(device)
    
    def eval(self):
        for layer in self.sub_model:
            layer.eval()
    
    def train(self):
        for layer in self.sub_model:
            layer.train()

    def zero_grad(self):
        for optm in self.optimizer_list:
            optm.zero_grad()

    def update_out_Y(self,out_y):
        self.out_Y.append(out_y)

    def update_out_X(self,out_x):
        self.out_X.append(out_x)

    def forward(self,x):
        if self.stage_ID==1:
            B, T = x.shape
            # 定义词元的位置，形状为(T)
            pos = torch.arange(0, T, dtype=torch.long, device=x.device)
        # 词元语义特征
            tok_emb = self.sub_model[0](x)       # (B, T,  C)
        # 位置特征
            pos_emb = self.sub_model[1](pos)  # (   T,  C)
            x = tok_emb + pos_emb
            self.update_out_Y(x)
            for i in range(2,len(self.sub_model)):
                x=torch.tensor(data=x, dtype=torch.float, requires_grad=True, device=self.device)
                self.update_out_X(x)
                x=self.sub_model[i](x)
                self.update_out_Y(x)
            return x
        else:
            for layer in self.sub_model:
                x=torch.tensor(data=x, dtype=torch.float, requires_grad=True, device=self.device)
                self.update_out_X(x)
                x=layer(x) 
                self.update_out_Y(x)                
            return x
        
    def forward_send(self):

        dist.send(tensor=self.out_Y[-1],dst=self.stage_ID,tag=self.stage_ID)

    def forward_recv(self):
        dist.recv(tensor=self.out_x,src=self.stage_ID-2,tag=self.stage_ID-1)
        self.out_x.to(self.device)
        
    def backward_tail(self,labels):
        # print(self.out_Y[-1].shape)
        logits = self.out_Y[-1].transpose(-2, -1)
        loss = F.cross_entropy(logits, labels)
        self.lossi.append(loss.item())
        loss.backward()
        self.optimizer_list[-1].step()
        # print(self.out_X[-1].grad.shape)
        # print(type(self.out_X[0].grad))
        for i in range(2,len(self.out_Y)+1):
            # print(i)
            # print(len(self.out_X))
            # print(self.out_X[-i].shape)
            # # print(self.out_X[-i].grad.shape)
            # print(self.out_Y[-i].shape)
            self.out_Y[-i].backward(self.out_X[-i+1].grad) 
            self.optimizer_list[-i].step() 

    def backward(self):
        for i in range(1,len(self.out_Y)+1):
            if i==1:
                self.out_Y[-i].backward(self.grad_y)
                self.optimizer_list[-i].step() 
            else:               
                self.out_Y[-i].backward(self.out_X[-i+1].grad)
                self.optimizer_list[-i].step()
            
    def backward_send(self):
        dist.send(tensor=self.out_X[0].grad,dst=self.stage_ID-2,tag=self.stage_ID)

    def backward_recv(self):
        dist.recv(tensor=self.grad_y,src=self.stage_ID,tag=self.stage_ID+1)
        self.grad_y.to(self.device)

    def __send_update(self,index):
        self.sub_model[index].to('cpu')
        self.model_idx.pop(index)
        self.sub_model.pop(index)
        if len(self.model_idx)==0:
            self.is_training=False
        op=self.optimizer_list.pop(index)
        del op

    def send_weight(self,S_R_Pair):
        if S_R_Pair[0]>S_R_Pair[1]:
            for param in self.sub_model[0].parameters():
                dist.send(tensor=param.data,dst=self.stage_ID-2,tag=self.stage_ID)
                param.data=param.data.to('cpu')
            self.__send_update(0)
        if S_R_Pair[0]<S_R_Pair[1]:
            for param in self.sub_model[-1].parameters():
                dist.send(tensor=param.data,dst=self.stage_ID,tag=self.stage_ID)
                param.data=param.data.to('cpu')
            self.__send_update(-1)

    def __recv_update(self,model,index):
        if index==-1:
            idx=self.model_idx[-1]
            self.model_idx.append(idx+1)
            model[idx+1].to(self.device)
            self.sub_model.append(model[idx+1])
            optimzer=optim.Adam(model[idx+1].parameters(), lr=learning_rate)
            self.optimizer_list.append(optimzer)
            if len(self.model_idx)>0:
                self.is_training=True
        if index==0:
            idx=self.model_idx[0]
            self.model_idx.insert(0,idx-1)
            model[idx-1].to(self.device)
            self.sub_model.insert(0,model[idx-1])
            optimzer=optim.Adam(model[idx-1].parameters(), lr=learning_rate)
            self.optimizer_list.insert(0,optimzer)
            if len(self.model_idx)>0:
                self.is_training=True

    def recv_weight(self,model,S_R_Pair):
        if S_R_Pair[0]>S_R_Pair[1]:
            self.__recv_update(model,-1)
            for param in model[self.model_idx[-1]+1].parameters():
                temp_recv=torch.zeros_like(param.data).to(self.device)
                dist.recv(tensor=temp_recv,src=self.stage_ID,tag=self.stage_ID+1)
                param.data=temp_recv
            
        if S_R_Pair[0]<S_R_Pair[1]:
            self.__recv_update(model,0)
            for param in model[self.model_idx[0]+1].parameters():
                temp_recv=torch.zeros_like(param.data).to(self.device)
                dist.recv(tensor=temp_recv,src=self.stage_ID-2,tag=self.stage_ID-1)
                param.data=temp_recv
            

    def clear(self):
        self.out_Y.clear()
        self.out_X.clear()
        
# 通信域创建
env_dict = {
        key: os.environ[key]
        for key in ("MASTER_ADDR", "MASTER_PORT", "WORLD_SIZE", "LOCAL_WORLD_SIZE")
    }
print(f"[{os.getpid()}] Initializing process group with: {env_dict}")
dist.init_process_group(backend="NCCL", timeout=datetime.timedelta(seconds=30)) # gloo
global_rank = int(os.environ["RANK"])

if global_rank==0:
    DEVICE = 'cuda:0'
elif global_rank==1:
    DEVICE = 'cuda:1'
elif global_rank==2:
    DEVICE = 'cuda:2'
elif global_rank==3:
    DEVICE = 'cuda:3'

print(DEVICE)


# 将数据分为训练集和测试集
tokenized = datasets.train_test_split(test_size=0.1, seed=1024, shuffle=True)
# 将文本转换为训练数据，里面包含inputs和labels
tokenized = tokenized.map(process, batched=True, remove_columns=datasets.column_names)
tokenized.set_format(type='torch', device=DEVICE)
print(tokenized['train']['inputs'].shape, tokenized['train']['labels'].shape)
# 构建数据读取器
train_loader = DataLoader(tokenized['train'], batch_size=batch_size, shuffle=True)
test_loader = DataLoader(tokenized['test'], batch_size=batch_size, shuffle=True)


model_config={"em_tokn":[98,128],"em_pos":[128,128],"decoder1":[128,8],"decoder2":[128,8],"decoder3":[128,8],
              "decoder4":[128,8],"decoder5":[128,8],"decoder6":[128,8],"decoder7":[128,8],"decoder8":[128,8],
              "decoder9":[128,8],"ln":[128],"lm_head":[128,98]}

gpt=init_model(model_config=model_config)
model_idx1=[0,1,2,3]
model_idx2=[4,5,6]
model_idx3=[7,8]
model_idx4=[9,10,11,12]
s1=Stage(1,gpt,model_idx1,learning_rate,DEVICE,batch_size)
s2=Stage(2,gpt,model_idx2,learning_rate,DEVICE,batch_size)
s3=Stage(3,gpt,model_idx3,learning_rate,DEVICE,batch_size)
s4=Stage(4,gpt,model_idx4,learning_rate,DEVICE,batch_size)

Stage_list=[s1,s2,s3,s4]

for i in range(len(Stage_list)):
    if i==global_rank:
        Stage_list[i].to(DEVICE)



for i, data in tqdm(enumerate(train_loader, 0)):
    if global_rank==0:
        if Stage_list[0].is_training:
            inputs, labels = data['inputs'], data['labels']
            Stage_list[0].zero_grad()
            out_y = Stage_list[0].forward(inputs)
            Stage_list[0].forward_send()

            Stage_list[0].backward_recv()
            Stage_list[0].backward()

            Stage_list[0].clear()

        flag,transfer_list=is_Transfer(i)
        if flag:
            for transfer in transfer_list:
                if global_rank+1 in transfer:
                    if global_rank+1==transfer[0]:
                        Stage_list[0].send_weight(transfer)
                    else:
                        Stage_list[0].recv_weight(gpt,transfer)
                    print(global_rank)
                    print("Transfer Finished")

    elif global_rank>0 and global_rank<len(Stage_list)-1:
        if Stage_list[global_rank].is_training:
            Stage_list[global_rank].zero_grad()
            Stage_list[global_rank].forward_recv()
            Stage_list[global_rank].forward(Stage_list[global_rank].out_x)
            Stage_list[global_rank].forward_send()

            Stage_list[global_rank].backward_recv()
            Stage_list[global_rank].backward()
            Stage_list[global_rank].backward_send()

            Stage_list[global_rank].clear()

        flag,transfer_list=is_Transfer(i)
        if flag:
            for transfer in transfer_list:
                if global_rank+1 in transfer:
                    if global_rank+1==transfer[0]:
                        Stage_list[global_rank].send_weight(transfer)
                    else:
                        Stage_list[global_rank].recv_weight(gpt,transfer)
                    print(global_rank)
                    print("Transfer Finished")
    else:
        if Stage_list[global_rank].is_training:
            inputs, labels = data['inputs'], data['labels']
            Stage_list[global_rank].zero_grad()
            Stage_list[global_rank].forward_recv()
            Stage_list[global_rank].forward(Stage_list[global_rank].out_x)

            Stage_list[global_rank].backward_tail(labels)
            Stage_list[global_rank].backward_send()

            Stage_list[global_rank].clear()

        flag,transfer_list=is_Transfer(i)
        if flag:
            for transfer in transfer_list:
                if global_rank+1 in transfer:
                    if global_rank+1==transfer[0]:
                        Stage_list[global_rank].send_weight(transfer)
                    else:
                        Stage_list[global_rank].recv_weight(gpt,transfer)
                    print(global_rank)
                    print("Transfer Finished")


# 测试样本生成 
# begin_text = torch.tensor(tok.encode('def'), device=DEVICE).unsqueeze(0)
# print(''.join(tok.decode(generate_batch(Stage_list, begin_text))))
# for i in s2.sub_model:
#     print(i)
