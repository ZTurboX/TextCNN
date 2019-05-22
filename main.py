import const
import utils
import os
import sys
import argparse
import model
import torch
import torch.nn.functional as F
from torch import optim
config=const.Config()

parse=argparse.ArgumentParser()

parse.add_argument('--mode',default='train',help='train/test')
parse.add_argument('--cuda',default=False)
parse.add_argument('--device',default="3")

args=parse.parse_args()

mode=args.mode
use_cuda=args.cuda
device_id=args.device
if use_cuda:
    torch.cuda.manual_seed(1)
    os.environ["CUDA_VISIBLE_DEVICES"] = device_id

model=model.TextCNN(config)
if use_cuda:
    model.cuda()
if mode=="test":
    print("loading model")
    state_dict=torch.load(open(config.model),'rb')
    model.load_state_dict(state_dict)


def train_step(train_data,test_data,optimizer):
    model.train()
    count = 0
    total_loss = 0
    for j in range(0,len(train_data),config.batch_size):
        optimizer.zero_grad()
        print("run batch : %d " %j )
        batch=train_data[j:j+config.batch_size]
        X_tensor,Y_tensor=utils.get_batch(batch,use_cuda)
        logits=model(X_tensor)
        loss=F.cross_entropy(logits,Y_tensor)
        loss.backward()
        optimizer.step()
        print("minibatch : %d , loss : %.5f " % (j, loss.item()))
        total_loss += loss.item()
        count += 1
    print("-------------------------------------------------------------")
    print("avg loss : %.5f" % (total_loss / count))
    print("-------------------------------------------------------------")
    acc = test(test_data)
    return acc

def dev_step(dev_data):
    print("dev the model...")
    model.eval()
    correct=0
    for j in range(0,len(dev_data),config.batch_size):
        batch=dev_data[j:j+config.batch_size]
        X_tensor,Y_tensor=utils.get_batch(batch,use_cuda)
        logits=model(X_tensor)
        predict=torch.max(logits,1)[1]
        for p,g in zip(predict,Y_tensor):
            correct+=1 if p==g else 0
    acc=correct/len(dev_data)
    print("dev model: accuarcy : %.4f " % acc)
    return acc

def test(test_data):
    print("test the model...")
    model.eval()
    correct=0
    for j in range(0,len(test_data),config.batch_size):
        batch=test_data[j:j+config.batch_size]
        X_tensor,Y_tensor=utils.get_batch(batch,use_cuda)
        logits=model(X_tensor)
        predict = torch.max(logits, 1)[1]
        for p, g in zip(predict, Y_tensor):
            correct += 1 if p == g else 0
    acc=correct/len(test_data)
    print("test model: accuarcy : %.4f " % acc)
    return acc



def train(train_data,test_data):
    optimizer=optim.Adam(model.parameters(),lr=config.lr)
    best_acc=0
    for i in range(config.epoch_size):
        print("train epoch : %d " % i)
        acc=train_step(train_data,test_data,optimizer)
        if acc > best_acc:
            best_acc=acc
            print("-----------------------------------------")
            print("best acc : %.3f " % best_acc)
            print("-----------------------------------------")
            print("save model....")
            best_model_file = os.path.join(config.model_file, "epoch-%d_acc-%.3f.pt" % (i, best_acc))
            torch.save(model.state_dict(), best_model_file)






if __name__=='__main__':

    train_data=utils.get_data(config.train_data_file)
    dev_data=utils.get_data(config.dev_data_file)
    test_data=utils.get_data(config.test_data_file)
    if mode=="train":
        train(train_data,test_data)
    elif mode=="test":
        test(test_data)