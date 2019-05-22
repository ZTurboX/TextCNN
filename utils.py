import const
import json
from torch.autograd import Variable
import torch
config=const.Config()


def get_data(data_file):

    all_data=[]
    with open(data_file,'r',encoding='utf-8') as f:
        for line in f:
            line=line.strip().split()
            label=line[0]

            sentence=line[1:]
            data=[]
            word_list=[]
            for word in sentence:
                if word in config.vocab:
                    word_list.append(config.vocab[word])
                else:
                    word_list.append(config.vocab["<unk>"])

            data.append(word_list)
            data.append(int(label))
            all_data.append(data)

    return all_data

def convert_long_tensor(var,use_cuda):
    var=torch.LongTensor(var)
    if use_cuda:
        var=var.cuda(async=True)
    return var

def convert_long_variable(var,use_cuda):
    return Variable(convert_long_tensor(var,use_cuda))


def get_batch(batch,use_cuda):
    batch=sorted(batch,key=lambda x:len(x[0]),reverse=True)
    X_len=[len(s[0]) for s in batch]
    max_batch_sent_len=max(X_len)
    X=[]
    Y=[]
    for s in batch:
        X.append(s[0]+[config.vocab["<unk>"]]*(max_batch_sent_len-len(s[0])))
        Y.append(s[1])
    X_tensor=convert_long_variable(X,use_cuda)
    Y_tensor=convert_long_tensor(Y,use_cuda)
    return X_tensor,Y_tensor



