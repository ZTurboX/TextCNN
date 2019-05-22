
import random
import json

def split_data():
    data=[]
    for line in open("./data/pos.txt",'r+',encoding='utf-8'):
        line='1 '+line
        data.append(line)

    for line in open("./data/neg.txt",'r+',encoding='utf-8'):
        line='0 '+line
        data.append(line)


    random.shuffle(data)
    print(len(data))
    train=data[:int(len(data)*0.9)]
    random.shuffle(train)
    valid=train[:int(len(train)*0.1)]
    test=data[int(len(data)*0.9):]

    print(len(train))
    print(len(valid))
    print(len(test))

    with open('./data/train.txt','a',encoding='utf-8') as f:
        for item in train:
            f.write(item)
    f.close()

    with open('./data/valid.txt','a',encoding='utf-8') as fs:
        for item in valid:
            fs.write(item)
    fs.close()

    with open('./data/test.txt','a',encoding='utf-8') as fw:
        for item in test:
            fw.write(item)
    fw.close()

def get_split_train():
    data=[]
    with open('./data/train.txt','r',encoding='utf-8') as f:
        for line in f:
            line=line.strip().split()
            line=line[1:]
            line=' '.join(line)
            data.append(line)
    f.close()

    with open('./data/split_data.txt','a',encoding='utf-8') as fs:
        for item in data:
            fs.write(item)
            fs.write('\n')
    fs.close()

def get_vocab():
    vocab=['<unk>']
    with open('./data/train.txt','r',encoding='utf-8') as f:
        for line in f:
            line=line.strip().split()
            line=line[1:]
            for word in line:
                if word not in vocab:
                    vocab.append(word)
    f.close()

    vocab_dict={j:i for i,j in enumerate(vocab)}

    with open('./data/vocab.json','w',encoding='utf-8') as f:
        f.write(json.dumps(vocab_dict,indent=4))
    f.close()




if __name__=='__main__':
    #split_data()
    #get_split_train()
    get_vocab()
