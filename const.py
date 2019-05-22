import os
import json

class Config:
    def __init__(self):
        home='./'

        self.train_data_file=os.path.join(home,'data','train.txt')
        self.dev_data_file=os.path.join(home,'data','dev.txt')
        self.test_data_file=os.path.join(home,'data','test.txt')

        self.vocab_file=os.path.join(home,'data','vocab.json')
        self.vector_file=os.path.join(home,'data','vectors-300.txt')
        self.model_file=os.path.join(home,'data/output')

        self.embedding_size=300
        self.batch_size=100
        self.epoch_size=100
        self.filter_size=[3,4,5]
        self.filter_num=100
        self.dropout=0.5
        self.lr=0.001

        self.vocab=json.load(open(self.vocab_file,'r',encoding='utf-8'))
        self.vocab_size=len(self.vocab)