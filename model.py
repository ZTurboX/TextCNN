import torch
import torch.nn as nn
import torch.nn.functional as F
import const
import numpy as np

class TextCNN(nn.Module):
    def __init__(self,config:const.Config()):
        super(TextCNN, self).__init__()
        self.embedding_size=config.embedding_size
        self.filter_size=config.filter_size
        self.filter_num=config.filter_num
        self.vocab_size=config.vocab_size
        self.vocab=config.vocab
        self.word_embedding=nn.Embedding(self.vocab_size,self.embedding_size)
        self.load_pretrained_embedding(config)
        self.convs=nn.ModuleList(
            [nn.Conv2d(1,self.filter_num,(size,self.embedding_size)) for size in self.filter_size]
        )
        self.dropout=nn.Dropout(config.dropout)
        self.fc=nn.Linear(len(self.filter_size)*self.filter_num,2)

    def load_pretrained_embedding(self, config: const.Config):
        words_vectors = {}
        for line in open(config.vector_file, encoding='utf-8').readlines():
            items = line.strip().split()
            words_vectors[items[0]] = [float(x) for x in items[1:]]
        embeddding_matrix = np.asarray(np.random.normal(0, 0.9, (self.vocab_size, 300)), dtype='float32')

        for word in self.vocab:
            if word in words_vectors:
                embeddding_matrix[self.vocab[word]] = words_vectors[word]
        self.word_embedding.weight = nn.Parameter(torch.tensor(embeddding_matrix))

    def forward(self,input):
        x=self.word_embedding(input)
        x=x.unsqueeze(1)
        x=[F.relu(conv(x)).squeeze(3) for conv in self.convs]
        x=[F.max_pool1d(item,item.size(2)).squeeze(2) for item in x]
        x=torch.cat(x,1)
        x=self.dropout(x)
        logits=self.fc(x)
        return logits


