#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torch import nn
from torch.nn import Sequential
from torchvision import models
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence


# In[ ]:


class encoder(nn.Module):
    def __init__(self,output_dim=1000):
        super(encoder, self).__init__()
        gnet = models.googlenet(pretrained=True)
        for param in gnet.parameters():
            param.requires_grad=False
        self.gnet = Sequential(*list(gnet.children())[:-1])
        self.linear = nn.Linear(gnet.fc.in_features, output_dim)
        self.batchnorm = nn.BatchNorm1d(output_dim)
        #self.weights()
    
#     def weights(self):
#         self.linear.weight.data.xavier_normal_()
#         self.linear.bias.data.fill_(0)
    
    def forward(self,x):
        x = self.gnet(x)
        x = torch.flatten(x,-1)
        x = self.linear(x)
        return x
    
class decoder(nn.Module):
    
    def __init__(self, embed_size, hidden_size, vocab_size, num_layer = 1):
        
        super(decoder, self).__init__()
        self.embeddings = nn.embedding(vocab_size, embed_size)
        self.unit = decoder.nn.LSTM(embed_size, hidden_size, num_layer, batch_first = True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, features, captions, lengths):
        
        embeddings = self. embeddings(captions)
        inputs = torch.cat((features.unsqueeze(1), embeddings), 1)
        packed_ip = pack_padded_sequence(inputs, lengths, batch_first=True)  
        h_state, _ = self.unit(packed_ip)
        outputs = self.linear(h_state[0])
        return outputs

    def sample(self, features, max_len=30):

        output_ids = []
        states = None
        inputs = features.unsqueeze(1)

        for i in range(max_len):
            
            h_state, states = self.unit(inputs, states)
            outputs = self.linear(h_state.squeeze(1))
            predicted = outputs.max(1)[1]
            output_ids.append(predicted)
            inputs = self.embeddings(predicted)
            inputs = inputs.unsqueeze(1)
            
        output_ids = torch.stack(output_ids, 1)
        return output_ids.squeeze()

