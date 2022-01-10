# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 13:34:41 2021

@author: nina_
"""
import torch 
import torch.nn as nn


class CNN_BLSTM_SELF_ATTN(torch.nn.Module):
    def __init__(self, input_size, cnn_filter_size, \
                     num_layers_lstm, hidden_size_lstm,\
                     num_heads_self_attn, 
                     num_emotion_class, num_gender_class, num_age_class):
        
        super(CNN_BLSTM_SELF_ATTN, self).__init__()
        
        self.input_size=input_size
        self.cnn_filter_size=cnn_filter_size
        
        self.num_layers_lstm=num_layers_lstm
        self.hidden_size_lstm=hidden_size_lstm
        
        self.num_heads_self_attn=num_heads_self_attn
        
        self.num_emotion_class=num_emotion_class
        self.num_gender_class=num_gender_class
        self.num_age_class=num_age_class
        
        self.conv_1 = nn.Sequential(
            nn.Conv1d(self.input_size,self.cnn_filter_size,3,1),
            nn.MaxPool1d(2)
        )
        
        
        self.conv_2 = nn.Sequential(
            nn.Conv1d(self.cnn_filter_size,self.cnn_filter_size,3,1),
            nn.MaxPool1d(2)
        )
        
        self.conv_3 = nn.Sequential(
            nn.Conv1d(self.cnn_filter_size,self.cnn_filter_size,3,1),
            nn.MaxPool1d(2)
        )
        
        ## LSTM
        self.lstm = nn.LSTM(input_size=self.cnn_filter_size, hidden_size = self.hidden_size_lstm,num_layers = self.num_layers_lstm, \
                            bidirectional=True, dropout = 0.5, batch_first=True)
        ## Transformer
        self.encoder_layer = nn.TransformerEncoderLayer(d_model = self.hidden_size_lstm*2, dim_feedforward = 512, nhead = self.num_heads_self_attn)
        
        ## Linear predictions
        self.gender_layer  = nn.Linear(self.hidden_size_lstm*4, self.num_gender_class)
        self.emotion_layer = nn.Linear(self.hidden_size_lstm*4, self.num_emotion_class)
        self.age_layer = nn.Linear(self.hidden_size_lstm*4, self.num_age_class)


    def forward(self,inputs):
        out = self.conv_1(inputs)
        
        out = self.conv_2(out)

        out = self.conv_3(out)
        
        out = out.permute(0, 2, 1)
        out, (final_hidden_state, final_cell_state) = self.lstm(out)
        
        out = self.encoder_layer(out)
        
        mean = torch.mean(out,1)
        std = torch.std(out,1)
        stat = torch.cat((mean,std),1)
        
        pred_gender=self.gender_layer(stat)
        pred_age = self.age_layer(stat)
        pred_emotion = self.emotion_layer(stat)
        
        return pred_gender, pred_age, pred_emotion