# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 13:49:28 2021

@author: nina_
"""

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


class Datasets():
    def __init__(self, data_path_csv, age_groups = 3):
        
        self.data_path_csv = data_path_csv
        self.df = pd.read_csv(self.data_path_csv)
        
        #self.df.drop(self.df[self.df.path	== "common_voice_ru_20992882_pad.npy"].index, inplace=True)
        #self.df.drop(self.df[self.df.path	== "common_voice_ru_20991221_pad.npy"].index, inplace=True)
        
       # if age_groups == 3:

       #     self.df["age"][self.df.age == "twenties"] = 'middle'
       #     self.df["age"][self.df.age == "thirties"] = 'middle'
            
       #     self.df["age"][self.df.age == "fourties"] = 'old'
       #     self.df["age"][self.df.age == "fifties"] = 'old'
            
       #     self.df["emotions"][self.df.emotions == "Злость"] = 'Раздражение'
        

    def label_enconding(self):
        
        label_encoder = LabelEncoder()
        
        train_label_ids_gender = label_encoder.fit_transform(self.df.gender)
        train_label_ids_age = label_encoder.fit_transform(self.df.age)
        train_label_ids_emotions = label_encoder.fit_transform(self.df.emotions)
        
        labels_train_gender = torch.LongTensor(train_label_ids_gender)
        labels_train_age = torch.LongTensor(train_label_ids_age)
        labels_train_emotions = torch.LongTensor(train_label_ids_emotions)
        
        
        all_dataset = list(zip(self.df.path,labels_train_gender,labels_train_age, labels_train_emotions))
        
        self.train_files, self.val_files = train_test_split(all_dataset, test_size=0.25)
        
        return self.train_files, self.val_files
        


    


