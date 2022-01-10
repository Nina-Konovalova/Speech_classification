# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 14:26:38 2021

@author: nina_
"""

from tqdm.notebook import tqdm
from sklearn.metrics import accuracy_score
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import Datasets
from models.CNN_BLSTM_SELF_ATTN import CNN_BLSTM_SELF_ATTN
from wav_to_mel import Wav_to_mel
import warnings
warnings.filterwarnings(action='ignore')
import sys


class Train ():
    def __init__(self, path_for_data, length_to_padd,
                 n_epochs=10,
                 batch_size=128,
                 device="cuda:0",  info=True, writer=None):

        self.device = device
        self.info = info
        self.writer = writer

        self.path_for_data, self.length_to_padd = path_for_data, length_to_padd

        self.model = CNN_BLSTM_SELF_ATTN(input_size=128, cnn_filter_size=3, num_layers_lstm=2,
                                         num_heads_self_attn=16, hidden_size_lstm=64, num_emotion_class=6,
                                         num_gender_class=2, num_age_class=3)

        self.model = self.model.to(self.device)
        self.n_epochs = n_epochs

        train_files, val_files = Datasets(self.path_for_data).label_enconding()
        self.train_files = train_files
        self.val_files = val_files
        self.batch_size = batch_size

        self.criterium = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, betas=(0.9, 0.98), eps=1e-9)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=15, gamma=0.1)
        
        
    def collate_batch_train(self, batch):
            arr = []
        
            labels_gender = torch.LongTensor([label for _, label, _, _ in batch])
            labels_age = torch.LongTensor([label for _, _, label, _ in batch])
            labels_emotion = torch.LongTensor([label for _, _, _, label in batch])
            for train_path in batch:
                array = Wav_to_mel(train_path[0]).final_mels(scaling=True, padding=False, augmentation=True,
                                                             length_to_padd=self.length_to_padd)
                arr.append(array)

            arr_pad = torch.Tensor(arr).to(self.device)
        
            return [arr_pad, labels_gender,  labels_age, labels_emotion]

    def collate_batch_val(self, batch):
            arr = []

            labels_gender = torch.LongTensor([label for _, label, _, _ in batch])
            labels_age = torch.LongTensor([label for _, _, label, _ in batch])
            labels_emotion = torch.LongTensor([label for _, _, _, label in batch])
            for train_path in batch:
                array = Wav_to_mel(train_path[0]).final_mels(scaling=True, padding=True, augmentation=False,
                                                             length_to_padd=length_to_padd)
                arr.append(array)

            arr_pad = torch.Tensor(arr).to(self.device)

            return [arr_pad, labels_gender, labels_age, labels_emotion]
    
    def train(self):
        

        train_dataloader = DataLoader(self.train_files, batch_size=self.batch_size, shuffle=True, 
                                      collate_fn=self.collate_batch_train)

        val_dataloader = DataLoader(self.val_files, batch_size=self.batch_size, shuffle=False, 
                                    collate_fn=self.collate_batch_val)
        
        if self.info:
            print(f"Device: {self.device}")
            print(f"Model has {self.count_parameters(self.model):,} trainable parameters")

        best_val_acc = float('-inf')

        alpha, beta, gamma = 1, 1, 1

        train_accuracy_gender = []
        train_accuracy_age = []
        train_accuracy_emotion = []
        train_accuracy_all = []   
        
        val_accuracy_gender = []
        val_accuracy_age = []
        val_accuracy_emotion = []
        val_accuracy_all = []
        
        #loss
        val_loss_gender = []
        val_loss_age = []
        val_loss_emotion = []
        val_loss_all = []   
        
        train_loss_gender = []
        train_loss_age = []
        train_loss_emotion = []
        train_loss_all = []   

        for epoch in tqdm(range(self.n_epochs)):
        
                train_gender_loss, train_emotion_loss, train_age_loss, train_all_loss,\
                train_gender_acc,  train_emotion_acc,  train_age_acc, train_all_acc = self.run_epoch(self.model,
                                                                                                     train_dataloader,
                                                                                                     alpha, beta, gamma,
                                                                                                     self.criterium,
                                                                                                     self.optimizer,
                                                                                                     phase='train',
                                                                                                     epoch=epoch,
                                                                                                     writer=None)
              

                val_gender_loss, val_emotion_loss, val_age_loss, val_all_loss,\
                val_gender_acc,  val_emotion_acc,  val_age_acc, val_all_acc = self.run_epoch(self.model,
                                                                                             val_dataloader,
                                                                                             alpha, beta, gamma,
                                                                                             self.criterium,
                                                                                             None,
                                                                                             phase='val',
                                                                                             epoch=epoch,
                                                                                             writer=None)
                self.scheduler.step()
              
                train_accuracy_gender.append(train_gender_acc)
                train_accuracy_age.append(train_age_acc)
                train_accuracy_emotion.append(train_emotion_acc)
                train_accuracy_all.append(train_all_acc)

                val_accuracy_gender.append(val_gender_acc)
                val_accuracy_age.append(val_age_acc)
                val_accuracy_emotion.append(val_emotion_acc)
                val_accuracy_all.append(val_all_acc)

                val_loss_gender.append(val_gender_loss)
                val_loss_age.append(val_age_loss)
                val_loss_emotion.append(val_emotion_loss)
                val_loss_all.append(val_all_loss)

                train_loss_gender.append(train_gender_loss)
                train_loss_age.append(train_emotion_loss)
                train_loss_emotion.append(train_age_loss)
                train_loss_all.append(train_all_loss)

                if val_all_acc > best_val_acc:
                    best_val_acc = val_all_acc
                    torch.save(self.model.state_dict(), logdir + "/TTS1.pth")

                print(f'Epoch: {epoch+1:02}')
                print(f'\t All Train Loss: {train_all_loss:.3f} | All Train Accuracy: {train_all_acc:.3f}')
                print(f'\t All Val Loss: {val_all_loss:.3f} | All Val Accuracy: {val_all_acc:.3f}')

                logdir = '.'

                if self.writer is None:
                      np.save(logdir + '/train_acc_gender.npy', np.array(train_accuracy_gender))
                      np.save(logdir + '/train_acc_age.npy', np.array(train_accuracy_age))
                      np.save(logdir + '/train_acc_emotion.npy', np.array(train_accuracy_emotion))
                      np.save(logdir + '/train_acc_all.npy', np.array(train_accuracy_all))

                      np.save(logdir + '/val_acc_gender.npy', np.array(val_accuracy_gender))
                      np.save(logdir + '/val_acc_age.npy', np.array(val_accuracy_age))
                      np.save(logdir + '/val_acc_emotion.npy', np.array(val_accuracy_emotion))
                      np.save(logdir + '/val_acc_all.npy', np.array(val_accuracy_all))

                      np.save(logdir + '/train_loss_gender.npy', np.array(train_loss_gender))
                      np.save(logdir + '/train_loss_age.npy', np.array(train_loss_age))
                      np.save(logdir + '/train_loss_emotion.npy', np.array(train_loss_emotion))
                      np.save(logdir + '/train_loss_all.npy', np.array(train_loss_all))

                      np.save(logdir + '/val_loss_gender.npy', np.array(val_loss_gender))
                      np.save(logdir + '/val_loss_age.npy', np.array(val_loss_age))
                      np.save(logdir + '/val_loss_emotion.npy', np.array(val_loss_emotion))
                      np.save(logdir + '/val_loss_all.npy', np.array(val_loss_all))
    
    def count_parameters(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def run_epoch(self, model, dataloader,
              alpha, beta, gamma,
              criterium,
              optimizer,
              phase='train',
              epoch=0,
              writer=None):
  
        is_train = (phase == 'train')
        
        if is_train:
            model.train()
        else:
            model.eval()
        
        all_gender_loss = 0
        all_emotion_loss = 0
        all_age_loss = 0
        all_loss = 0
    
        all_gender_acc = 0
        all_emotion_acc = 0
        all_age_acc = 0
        all_acc = 0
    
        with torch.set_grad_enabled(is_train):
            for i, (spectrogram, gender, age, emotion) in enumerate(tqdm(dataloader)):                  
                spectrogram = spectrogram.to(self.device)
                gender, age, emotion =  gender.to(self.device),  age.to(self.device), emotion.to(self.device)
                
                spectrogram.requires_grad = True
                
                pred_gender, pred_age, pred_emotion = model(spectrogram)

                # loss
                emotion_loss = criterium(pred_emotion,emotion.squeeze())
                gender_loss = criterium( pred_gender,gender.squeeze())
                age_loss = criterium( pred_age,age.squeeze())
    
                total_loss = alpha*emotion_loss + beta*gender_loss + gamma*age_loss
                
                if is_train:
                    optimizer.zero_grad()
                    total_loss.backward()
                    optimizer.step()

                all_gender_loss += gender_loss.item()
                all_emotion_loss += emotion_loss.item()
                all_age_loss += age_loss.item()
    
                all_loss += total_loss.item()

                # accuracy
                predictions_emotion = np.argmax(pred_emotion.detach().cpu().numpy(),axis=1)
                predictions_gender = np.argmax(pred_gender.detach().cpu().numpy(),axis=1)
                predictions_age = np.argmax(pred_age.detach().cpu().numpy(),axis=1)
                
                #print(predictions_age, )
                accuracy_emotion = accuracy_score(emotion.detach().cpu().numpy(),predictions_emotion)
                accuracy_gender = accuracy_score(gender.detach().cpu().numpy(),predictions_gender)
                accuracy_age = accuracy_score(age.detach().cpu().numpy(),predictions_age)
    
                total_acc = (accuracy_gender + accuracy_emotion + accuracy_age)
    
    
                all_gender_acc += accuracy_gender
                all_emotion_acc += accuracy_emotion
                all_age_acc += accuracy_age
    
                all_acc += total_acc

            if writer is not None:
                writer.add_scalar(f"gender_loss/{phase}", all_gender_loss / len(dataloader), epoch)
                writer.add_scalar(f"emotion_loss/{phase}", all_emotion_loss / len(dataloader), epoch)
                writer.add_scalar(f"age_loss/{phase}", all_age_loss / len(dataloader), epoch)
                writer.add_scalar(f"all_loss/{phase}", all_loss / len(dataloader), epoch)
    
                writer.add_scalar(f"gender_accuracy/{phase}", all_gender_acc / len(dataloader), epoch)
                writer.add_scalar(f"emotion_accuracy/{phase}", all_emotion_acc / len(dataloader), epoch)
                writer.add_scalar(f"age_accuracy/{phase}", all_age_acc / len(dataloader), epoch)
                writer.add_scalar(f"all_accuracy/{phase}", all_acc / (3*len(dataloader)), epoch)

        return all_gender_loss / len(dataloader), all_emotion_loss / len(dataloader), all_age_loss / len(dataloader),\
               all_loss / len(dataloader),\
               all_gender_acc / len(dataloader),  all_emotion_acc / len(dataloader),  all_age_acc / len(dataloader),\
               all_acc / (3*len(dataloader))


def main():
    path_for_data = sys.argv[1]
    length_to_padd = sys.argv[2]

    Train().train(path_for_data, length_to_padd)


if __name__ == '__main__':
    main()