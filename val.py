import torch
import numpy as np
import librosa

from models.wav2vec_model import Wav2Vec_classification

path = './TTS_wav (3).pth'


class Val():
    def __init__(self):

#         self.model = CNN_BLSTM_SELF_ATTN(input_spec_size=128, cnn_filter_size=3, num_layers_lstm=2,
#                                          num_heads_self_attn=16, hidden_size_lstm=64, num_emo_class=6,
#                                          num_gender_class=2, num_age_class=5)
        self.model = Wav2Vec_classification(input_amp_size=128, num_emo_class=6, num_gender_class=2, num_age_class=5)

        self.model.load_state_dict(torch.load(path, map_location='cpu'))
        self.model.eval()

    def predict(self, path_to_wav):
        #spectrogram = Wav_to_mel(path_to_wav).final_mels(scaling=True, padding=True, augmentation=False,
        #                                                 length_to_padd=1000)
        spectrogram = librosa.load(path_to_wav, 16_000)[0]
        print(spectrogram.shape)
        # if len(spectrogram) < 45_000:
        #     len_pad = 45000 - len(spectrogram)
        #     spectrogram = np.pad(spectrogram, (0, len_pad), mode='reflect')
        #     print(spectrogram.shape)
        # spectrogram = spectrogram[15_000:30_000]

        spectrogram = torch.Tensor(spectrogram)
        spectrogram = torch.unsqueeze(spectrogram, dim=0)
        
        pred_gender, pred_age, pred_emotion = self.model(spectrogram)
        #print(pred_gender)
        
        predictions_emotion = np.argmax(pred_emotion.detach().cpu().numpy())
        #print(predictions_emotion)
        dict_emo = dict(np.load('data_dict/dict_emo.npy', allow_pickle='TRUE').item())
        #print( dict_emo)
        predictions_emotion = dict_emo[predictions_emotion]
        
        
        predictions_gender = np.argmax(pred_gender.detach().cpu().numpy())
        #print(predictions_gender)
        dict_gender = dict(np.load('data_dict/dict_gender.npy', allow_pickle='TRUE').item())
        predictions_gender = dict_gender[predictions_gender]
        
        predictions_age = np.argmax(pred_age.detach().cpu().numpy())
        dict_age = dict(np.load('data_dict/dict_age.npy', allow_pickle='TRUE').item())
        #print(dict_age)
        predictions_age = dict_age[predictions_age]
        
        return  predictions_gender, predictions_age, predictions_emotion