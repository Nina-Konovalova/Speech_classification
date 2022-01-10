import torch 
import torch.nn as nn
import os,sys,inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
sys.path.insert(0, parentdir + '/speechbrain')

from speechbrain.pretrained import EncoderClassifier



class Wav2Vec_classification(nn.Module):
    def __init__(self, input_amp_size, num_emo_class, num_gender_class, num_age_class, type_='dense'):
        super(Wav2Vec_classification, self).__init__()

        self.input_amp_size = input_amp_size
        self.num_emo_class = num_emo_class
        self.num_gender_class = num_gender_class
        self.num_age_class = num_age_class
        self.type_ = type_

        self.classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")

        self.dense_layer_1 = nn.Linear(192, 128)
        self.dense_layer_2 = nn.Linear(128, 128)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

        self.gender_layer = nn.Linear(128, self.num_gender_class)
        self.emotion_layer = nn.Linear(128, self.num_emo_class)
        self.age_layer = nn.Linear(128, self.num_age_class)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, inputs):
        embeddings = self.classifier.encode_batch(inputs)
        embeddings = embeddings.squeeze()

        output = torch.tensor(embeddings)
        output = self.dense_layer_1(output)
        pred_gender = (self.gender_layer(output))
        pred_age = (self.age_layer(output))
        pred_emo = (self.emotion_layer(output))
        return pred_gender.squeeze(), pred_age.squeeze(), pred_emo.squeeze()