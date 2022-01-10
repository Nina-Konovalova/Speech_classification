import speechbrain


import os
import sys
import inspect

from os import listdir
from os.path import isfile, join
from speechbrain.pretrained import SpeakerRecognition



class Verify():
    def __init__(self):

        self.model = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="pretrained_models/spkrec-ecapa-voxceleb")

    def predict(self, path1, path2):
        #score - cosine similarity (?)
        # prediction - whether the same person is on recording or not

        score, prediction = self.model.verify_files(path1,  path2)

        if score > 0.3:
          prediction = True
        else:
          prediction = False
        return  score, prediction