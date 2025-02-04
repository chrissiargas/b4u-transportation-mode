
import os

import numpy as np
import pandas as pd

from config_parser import Parser
from preprocessing import builder
from architectures import get_attMIL
from tensorflow.python.client import device_lib
import time

def get_available_devices():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU' or x.device_type == 'CPU']


class Fusion_MIL:
    def __init__(self):
        print(get_available_devices())

        self.conf = Parser()
        self.conf.get_args()

        self.builder = builder()

        self.model_name = 'fusion_MIL'
        self.model_dir = os.path.join('models', '%s.h5' % self.model_name)

        self.model = get_attMIL(self.builder.input_shape)

        self.model.compile()
        self.model.summary()
        self.model.load_weights(self.model_dir)

        self.one_batch = True

    def get_output(self, Y, t):
        Y_ = np.argmax(Y, axis=1)
        modes = list(map(lambda y_: self.builder.modes[y_], Y_))
        output = pd.DataFrame(columns=['subject', 'timestamp', 'mode'])
        output['subject'] = t[:, 0]
        output['timestamp'] = t[:, 2]
        output['mode'] = modes

        return output

    def __call__(self, data):
        X, t = self.builder(data)
        Y = self.model.predict(X, verbose=0)
        output = self.get_output(Y, t)

        return output








