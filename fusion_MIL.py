
import os
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

    def __call__(self, data):
        X = self.builder(data)
        y = self.model.predict(X)

        return y








