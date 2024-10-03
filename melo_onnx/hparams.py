import json
import os

class HParams:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if type(v) == dict:
                v = HParams(**v)
            self[k] = v

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()

    def __len__(self):
        return len(self.__dict__)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def __contains__(self, key):
        return key in self.__dict__

    def __repr__(self):
        return self.__dict__.__repr__()
    
    @staticmethod
    def load_from_file(file_path:str):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Can not found the configuration file \"{file_path}\"")
        with open(file_path, "r", encoding="utf-8") as f:
            hps = json.load(f)
            return HParams(**hps)