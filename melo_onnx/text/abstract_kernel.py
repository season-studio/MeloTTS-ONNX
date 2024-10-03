import os

from tokenizers import Tokenizer
from .bert_infer import create_bert_infer

class AbstractKernel():
    def __init__(self, model_path:str, onnx_providers:list, session_opts = None, onnx_params:dict = None):
        tokenizer = Tokenizer.from_file(os.path.join(model_path, "tokenizer.json"))
        bert_infer = create_bert_infer(model_path, onnx_providers, session_opts, tokenizer, onnx_params)
        self._tokenizer = tokenizer
        self.get_bert_feature = bert_infer

    def g2p(self, text, **kwargs):
        raise NotImplementedError()
    
    def text_normalize(self, text):
        raise NotImplementedError()