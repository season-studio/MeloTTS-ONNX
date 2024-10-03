import onnxruntime as ort
import os
from tokenizers import Tokenizer
import numpy as np

def create_bert_infer(model_path:str, onnx_providers:list, session_opts:ort.SessionOptions, tokenizer:Tokenizer, onnx_params:dict = None):
    bert_session = ort.InferenceSession(os.path.join(model_path, "bert_lml_model.onnx"), sess_options=session_opts, providers=onnx_providers, **(onnx_params or {}))
    def get_bert_feature(text, word2ph):
        inputs = tokenizer.encode(text, add_special_tokens=True)
        res = bert_session.run(None, {
            'input_ids': np.array([inputs.ids]),
            'token_type_ids': np.array([inputs.type_ids]),
            'attention_mask': np.array([inputs.attention_mask])
        })[0][0]
        phone_level_feature = []
        for i in range(len(word2ph)):
            repeat_feature = np.repeat(np.array([res[i]]), word2ph[i], axis=0)
            phone_level_feature.append(repeat_feature)

        phone_level_feature = np.concatenate(phone_level_feature, axis=0)
        return phone_level_feature.T
    get_bert_feature._session = bert_session
    return get_bert_feature
