import onnxruntime as ort
import os
import re
import numpy as np

from .text.split_utils import split_sentence
from .hparams import HParams
from .text import TextUtils
from . import commons

class MeloTTS_ONNX():

    PreferredProviders = ['CUDAExecutionProvider', 'CPUExecutionProvider']

    def __init__(self, model_path, execution_provider:str = None, verbose:bool = False, onnx_session_options:ort.SessionOptions = None, onnx_params:dict = None):
        self.__verbose = verbose

        if verbose:
            print("Loading the configuration...")
        config_path = os.path.join(model_path, "configuration.json")
        self.__hparams = HParams.load_from_file(config_path)
        symbols = self.__hparams.symbols
        self.__symbol_to_id = {s: i for i, s in enumerate(symbols)}

        execution_provider = f"{execution_provider}ExecutionProvider" if (execution_provider is not None) and (not execution_provider.endswith("ExecutionProvider")) else execution_provider
        available_providers = ort.get_available_providers()
        self.__execution_providers = [execution_provider if execution_provider in available_providers else next((provider for provider in MeloTTS_ONNX.PreferredProviders if provider in available_providers), 'CPUExecutionProvider')]
        if verbose:
            print(f"Use \"{self.__execution_providers}\" as onnx execution provider")

        if verbose:
            print("Creating onnx session for tts...")
        self.__tts_infer = self.create_tts_infer(os.path.join(model_path, "tts_model.onnx"), self.__execution_providers, onnx_session_options, onnx_params)

        if verbose:
            print("Loading text utils...")
        self.__text_utils = TextUtils(self.__lang_from_hparams(self.__hparams), model_path, self.__execution_providers, self.__adjust_onnx_session_options(onnx_session_options), onnx_params)


    def speak(self, text:str, speaker:str, sdp_ratio=0.2, noise_scale=0.6, noise_scale_w=0.8, speed=1.0, pbar=None):
        speaker_id = self.__hparams.data.spk2id[speaker]
        language = self.language

        texts = self.__split_sentences_into_pieces(text, language, self.__verbose)
        if pbar:
            texts = pbar(texts)

        audio_list = []
        for sentence in texts:
            if language in ['EN', 'ZH_MIX_EN']:
                sentence = re.sub(r'([a-z])([A-Z])', r'\1 \2', sentence)
            
            bert, ja_bert, phones, tones, lang_ids = self.__get_text_for_tts_infer(sentence)
            
            x_tst = np.expand_dims(phones, axis=0)
            tones = np.expand_dims(tones, axis=0)
            lang_ids = np.expand_dims(lang_ids, axis=0)
            bert = np.expand_dims(bert, axis=0)
            ja_bert = np.expand_dims(ja_bert, axis=0)
            x_tst_lengths = np.array([phones.shape[0]], dtype=np.int64)
            del phones
            speakers = np.array([speaker_id], dtype=np.int64)

            if self.__verbose:
                print('x_tst', x_tst.shape, x_tst)
                print('x_tst_lengths', x_tst_lengths.shape, x_tst_lengths)
                print('tones', tones.shape, tones)
                print('lang_ids', lang_ids.shape, lang_ids)
                print('bert', bert.shape, bert)
                print('ja_bert', ja_bert.shape, ja_bert)
            
            audio = self.__tts_infer(
                x = x_tst,
                x_lengths = x_tst_lengths,
                sid = speakers,
                tone = tones,
                language = lang_ids,
                bert = bert.astype(np.float32),
                ja_bert = ja_bert.astype(np.float32),
                noise_scale = np.array(noise_scale, dtype=np.float32),
                length_scale = np.array(1. / speed, dtype=np.float32),
                noise_scale_w = np.array(noise_scale_w, dtype=np.float32),
                sdp_ratio = np.array(sdp_ratio, dtype=np.float32))
            del x_tst, tones, lang_ids, bert, ja_bert, x_tst_lengths, speakers
            audio_list.append(audio)

        return self.__concat_audio(audio_list, sample_rate=self.sample_rate, speed=speed)

    @property
    def language(self):
        return self.__lang_from_hparams(self.__hparams)
    
    @property
    def speakers(self):
        speaker_ids = self.__hparams.data.spk2id
        return list(speaker_ids.keys())
    
    @property
    def sample_rate(self):
        return getattr(self.__hparams.data, 'sampling_rate', 44100)
    
    @staticmethod
    def __lang_from_hparams(hparams: HParams):
        return getattr(hparams, 'language', None)
    
    @staticmethod
    def __split_sentences_into_pieces(text, language, verbose):
        texts = split_sentence(text, language_str=language)
        if verbose:
            print(" > Text split to sentences:")
            for i in texts:
                print("   ", i)
        return texts
    
    @staticmethod
    def __concat_audio(segment_data_list, sample_rate, speed=1.):
        audio_segments = []
        for segment_data in segment_data_list:
            audio_segments += segment_data.reshape(-1).tolist()
            audio_segments += [0] * int((sample_rate * 0.05) / speed)
        audio_segments = np.array(audio_segments).astype(np.float32)
        return audio_segments
    
    @staticmethod
    def get_def_onnx_session_options():
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        return session_options
    
    @staticmethod
    def __adjust_onnx_session_options(session_options:ort.SessionOptions = None):
        return session_options or MeloTTS_ONNX.get_def_onnx_session_options()

    @staticmethod
    def create_tts_infer(onnx_model_path:str, providers:list, session_options:ort.SessionOptions = None, onnx_params:dict = None):
        if not os.path.exists(onnx_model_path):
            raise FileNotFoundError(f"Can not found the onnx model file \"{onnx_model_path}\"")
        tts_session = ort.InferenceSession(onnx_model_path, sess_options=MeloTTS_ONNX.__adjust_onnx_session_options(session_options), providers=providers, **(onnx_params or {}))
        def infer(**kwargs):
            return tts_session.run(None, kwargs)[0][0, 0]
        infer._session = tts_session
        return infer
    
    def __get_text_for_tts_infer(self, text):
        language_str = self.language
        utils = self.__text_utils
        hps = self.__hparams
        norm_text, phone, tone, word2ph = utils.clean_text(text)
        
        phone, tone, language = utils.cleaned_text_to_sequence(phone, tone, language_str, self.__symbol_to_id)

        if hps.data.add_blank:
            phone = commons.intersperse(phone, 0)
            tone = commons.intersperse(tone, 0)
            language = commons.intersperse(language, 0)
            for i in range(len(word2ph)):
                word2ph[i] = word2ph[i] * 2
            word2ph[0] += 1

        if getattr(hps.data, "disable_bert", False):
            bert = np.zeros((1024, len(phone)))
            ja_bert = np.zeros((768, len(phone)))
        else:
            bert = utils.get_bert(norm_text, word2ph)
            del word2ph
            assert bert.shape[-1] == len(phone), phone

            if language_str == "ZH":
                ja_bert = np.zeros((768, len(phone)))
            elif language_str in ["JP", "EN", "ZH_MIX_EN", 'KR', 'SP', 'ES', 'FR', 'DE', 'RU']:
                ja_bert = bert
                bert = np.zeros((1024, len(phone)))
            else:
                raise NotImplementedError()

        assert bert.shape[-1] == len(
            phone
        ), f"Bert seq len {bert.shape[-1]} != {len(phone)}"

        phone = np.array(phone, dtype=np.int64)
        tone = np.array(tone, dtype=np.int64)
        language = np.array(language, dtype=np.int64)
        return bert, ja_bert, phone, tone, language

if __name__ == "__main__":
    pass