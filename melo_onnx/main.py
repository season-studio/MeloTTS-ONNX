import onnxruntime as ort
import os
import re
import numpy as np
from typing import Callable

from .text.split_utils import split_sentence
from .hparams import HParams
from .text import TextUtils
from . import commons

class BaseClassForOnnxInfer():
    @staticmethod
    def create_onnx_infer(infer_factor:Callable, onnx_model_path:str, providers:list, session_options:ort.SessionOptions = None, onnx_params:dict = None):
        if not os.path.exists(onnx_model_path):
            raise FileNotFoundError(f"Can not found the onnx model file \"{onnx_model_path}\"")
        session = ort.InferenceSession(onnx_model_path, sess_options=BaseClassForOnnxInfer.adjust_onnx_session_options(session_options), providers=providers, **(onnx_params or {}))
        fn = infer_factor(session)
        fn.__session = session
        return fn

    @staticmethod
    def get_def_onnx_session_options():
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        return session_options
    
    @staticmethod
    def adjust_onnx_session_options(session_options:ort.SessionOptions = None):
        return session_options or BaseClassForOnnxInfer.get_def_onnx_session_options()

class MeloTTS_ONNX(BaseClassForOnnxInfer):

    PreferredProviders = ['CUDAExecutionProvider', 'CPUExecutionProvider']

    def __init__(self, model_path, execution_provider:str = None, verbose:bool = False, onnx_session_options:ort.SessionOptions = None, onnx_params:dict = None):
        '''
        Create the instance of TTS

        Args:
            model_path (str): The path of the folder which contains the model
            execution_provider (str): The provider that onnxruntime used. Such as CPUExecutionProvider, CUDAExecutionProvider, etc. Or you can use CPU, CUDA as short one. If it is None, the constructor will choose a best one automaticlly
            verbose (bool): Set True to show more detail informations when working
            onnx_session_options (onnxruntime.SessionOptions): The custom options for onnx session
            onnx_params (dict): Other parameters you want to pass to the onnxruntime.InferenceSession constructor

        Returns:
            MeloTTS_ONNX: The instance of the TTS
        '''
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
        def tts_infer_factor(session):
            return lambda **kwargs: session.run(None, kwargs)[0][0, 0]
        self.__tts_infer = self.create_onnx_infer(tts_infer_factor, os.path.join(model_path, "tts_model.onnx"), self.__execution_providers, onnx_session_options, onnx_params)

        if verbose:
            print("Loading text utils...")
        self.__text_utils = TextUtils(self.__lang_from_hparams(self.__hparams), model_path, self.__execution_providers, self.adjust_onnx_session_options(onnx_session_options), onnx_params)


    def speak(self, text:str, speaker:str, sdp_ratio=0.2, noise_scale=0.6, noise_scale_w=0.8, speed=1.0, pbar=None):
        '''
        Do speech synthesis

        Args:
            text (str): The text you want to synthesis
            speaker (str): The voice you want to use. It should be one of the element in the speakers property
            sdp_ratio (float):
            noise_scale (float):
            noise_scale_w (float):
            speed (float): The speed of the dest audio
            pbar (Callable): A function for showing the progress bar. Such as tqdm

        Returns:
            numpy.array: The audio data
        '''
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
        '''
        The language of the model the instance had loaded
        '''
        return self.__lang_from_hparams(self.__hparams)
    
    @property
    def speakers(self):
        '''
        The list of the speakers that are available in the model
        '''
        speaker_ids = self.__hparams.data.spk2id
        return list(speaker_ids.keys())
    
    @property
    def sample_rate(self):
        '''
        The sample rate of the dest audio of the result
        '''
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

class OpenVoiceToneClone_ONNX(BaseClassForOnnxInfer):

    PreferredProviders = ['CUDAExecutionProvider', 'CPUExecutionProvider']

    def __init__(self, model_path, execution_provider:str = None, verbose:bool = False, onnx_session_options:ort.SessionOptions = None, onnx_params:dict = None):
        '''
        Create the instance of the tone cloner

        Args:
            model_path (str): The path of the folder which contains the model
            execution_provider (str): The provider that onnxruntime used. Such as CPUExecutionProvider, CUDAExecutionProvider, etc. Or you can use CPU, CUDA as short one. If it is None, the constructor will choose a best one automaticlly
            verbose (bool): Set True to show more detail informations when working
            onnx_session_options (onnxruntime.SessionOptions): The custom options for onnx session
            onnx_params (dict): Other parameters you want to pass to the onnxruntime.InferenceSession constructor

        Returns:
            OpenVoiceToneClone_ONNX: The instance of the tone cloner
        '''
        self.__verbose = verbose

        if verbose:
            print("Loading the configuration...")
        config_path = os.path.join(model_path, "configuration.json")
        self.__hparams = HParams.load_from_file(config_path)

        execution_provider = f"{execution_provider}ExecutionProvider" if (execution_provider is not None) and (not execution_provider.endswith("ExecutionProvider")) else execution_provider
        available_providers = ort.get_available_providers()
        self.__execution_providers = [execution_provider if execution_provider in available_providers else next((provider for provider in MeloTTS_ONNX.PreferredProviders if provider in available_providers), 'CPUExecutionProvider')]
        
        if verbose:
            print("Creating onnx session for tone color extractor...")
        def se_infer_factor(session):
            return lambda **kwargs: session.run(None, kwargs)[0]
        self.__se_infer = self.create_onnx_infer(se_infer_factor, os.path.join(model_path, "tone_color_extract_model.onnx"), self.__execution_providers, onnx_session_options, onnx_params)

        if verbose:
            print("Creating onnx session for tone clone ...")
        def tc_infer_factor(session):
            return lambda **kwargs: session.run(None, kwargs)[0][0, 0]
        self.__tc_infer = self.create_onnx_infer(tc_infer_factor, os.path.join(model_path, "tone_clone_model.onnx"), self.__execution_providers, onnx_session_options, onnx_params)

    hann_window = {}

    def __spectrogram_numpy(self, y, n_fft, sampling_rate, hop_size, win_size, onesided=True):
        if self.__verbose:
            if np.min(y) < -1.1:
                print("min value is ", np.min(y))
            if np.max(y) > 1.1:
                print("max value is ", np.max(y))

        # 填充
        y = np.pad(
            y,
            int((n_fft - hop_size) / 2),
            mode="reflect",
        )

        # 生成汉宁窗
        win_key = f"{str(y.dtype)}-{win_size}"
        if True or win_key not in hann_window:
            OpenVoiceToneClone_ONNX.hann_window[win_key] = np.hanning(win_size + 1)[:-1].astype(y.dtype)
        window = OpenVoiceToneClone_ONNX.hann_window[win_key]
        
        # 短时傅里叶变换
        y_len = y.shape[0]
        win_len = window.shape[0]
        count = int((y_len - win_len) // hop_size) + 1
        spec = np.empty((count, int(win_len / 2) + 1 if onesided else (int(win_len / 2) + 1) * 2, 2))
        start = 0
        end = start + win_len
        idx = 0
        while end <= y_len:
            segment = y[start:end]
            frame = segment * window
            step_result = np.fft.rfft(frame) if onesided else np.fft.fft(frame)
            spec[idx] = np.column_stack((step_result.real, step_result.imag))
            start = start + hop_size
            end = start + win_len
            idx += 1

        # 合并实部虚部
        spec = np.sqrt(np.sum(np.square(spec), axis=-1) + 1e-6)

        return np.array([spec], dtype=np.float32)
    
    def extract_tone_color(self, audio:np.array):
        '''
        Extract the tone color from an audio

        Args:
            audio (numpy.array): The data of the audio

        Returns:
            numpy.array: The tone color vector
        '''
        hps = self.__hparams
        y = self.to_mono(audio.astype(np.float32))
        y = self.__spectrogram_numpy(y, hps.data.filter_length,
                                    hps.data.sampling_rate, hps.data.hop_length, hps.data.win_length,
                                    )
        return self.__se_infer(input=y).reshape(1,256,1)
    
    def mix_tone_color(self, colors:list):
        '''
        Mix multi tone colors to a single one

        Args:
            color (list[numpy.array]): The list of the tone colors you want to mix. Each element should be the result of extract_tone_color.

        Returns:
            numpy.array: The tone color vector
        '''
        return np.stack(colors).mean(axis=0)
    
    def tone_clone(self, audio:np.array, target_tone_color:np.array, tau=0.3):
        '''
        Clone the tone

        Args:
            audio (numpy.array): The data of the audio that will be changed the tone
            target_tone_color (numpy.array): The tone color that you want to clone. It should be the result of the extract_tone_color or mix_tone_color.
            tau (float):

        Returns:
            numpy.array: The dest audio
        '''
        assert (target_tone_color.shape == (1,256,1)), "The target tone color must be an array with shape (1,256,1)"
        hps = self.__hparams
        src = self.to_mono(audio.astype(np.float32))
        src = self.__spectrogram_numpy(src, hps.data.filter_length,
                                      hps.data.sampling_rate, hps.data.hop_length, hps.data.win_length,
                                      )
        src_tone = self.__se_infer(input=src).reshape(1,256,1)
        src = np.transpose(src, (0, 2, 1))
        src_length = np.array([src.shape[-1]], dtype=np.int64)
        return self.__tc_infer(audio=src, audio_length=src_length, src_tone=src_tone, dest_tone=target_tone_color, tau=np.array([tau], dtype=np.float32))
    
    def to_mono(self, audio:np.array):
        '''
        Change the audio to be a mono audio

        Args:
            audio (numpy.array): The source audio

        Returns:
            numpy.array: The mono audio data
        '''
        return np.mean(audio, axis=1) if len(audio.shape) > 1 else audio

    def resample(self, audio:np.array, original_rate:int):
        '''
        Resample the audio to match the model. It is used for changing the sample rate of the audio.

        Args:
            audio (numpy.array): The source audio you want to resample.
            original_rate (int): The original sample rate of the source audio

        Returns:
            numpy.array: The dest data of the audio after resample
        '''
        audio = self.to_mono(audio)
        target_rate = self.__hparams.data.sampling_rate
        duration = audio.shape[0] / original_rate
        target_length = int(duration * target_rate)
        time_original = np.linspace(0, duration, num=audio.shape[0])
        time_target = np.linspace(0, duration, num=target_length)
        resampled_data = np.interp(time_target, time_original, audio)
        return resampled_data
    
    @property
    def sample_rate(self):
        '''
        The sample rate of the tone cloning result 
        '''
        return self.__hparams.data.sampling_rate

if __name__ == "__main__":
    pass