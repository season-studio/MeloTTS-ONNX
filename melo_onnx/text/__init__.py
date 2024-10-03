import copy

from .symbols import *

class TextUtils():
    def __init__(self, lang:str, model_path:str, onnx_providers:list, session_opts = None, onnx_params:dict = None):
        self.__language = lang
        match lang:
            case 'ZH_MIX_EN':
                from .chinese_mix_en_kernel import ChineseMixEnKernel
                self.__kernel = ChineseMixEnKernel(model_path, onnx_providers, session_opts, onnx_params)
            case _:
                raise NotImplementedError(f"Kernel of language \"{lang}\" is not implemented")
            
        self.get_bert = self.__kernel.get_bert_feature

    @property
    def language(self):
        return self.__language
    
    @property
    def kernel(self):
        return self.__kernel
    
    Default_Symbol_2_ID = None

    @staticmethod
    @property
    def DefaultSymbol2ID():
        if TextUtils.Default_Symbol_2_ID is None:
            TextUtils.Default_Symbol_2_ID = {s: i for i, s in enumerate(symbols)}
        return TextUtils.Default_Symbol_2_ID
    
    def clean_text(self, text):
        kernel = self.__kernel
        norm_text = kernel.text_normalize(text)
        phones, tones, word2ph = kernel.g2p(norm_text)
        return norm_text, phones, tones, word2ph
    
    def clean_text_bert(self, text):
        kernel = self.__kernel
        norm_text = kernel.text_normalize(text)
        phones, tones, word2ph = kernel.g2p(norm_text)
        
        word2ph_bak = copy.deepcopy(word2ph)
        for i in range(len(word2ph)):
            word2ph[i] = word2ph[i] * 2
        word2ph[0] += 1
        bert = kernel.get_bert_feature(norm_text, word2ph)
        
        return norm_text, phones, tones, word2ph_bak, bert

    def cleaned_text_to_sequence(self, cleaned_text, tones, language, symbol_to_id=None):
        """Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
        Args:
        text: string to convert to a sequence
        Returns:
        List of integers corresponding to the symbols in the text
        """
        symbol_to_id_map = symbol_to_id or self.Default_Symbol_2_ID
        phones = [symbol_to_id_map[symbol] for symbol in cleaned_text]
        tone_start = language_tone_start_map[language]
        tones = [i + tone_start for i in tones]
        lang_id = language_id_map[language]
        lang_ids = [lang_id for i in phones]
        return phones, tones, lang_ids
    