import os

from kaznlp.lid.lidnb import LidNB
from kaznlp.tokenization.tokrex import TokenizeRex

from src.models.base import Model, LangDetectorChoices, Language

tokrex = TokenizeRex()
landetector = LidNB(char_mdl=os.path.join('kaznlp', 'lid', 'char.mdl'))

class KazNLP(Model):
    def detect_lang_single(self, txt: str) -> Language:
        txt_lang = landetector.predict(tokrex.tokenize(txt, lower=True)[0])
        return txt_lang

    def detect_lang_probabilities(self, txt: str) -> LangDetectorChoices:
        txt_lang = landetector.predict_wp(tokrex.tokenize(txt, lower=True)[0])
        response = {lang: prob for lang, prob in txt_lang.items()}
        response["primary_lang"] = response['result']
        del response['result']
        return response
        
