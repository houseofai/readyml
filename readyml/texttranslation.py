import torch

class TranslationModel():
    """
    URL: https://github.com/pytorch/fairseq/blob/master/examples/translation/README.md
    """
    def __init__(self, model_name, bpe):
        self.model = torch.hub.load('pytorch/fairseq', \
                    f'transformer.{model_name}', \
                    tokenizer='moses', bpe=bpe)

        #self.model.cuda()

    def infer(self, text, beam=5):
        return self.model.translate(text, beam=beam)

class Translation_EnglishToFrench(TranslationModel):
    def __init__(self):
        super().__init__("wmt14.en-fr", bpe='subword_nmt')

class Translation_EnglishToGerman(TranslationModel):
    def __init__(self):
        super().__init__("wmt19.en-de", bpe='fastbpe')

class Translation_GermanToEnglish(TranslationModel):
    def __init__(self):
        super().__init__("wmt19.de-en", bpe='fastbpe')

class Translation_EnglishToRussian(TranslationModel):
    def __init__(self):
        super().__init__("wmt19.en-ru", bpe='fastbpe')

class Translation_RussianToEnglish(TranslationModel):
    def __init__(self):
        super().__init__("wmt19.ru-en", bpe='fastbpe')
