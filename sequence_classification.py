from transformers import pipeline
from transformers import BertForSequenceClassification
from transformers import BertJapaneseTokenizer


class SequenceClassification:
    def __init__(self):
        model_name = 'cl-tohoku/bert-base-japanese-whole-word-masking'
        tokenizer = BertJapaneseTokenizer.from_pretrained(model_name)
        model = BertForSequenceClassification.from_pretrained('content/model_transformers')

        self.sentence_classification = pipeline("text-classification", model=model, tokenizer=tokenizer)
        self.reply = [
            'それならデイルームに行ってこようかな',
            'なんだね。君は失礼ではないか',
            'いいよ',
            'はい、そうです',
            'きたなくしてしまったからなあー',
            'いいですよ',
            'わかりました',
            'いいよ',
            '大丈夫です',
            'ぐっすり眠れました',
            '大丈夫です',
            'いいよ',
            'いいよ',
            'いいよ',
            'それならデイルームに行ってこようかな',
            '一昨日変えたばかりだから、このまま使う',
            'いいよ',
            'そうだね',
            'よく見えないから適当に捨てちゃう',
            'そうだね。これでいいよ',
            'そのままにしておいて',
            'いいよ',
            'そうだね',
            '大丈夫だよ',
            'はい',
            'おはよう',
            'こんにちは',
            'こんばんは',
            'いいよ',
            '大丈夫だよ'
            ]

    def classification(self, text):
        output = self.sentence_classification(text)
        reply = self.reply[int(output[0]['label'][6:])]
        print(reply)
        return output, reply
