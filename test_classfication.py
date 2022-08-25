from transformers import pipeline
from transformers import BertForSequenceClassification
from transformers import BertJapaneseTokenizer


MODEL_NAME = 'cl-tohoku/bert-base-japanese-whole-word-masking'
tokenizer = BertJapaneseTokenizer.from_pretrained(MODEL_NAME)
model = BertForSequenceClassification.from_pretrained('content/model_transformers')

sentence_classification = pipeline("text-classification", model=model, tokenizer=tokenizer)

output = sentence_classification('川上さん、こんにちは')
print(output)