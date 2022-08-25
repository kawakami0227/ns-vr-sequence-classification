from sequence_classification import SequenceClassification

sequence_classification = SequenceClassification()

output = sequence_classification.classification(text='体調はどうですか')

print(output)
print(output[0]['label'])
print(output[0]['score'])