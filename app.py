from flask import Flask, jsonify

from sequence_classification import SequenceClassification

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

sequence_cl = SequenceClassification()

@app.route('/')
def index():
    return 'index'

@app.route('/<path:text>')
def classfication(text):
    output, reply = sequence_cl.classification(text=text)
    return jsonify({
        'label': output[0]['label'],
        'score': output[0]['score'],
        'reply': reply
    })


if __name__ == "__main__":
    app.run(host='127.0.0.1', port=8080, debug=True)
