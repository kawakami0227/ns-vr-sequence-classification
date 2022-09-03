from flask import Flask, jsonify

from sequence_classification import SequenceClassification

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False


@app.route('/<path:text>')
def classfication(text):
    sequence_cl = SequenceClassification()
    output, reply = sequence_cl.classification(text=text)
    return jsonify({
        'label': output[0]['label'],
        'score': output[0]['score'],
        'reply': reply
    })


if __name__ == "__main__":
    app.run(host='127.0.0.1', port=8888, debug=True)
