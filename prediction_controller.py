from flask import Flask
from flask import request
from flask import jsonify
import base64
import sys

from tf_predict_service import PredictService

app = Flask(__name__)
service = PredictService()


@app.route("/api/predict", methods=['POST'])
def predict():
    data = request.get_json()
    img = base64.b64decode(data['base64Image'])
    labels, pred = service.predict([img])
    response = dict()
    response['identifier'] = data['identifier']

    label_list = []
    for label in labels:
        label_list.append(label.decode('UTF-8'))
    score_list = []
    for score in pred[0]:
        score_list.append(float(score))

    num_results = data.get('prediction_results', len(label_list))
    sorted_scores, sorted_labels = zip(*sorted(zip(score_list, label_list), reverse=True))
    response['scores'] = sorted_scores[:num_results]
    response['labels'] = sorted_labels[:num_results]

    return jsonify(response)


@app.route("/api/labels", methods=['GET'])
def get_labels():
    response = dict()
    label_list = []
    for label in service.label:
        label_list.append(label.decode('UTF-8'))
    response['labels'] = label_list
    return jsonify(response)


if __name__ == "__main__":
    port = 5002
    if len(sys.argv) > 1:
        port = sys.argv[1]
    app.run(host='0.0.0.0', port=port, debug=True, use_reloader=False)
