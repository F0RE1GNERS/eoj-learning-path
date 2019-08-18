from flask import Flask, jsonify, request

from inference import inference

app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'EOJ learning path prediction'


@app.route('/predict', methods=["POST"])
def predict():
    solved = request.json.get("solved", [])
    if not solved:
        result = []
    else:
        result = inference(solved)
    return jsonify({
        "prediction": result
    })


if __name__ == '__main__':
    app.run()
