import keras
import numpy as np
from flask import Flask, jsonify, request

app = Flask(__name__)
model = keras.models.load_model("/models/learning-path.h5")
model._make_predict_function()


@app.route('/')
def hello_world():
  return 'EOJ learning path prediction'


@app.route('/predict', methods=["POST"])
def predict():
  INPUT_DIM = 3800
  solved = request.json.get("solved", [])
  if not solved:
    return jsonify([])
  else:
    input = np.zeros((1, INPUT_DIM))
    for x in solved:
      if x > 0 and x < INPUT_DIM:
        input[0][x] = 1
    predict = model.predict(input)[0]
    solved_set = set(solved)
    ret = [int(x) for x in (-predict).argsort() if x not in solved_set]
    return jsonify(ret[:100])


if __name__ == '__main__':
  app.run()
