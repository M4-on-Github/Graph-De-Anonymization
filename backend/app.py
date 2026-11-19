from flask import Flask, jsonify
from flask_cors import CORS
import os
import json

app = Flask(__name__)
CORS(app)

DATASET_FOLDER = "cyto-datasets"

@app.route('/cyto-datasets')
def get_datasets():
    files = [
        f for f in os.listdir(DATASET_FOLDER)
        if f.endswith('.json')
    ]
    return jsonify(files)


@app.route('/cyto-datasets/<filename>')
def get_dataset(filename):
    if not filename.endswith('.json') or os.path.basename(filename) != filename:
        return jsonify({"error": "Invalid filename"}), 400

    filepath = os.path.join(DATASET_FOLDER, filename)

    if not os.path.isfile(filepath):
        return jsonify({"error": "Dataset not found"}), 404

    with open(filepath, 'r') as file:
        data = json.load(file)

    return jsonify(data)


if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)