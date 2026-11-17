
from flask import Flask, jsonify
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

# Location of the datasets folder
DATASET_FOLDER = os.path.join(
    os.path.dirname(__file__), "datasets"
)

@app.route('/datasets', methods=['GET'])
def get_datasets():

    # Get all files in the datasets folder
    files = [
        file for file in os.listdir(DATASET_FOLDER)
        if os.path.isfile(os.path.join(DATASET_FOLDER, file))
    ]

    # Return filenames as JSON
    return jsonify(files)


if __name__ == '__main__':
    app.run(debug=True, port=5000)
