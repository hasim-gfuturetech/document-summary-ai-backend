from flask import Flask, request, jsonify
import os
from flask_cors import CORS
from summarizing import processing
app = Flask(__name__)
CORS(app)


@app.route('/', methods=['POST'])
def process():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'File not found'})

        file = request.files['file']

        if file.filename == '':
            return jsonify({'error': 'No selected file'})

        if file:
            result = processing(file)
            print('result', result)
            return jsonify({'message': 'File uploaded successfully', "data": result})
        else:
            return jsonify({'error': 'File upload failed'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)