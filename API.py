

from flask import Flask, request, jsonify, send_from_directory
import numpy as np
import torch
import os
from Model import *
from flask_cors import CORS  



app = Flask(__name__)
CORS(app)  # 

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')  

@app.route('/convert_to_fen', methods=['POST'])
def convert_to_fen():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    file = request.files['image']
    
    
    temp_image_path = os.path.join('temp_image.jpg')  
    file.save(temp_image_path)

    
    crops = get_image_crops(temp_image_path)  
    dictionary = get_preds_with_constraints(crops)  
    fen_string = dictionary_to_fen(dictionary)


    print("Generated FEN String:", fen_string)

    
    if os.path.exists(temp_image_path):
        os.remove(temp_image_path)

    return jsonify({'fen': fen_string})


if __name__ == '__main__':
    app.run(port=5000) 