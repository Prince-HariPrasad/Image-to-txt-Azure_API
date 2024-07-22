from flask import Flask, request, render_template, redirect, url_for, send_from_directory
import os
import json
import time
from PIL import Image, ImageDraw
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from msrest.authentication import CognitiveServicesCredentials

app = Flask(__name__)

# Load credentials
with open('Credentials.json') as f:
    credentials = json.load(f)
API_KEY = credentials['API_KEY']
ENDPOINT = credentials['ENDPOINT']

cv_client = ComputerVisionClient(ENDPOINT, CognitiveServicesCredentials(API_KEY))

# Define paths
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            
            # Process the image with Azure Computer Vision
            with open(file_path, 'rb') as image_stream:
                response = cv_client.read_in_stream(image_stream, language='en', raw=True)
                operation_location = response.headers['Operation-Location']
                operation_id = operation_location.split('/')[-1]

                # Wait for the operation to complete
                while True:
                    result = cv_client.get_read_result(operation_id)
                    if result.status not in ['running', 'notStarted']:
                        break
                    time.sleep(1)

                if result.status == 'succeeded':
                    read_results = result.analyze_result.read_results
                    text = ""
                    image = Image.open(file_path)
                    draw = ImageDraw.Draw(image)
                    for analyzed_result in read_results:
                        for line in analyzed_result.lines:
                            text += line.text + "\n"
                            
                            # Draw bounding boxes on the image
                            points = [(line.bounding_box[i], line.bounding_box[i+1]) for i in range(0, len(line.bounding_box), 2)]
                            draw.line(points + [points[0]], fill=(255, 0, 0), width=5)

                    result_image_path = os.path.join(app.config['RESULT_FOLDER'], file.filename)
                    image.save(result_image_path)
                    
                    return render_template('result.html', text=text, result_image=file.filename)
                else:
                    return "Error: " + result.status
    
    return render_template('index.html')

@app.route('/results/<filename>')
def send_image(filename):
    return send_from_directory(app.config['RESULT_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True, port=5500)
