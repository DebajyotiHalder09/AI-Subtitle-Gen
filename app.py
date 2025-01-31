# app.py
from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS
import whisper
import os
import json
import tempfile
import uuid

app = Flask(__name__, template_folder='templates')
CORS(app)  # Enable CORS for all routes

# Initialize Whisper model globally to avoid reloading
model = whisper.load_model("tiny")

# Create temporary directories for uploads and results
UPLOAD_FOLDER = tempfile.mkdtemp()
RESULTS_FOLDER = tempfile.mkdtemp()
ALLOWED_EXTENSIONS = {'mp3', 'wav', 'ogg', 'flac', 'm4a'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def format_time(seconds):
    """Convert seconds to timestamp format"""
    return seconds

def create_subtitle_json(segments):
    """Convert whisper segments to our subtitle format"""
    subtitles = []
    for segment in segments:
        subtitle = {
            "start": format_time(segment['start']),
            "end": format_time(segment['end']),
            "text": segment['text'].strip()
        }
        subtitles.append(subtitle)
    return subtitles

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/transcribe', methods=['POST'])
def transcribe_audio():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    
    file = request.files['audio']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed'}), 400

    try:
        # Create unique filename
        filename = str(uuid.uuid4()) + file.filename
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        # Transcribe with Whisper
        result = model.transcribe(filepath)
        
        # Convert to our subtitle format
        subtitles = create_subtitle_json(result['segments'])
        
        # Save JSON result
        result_filename = f"{filename}_subtitles.json"
        result_path = os.path.join(RESULTS_FOLDER, result_filename)
        
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(subtitles, f, ensure_ascii=False, indent=2)
        
        # Clean up audio file
        os.remove(filepath)
        
        return jsonify({
            'status': 'success',
            'result_id': result_filename
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/download/<result_id>', methods=['GET'])
def download_result(result_id):
    try:
        file_path = os.path.join(RESULTS_FOLDER, result_id)
        return send_file(
            file_path,
            as_attachment=True,
            download_name='subtitles.json',
            mimetype='application/json'
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 404

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
