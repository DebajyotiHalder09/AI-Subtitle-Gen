import whisper
import json
import os
import logging
import tempfile
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import tkinter as tk
from tkinter import filedialog
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={
    r"/*": {
        "origins": "*",
        "allow_headers": ["Content-Type", "Authorization"],
        "supports_credentials": True
    }
})

# Global variable to store subtitles temporarily
current_subtitles = None

# Load the Whisper model 
try:
    model = whisper.load_model("base")
    logger.info("Whisper model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load Whisper model: {e}")
    sys.exit(1)

@app.route('/')
def index():
    """
    Root route to render the index page (HTML).
    """
    return render_template('main.html')  # Replace with the actual name of your HTML file

def show_save_dialog(subtitles):
    """
    Open a file save dialog to select subtitle save location.
    
    Args:
        subtitles (list): Subtitle data to be saved
    
    Returns:
        str: Selected file path or None
    """
    try:
        root = tk.Tk()
        root.withdraw()
        root.lift()
        root.focus_force()
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json")],
            title="Save Subtitles File"
        )
        
        root.destroy()
        return file_path
    except Exception as e:
        logger.error(f"Error in save dialog: {e}")
        return None

@app.route('/transcribe', methods=['POST'])
def transcribe():
    """
    Transcribe uploaded audio file using Whisper.
    
    Returns:
        JSON response with transcription or error details
    """
    global current_subtitles
    
    if 'audio' not in request.files:
        logger.warning("No audio file uploaded")
        return jsonify({"error": "No audio file uploaded."}), 400
    
    audio_file = request.files['audio']
    
    allowed_extensions = ('.mp3', '.wav', '.m4a', '.webm', '.ogg')
    if not audio_file.filename.lower().endswith(allowed_extensions):
        logger.warning(f"Invalid file type: {audio_file.filename}")
        return jsonify({"error": "Invalid file type. Only audio files are allowed."}), 400

    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(audio_file.filename)[1]) as temp_audio:
        audio_file.save(temp_audio.name)
        temp_audio_path = temp_audio.name
    
    try:
        result = model.transcribe(
            temp_audio_path, 
            task="transcribe", 
            word_timestamps=True,
            fp16=False
        )
        
        current_subtitles = []
        for segment in result['segments']:
            current_subtitles.append({
                'start': segment['start'],
                'end': segment['end'],
                'text': segment['text'].strip()
            })
        
        logger.info(f"Successfully transcribed audio: {len(current_subtitles)} segments")
        return jsonify(current_subtitles)
    
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        return jsonify({"error": f"Error during transcription: {e}"}), 500
    
    finally:
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)

@app.route('/save-subtitles', methods=['GET'])
def save_subtitles():
    """
    Save current subtitles to a user-selected file.
    
    Returns:
        JSON response with save status or error details
    """
    global current_subtitles
    
    if current_subtitles is None:
        logger.warning("No subtitles available to save")
        return jsonify({"error": "No subtitles available to save."}), 404
    
    try:
        file_path = show_save_dialog(current_subtitles)
        
        if not file_path:
            logger.info("Save cancelled by user")
            return jsonify({"message": "Save cancelled by user"}), 200
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(current_subtitles, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Subtitles saved to: {file_path}")
        return jsonify({"message": "Subtitles saved successfully", "path": file_path}), 200
    
    except Exception as e:
        logger.error(f"Error saving subtitles: {e}")
        return jsonify({"error": f"Error saving subtitles: {e}"}), 500

if __name__ == '__main__':
    app.run(debug=False, port=5000)