import whisper
import torch
import json
import os
import logging
import tempfile
from flask import Flask, request, jsonify, render_template, make_response, send_file
from flask_cors import CORS
import traceback
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 25 * 1024 * 1024  # Limit file size to 25 MB

# CORS setup
CORS(app, resources={r"/*": {"origins": "*", "allow_headers": ["Content-Type"]}})

# Detect and select device
def select_device():
    if torch.cuda.is_available():
        logger.info("Using CUDA GPU")
        return "cuda"
    elif torch.backends.mps.is_available():
        logger.info("Using Apple Metal Performance Shaders")
        return "mps"
    else:
        logger.info("Using CPU")
        return "cpu"

# Load the Whisper model
try:
    device = select_device()
    model = whisper.load_model("tiny", device=device)
    logger.info(f"Whisper model loaded on {device}")
except Exception as e:
    logger.error(f"Failed to load Whisper model: {e}")
    raise

@app.route("/")
def index():
    """Root route to render the main page."""
    return render_template("main.html")

@app.route("/tutorial")
def tutorial():
    """Route for the tutorial page."""
    return render_template("tutorial.html")

@app.route("/about")
def about():
    """Route for the about page."""
    return render_template("about.html")

@app.route("/transcribe", methods=["POST"])
def transcribe():
    start_time = time.time()

    # Check if an audio file is uploaded
    if "audio" not in request.files:
        return jsonify({"error": "No audio file uploaded."}), 400

    audio_file = request.files["audio"]

    # Validate file type
    allowed_extensions = (".mp3", ".wav", ".m4a", ".webm", ".ogg")
    if not audio_file.filename.lower().endswith(allowed_extensions):
        return jsonify({"error": "Invalid file type. Only audio files are allowed."}), 400

    temp_audio_path = None
    try:
        # Save uploaded audio to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_audio:
            audio_file.save(temp_audio.name)
            temp_audio_path = temp_audio.name

        # Check file size
        if os.stat(temp_audio_path).st_size == 0:
            return jsonify({"error": "Uploaded file is empty."}), 400

        # Transcribe audio
        logger.info(f"Starting transcription for {temp_audio_path}")
        result = model.transcribe(
            temp_audio_path,
            task="transcribe",
            word_timestamps=False,
            fp16=(device != "cpu")  # Use FP16 only if not on CPU
        )

        # Process transcription result
        subtitles = [
            {"start": segment.get("start", 0), "end": segment.get("end", 0), "text": segment["text"].strip()}
            for segment in result.get("segments", [])
        ]

        # Set global variable for subtitles
        global current_subtitles
        current_subtitles = subtitles

        process_time = time.time() - start_time
        logger.info(f"Transcription completed in {process_time:.2f} seconds")
        return jsonify({"subtitles": subtitles, "processing_time": process_time})

    except Exception as e:
        logger.error(f"Transcription error: {traceback.format_exc()}")
        return jsonify({"error": f"Transcription failed: {str(e)}"}), 500

    finally:
        # Clean up temporary file
        if temp_audio_path and os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)

@app.route('/save-subtitles', methods=['GET'])
def save_subtitles():
    global current_subtitles
    if current_subtitles is None:
        return jsonify({'error': 'No subtitles available to download'}), 400

    # Save the subtitles to a file (e.g., subtitles.json)
    subtitles_file_path = 'subtitles.json'
    with open(subtitles_file_path, 'w') as file:
        json.dump(current_subtitles, file)

    # Send the file for download
    return send_file(subtitles_file_path, as_attachment=True)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
