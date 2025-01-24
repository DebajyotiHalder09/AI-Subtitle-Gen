import whisper
import torch
import json
import os
import logging
import tempfile
from flask import Flask, request, jsonify, render_template, make_response
from flask_cors import CORS
import traceback
import time
import threading

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 25 * 1024 * 1024  # Limit file size to 25 MB

# CORS setup for Render
CORS(app, resources={
    r"/*": {
        "origins": "*",  # More permissive for Render deployment
        "allow_headers": ["Content-Type", "Authorization"],
        "supports_credentials": True,
    }
})

# Global variable to store subtitles temporarily
current_subtitles = None

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

# Load the Whisper model with device selection
try:
    device = select_device()
    model = whisper.load_model("base", device=device)
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

def transcribe_audio(temp_audio_path):
    global current_subtitles
    try:
        # Transcribe with reduced complexity
        logger.info(f"Starting transcription for {temp_audio_path}")
        result = model.transcribe(
            temp_audio_path,
            task="transcribe",
            word_timestamps=False,
            fp16=(device != "cpu")  # Use FP16 only if not on CPU
        )

        # Process the transcription result
        current_subtitles = [
            {"start": segment.get("start", 0), "end": segment.get("end", 0), "text": segment["text"].strip()}
            for segment in result.get("segments", [])
        ]

        logger.info("Transcription completed successfully")

    except Exception as e:
        logger.error(f"Transcription error: {traceback.format_exc()}")

@app.route("/transcribe", methods=["POST"])
def transcribe():
    global current_subtitles
    start_time = time.time()

    if "audio" not in request.files:
        return jsonify({"error": "No audio file uploaded."}), 400

    audio_file = request.files["audio"]

    # Validate file type
    allowed_extensions = (".mp3", ".wav", ".m4a", ".webm", ".ogg")
    if not audio_file.filename.lower().endswith(allowed_extensions):
        return jsonify({"error": "Invalid file type. Only audio files are allowed."}), 400

    temp_audio_path = None
    try:
        # Create unique temporary file
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_audio:
            audio_file.save(temp_audio.name)
            temp_audio_path = temp_audio.name

        # Check file size
        if os.stat(temp_audio_path).st_size == 0:
            return jsonify({"error": "Uploaded file is empty."}), 400

        # Start transcription in a separate thread to avoid blocking Flask
        transcription_thread = threading.Thread(target=transcribe_audio, args=(temp_audio_path,))
        transcription_thread.start()

        # Wait for the transcription to finish (you can also set a timeout here)
        transcription_thread.join(timeout=60)  # Timeout after 60 seconds

        if current_subtitles is None:
            return jsonify({"error": "Transcription failed or timed out."}), 500

        process_time = time.time() - start_time
        logger.info(f"Transcription completed in {process_time:.2f} seconds")
        return jsonify({"subtitles": current_subtitles, "processing_time": process_time})

    except Exception as e:
        logger.error(f"Transcription error: {traceback.format_exc()}")
        return jsonify({"error": f"Transcription failed: {str(e)}"}), 500

    finally:
        # Clean up temporary file
        if temp_audio_path and os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)

@app.route("/save-subtitles", methods=["GET"])
def save_subtitles():
    global current_subtitles

    if current_subtitles is None:
        return jsonify({"error": "No subtitles available."}), 404

    try:
        json_data = json.dumps(current_subtitles, ensure_ascii=False, indent=2).encode('utf-8')

        response = make_response(json_data)
        response.headers['Content-Disposition'] = 'attachment; filename=subtitles.json'
        response.headers['Content-Type'] = 'application/json; charset=utf-8'

        return response
    except Exception as e:
        logger.error(f"Subtitle save error: {traceback.format_exc()}")
        return jsonify({"error": f"Save failed: {str(e)}"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
