import whisper
import torch
import json
import os
import logging
import tempfile
from flask import Flask, request, jsonify, render_template, send_file
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

# Enable CORS with more specific configuration
CORS(app, resources={
    r"/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization", "Accept"]
    }
})

# Initialize global variable for subtitles
current_subtitles = None

# Detect and select device
def select_device():
    try:
        if torch.cuda.is_available():
            logger.info("Using CUDA GPU")
            return "cuda"
        elif torch.backends.mps.is_available():
            logger.info("Using Apple Metal Performance Shaders")
            return "mps"
        else:
            logger.info("Using CPU")
            return "cpu"
    except Exception as e:
        logger.error(f"Error selecting device: {e}")
        return "cpu"

# Load the Whisper model with error handling
def load_whisper_model():
    try:
        device = select_device()
        model = whisper.load_model("tiny", device=device)
        logger.info(f"Whisper model loaded successfully on {device}")
        return model
    except Exception as e:
        logger.error(f"Failed to load Whisper model: {e}")
        raise RuntimeError(f"Failed to initialize Whisper model: {str(e)}")

# Initialize model
try:
    model = load_whisper_model()
except Exception as e:
    logger.error(f"Critical error loading model: {e}")
    model = None  # Will trigger error handling in routes

@app.route("/")
def index():
    """Root route to render the main page."""
    return render_template("test.html")

@app.route("/transcribe", methods=["POST", "OPTIONS"])
def transcribe():
    """Handles audio transcription and updates global subtitles."""
    if request.method == "OPTIONS":
        # Handle preflight request
        return "", 204

    if model is None:
        return jsonify({"error": "Server not properly initialized. Please contact administrator."}), 500

    logger.info("Received request at /transcribe")
    start_time = time.time()

    try:
        if "audio" not in request.files:
            return jsonify({"error": "No audio file uploaded."}), 400

        audio_file = request.files["audio"]
        if not audio_file.filename:
            return jsonify({"error": "No selected file."}), 400

        allowed_extensions = (".mp3", ".wav", ".m4a", ".webm", ".ogg")
        if not audio_file.filename.lower().endswith(allowed_extensions):
            return jsonify({"error": "Invalid file type. Only audio files are allowed."}), 400

        # Create temp file
        temp_audio_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_audio:
                audio_file.save(temp_audio.name)
                temp_audio_path = temp_audio.name

            if os.stat(temp_audio_path).st_size == 0:
                return jsonify({"error": "Uploaded file is empty."}), 400

            # Process with Whisper
            result = model.transcribe(
                temp_audio_path,
                task="transcribe",
                word_timestamps=False,
                fp16=(select_device() != "cpu")
            )

            # Format subtitles
            global current_subtitles
            current_subtitles = [
                {
                    "start": segment.get("start", 0),
                    "end": segment.get("end", 0),
                    "text": segment["text"].strip()
                }
                for segment in result.get("segments", [])
            ]

            process_time = time.time() - start_time
            logger.info(f"Transcription completed in {process_time:.2f} seconds")
            
            return jsonify({
                "status": "success",
                "subtitles": current_subtitles,
                "processing_time": process_time
            })

        finally:
            if temp_audio_path and os.path.exists(temp_audio_path):
                try:
                    os.remove(temp_audio_path)
                except Exception as e:
                    logger.error(f"Error removing temporary file: {e}")

    except Exception as e:
        error_msg = f"Transcription error: {str(e)}"
        logger.error(f"{error_msg}\n{traceback.format_exc()}")
        return jsonify({"error": error_msg}), 500

# Rest of your routes remain the same...

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)  # Set debug=False for production