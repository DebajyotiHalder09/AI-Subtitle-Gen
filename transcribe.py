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

# Enable CORS
CORS(app, resources={r"/*": {"origins": "*"}})

# Initialize global variable for subtitles
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
    return render_template("test.html")

@app.route("/transcribe", methods=["POST"])
@app.route("/transcribe", methods=["POST"])
def transcribe():
    """Handles audio transcription and updates global subtitles."""
    start_time = time.time()

    if "audio" not in request.files:
        return jsonify({"error": "No audio file uploaded."}), 400

    audio_file = request.files["audio"]
    allowed_extensions = (".mp3", ".wav", ".m4a", ".webm", ".ogg")
    if not audio_file.filename.lower().endswith(allowed_extensions):
        return jsonify({"error": "Invalid file type. Only audio files are allowed."}), 400

    temp_audio_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_audio:
            audio_file.save(temp_audio.name)
            temp_audio_path = temp_audio.name

        if os.stat(temp_audio_path).st_size == 0:
            return jsonify({"error": "Uploaded file is empty."}), 400

        result = model.transcribe(
            temp_audio_path,
            task="transcribe",
            word_timestamps=False,
            fp16=(device != "cpu")
        )

        global current_subtitles
        current_subtitles = [
            {"start": segment.get("start", 0), "end": segment.get("end", 0), "text": segment["text"].strip()}
            for segment in result.get("segments", [])
        ]

        process_time = time.time() - start_time
        return jsonify({"subtitles": current_subtitles, "processing_time": process_time})

    except Exception as e:
        logger.error(f"Transcription error: {traceback.format_exc()}")
        return jsonify({"error": f"Transcription failed: {str(e)}"}), 500

    finally:
        if temp_audio_path and os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)


@app.route('/generate', methods=['POST'])
def generate():
    """Generates and sends the subtitles JSON file."""
    global current_subtitles
    if current_subtitles is None:
        return jsonify({'error': 'No subtitles available. Please transcribe first.'}), 400

    try:
        temp_json_path = os.path.join(tempfile.gettempdir(), "subtitles.json")
        with open(temp_json_path, 'w', encoding='utf-8') as temp_json:
            json.dump(current_subtitles, temp_json, ensure_ascii=False, indent=4)

        return send_file(temp_json_path, as_attachment=True, download_name="subtitles.json", mimetype='application/json')

    except Exception as e:
        logger.error(f"Error generating JSON: {traceback.format_exc()}")
        return jsonify({'error': 'Failed to generate JSON.'}), 500

@app.route('/save-subtitles', methods=['GET'])
def save_subtitles():
    global current_subtitles
    if current_subtitles is None:
        return jsonify({'error': 'No subtitles available to download'}), 400

    # Save the subtitles to a temporary file
    try:
        temp_json_path = os.path.join(tempfile.gettempdir(), "subtitles.json")
        with open(temp_json_path, 'w', encoding='utf-8') as temp_json:
            json.dump(current_subtitles, temp_json, ensure_ascii=False, indent=4)

        # Send the file for download
        return send_file(temp_json_path, as_attachment=True, download_name="subtitles.json", mimetype='application/json')

    except Exception as e:
        logger.error(f"Error saving subtitles: {traceback.format_exc()}")
        return jsonify({'error': 'Failed to save subtitles.'}), 500


if __name__ == "__main__":
    # Use environment variable PORT for hosting services like Render
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
