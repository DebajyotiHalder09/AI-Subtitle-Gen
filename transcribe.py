import whisper
import json
import os
import logging
import tempfile
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import sys
from flask import send_file


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={
    r"/*": {
        "origins": "*",
        "allow_headers": ["Content-Type", "Authorization"],
        "supports_credentials": True,
    }
})

# Global variable to store subtitles temporarily
current_subtitles = None

# Load the Whisper model
try:
    model = whisper.load_model("tiny")
    logger.info("Whisper model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load Whisper model: {e}")
    sys.exit(1)

# Routes
@app.route("/")
def index():
    """
    Root route to render the main page.
    """
    return render_template("main.html")

@app.route("/tutorial")
def tutorial():
    """
    Route for the tutorial page.
    """
    return render_template("tutorial.html")

@app.route("/about")
def about():
    """
    Route for the about page.
    """
    return render_template("about.html")

@app.route("/transcribe", methods=["POST"])
def transcribe():
    """
    Transcribe uploaded audio file using Whisper.
    
    Returns:
        JSON response with transcription or error details
    """
    global current_subtitles

    if "audio" not in request.files:
        logger.warning("No audio file uploaded")
        return jsonify({"error": "No audio file uploaded."}), 400

    audio_file = request.files["audio"]

    allowed_extensions = (".mp3", ".wav", ".m4a", ".webm", ".ogg")
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
            fp16=False,
        )

        current_subtitles = []
        for segment in result["segments"]:
            current_subtitles.append({
                "start": segment["start"],
                "end": segment["end"],
                "text": segment["text"].strip(),
            })

        logger.info(f"Successfully transcribed audio: {len(current_subtitles)} segments")
        return jsonify(current_subtitles)

    except Exception as e:
        logger.error(f"Transcription error: {e}")
        return jsonify({"error": f"Error during transcription: {e}"}), 500

    finally:
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)

@app.route("/save-subtitles", methods=["GET"])
def save_subtitles():
    """
    Serve current subtitles as a downloadable JSON file.

    Returns:
        File response with subtitles or error details.
    """
    global current_subtitles

    if current_subtitles is None:
        logger.warning("No subtitles available to save")
        return jsonify({"error": "No subtitles available to save."}), 404

    try:
        # Create a temporary JSON file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
        temp_file_path = temp_file.name
        temp_file.close()

        with open(temp_file_path, "w", encoding="utf-8") as f:
            json.dump(current_subtitles, f, ensure_ascii=False, indent=2)

        # Send the file to the client for download
        return send_file(temp_file_path, as_attachment=True, download_name="subtitles.json")

    except Exception as e:
        logger.error(f"Error saving subtitles: {e}")
        return jsonify({"error": f"Error saving subtitles: {e}"}), 500

    finally:
        # Ensure the temporary file is deleted after sending
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Use the PORT environment variable
    app.run(debug=False, host="0.0.0.0", port=port)
