import whisper
import json
import os
import logging
import tempfile
from flask import Flask, request, jsonify, render_template, make_response
from flask_cors import CORS
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # Limit file size to 50 MB

# CORS setup: Restrict in production
CORS(app, resources={
    r"/*": {
        "origins": "http://localhost:3000",  # Update with your domain in production
        "allow_headers": ["Content-Type", "Authorization"],
        "supports_credentials": True,
    }
})

# Global variable to store subtitles temporarily
current_subtitles = None

# Load the Whisper model
try:
    # Load a lightweight model; use CPU if no GPU is available
    model = whisper.load_model("tiny")
    logger.info("Whisper model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load Whisper model: {e}")
    raise

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
    global current_subtitles

    if "audio" not in request.files:
        logger.warning("No audio file uploaded")
        return jsonify({"error": "No audio file uploaded."}), 400

    audio_file = request.files["audio"]

    # Validate file type
    allowed_extensions = (".mp3", ".wav", ".m4a", ".webm", ".ogg")
    if not audio_file.filename.lower().endswith(allowed_extensions):
        logger.warning(f"Invalid file type: {audio_file.filename}")
        return jsonify({"error": "Invalid file type. Only audio files are allowed."}), 400

    temp_audio_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_audio:
            audio_file.save(temp_audio.name)
            temp_audio_path = temp_audio.name

        # Check if the uploaded file is empty
        if os.stat(temp_audio_path).st_size == 0:
            logger.warning("Uploaded file is empty")
            return jsonify({"error": "Uploaded file is empty."}), 400

        # Transcribe the audio
        result = model.transcribe(temp_audio_path, task="transcribe", word_timestamps=True, fp16=False)

        if "segments" not in result:
            logger.error("Whisper did not return any segments")
            return jsonify({"error": "Failed to transcribe audio. No subtitles generated."}), 500

        current_subtitles = [
            {"start": segment["start"], "end": segment["end"], "text": segment["text"].strip()}
            for segment in result["segments"]
        ]
        logger.info(f"Successfully transcribed {len(current_subtitles)} segments")
        return jsonify(current_subtitles)

    except Exception as e:
        logger.error(f"Transcription error: {traceback.format_exc()}")
        return jsonify({"error": f"Error during transcription: {str(e)}"}), 500

    finally:
        # Ensure temporary file is deleted
        if temp_audio_path and os.path.exists(temp_audio_path):
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
        # Create JSON string
        json_data = json.dumps(current_subtitles, ensure_ascii=False, indent=2)
        
        # Create response with proper headers for file download
        response = make_response(json_data)
        response.headers['Content-Disposition'] = 'attachment; filename=subtitles.json'
        response.headers['Content-Type'] = 'application/json; charset=utf-8'
        
        return response

    except Exception as e:
        logger.error(f"Error saving subtitles: {traceback.format_exc()}")
        return jsonify({"error": f"Error saving subtitles: {str(e)}"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Use the PORT environment variable for Render
    app.run(debug=False, host="0.0.0.0", port=port)
