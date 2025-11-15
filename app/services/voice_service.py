import os
import uuid

# Guard gTTS import
try:
    from gtts import gTTS
    from gtts.lang import tts_langs
    _GTTS_AVAILABLE = True
except Exception:
    gTTS = None
    _GTTS_AVAILABLE = False


class VoiceService:
    def __init__(self):
        # Create audio directory if it doesn't exist
        self.audio_dir = "audio_files"
        os.makedirs(self.audio_dir, exist_ok=True)

        # Supported languages for TTS
        self.supported_languages = {
            "en": "English",
            "es": "Spanish",
            "fr": "French",
            "de": "German",
            "it": "Italian",
            "pt": "Portuguese",
            "ru": "Russian",
            "ja": "Japanese",
            "ko": "Korean",
            "zh": "Chinese",
            "ar": "Arabic",
            "hi": "Hindi"
        }

    def text_to_speech(self, text: str, language: str = "en") -> str:
        """Convert text to speech and save as audio file

        Returns the generated filename on success or an empty string on failure.
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        # Validate language
        if language not in self.supported_languages:
            language = "en"  # Default to English

        # Generate unique filename
        filename = f"{uuid.uuid4()}.mp3"
        file_path = os.path.join(self.audio_dir, filename)

        # If gTTS not available, return empty string to indicate failure gracefully
        if not _GTTS_AVAILABLE:
            print("gTTS not available — text-to-speech disabled in this environment.")
            return ""

        try:
            # Create TTS object and save
            tts = gTTS(text=text, lang=language, slow=False)
            tts.save(file_path)
            return filename
        except Exception as e:
            print(f"Text-to-speech failed: {e}")
            return ""

    def get_audio_file_path(self, filename: str) -> str:
        """Get full path to audio file"""
        return os.path.join(self.audio_dir, filename)

    def delete_audio_file(self, filename: str) -> bool:
        """Delete audio file"""
        try:
            file_path = self.get_audio_file_path(filename)
            if os.path.exists(file_path):
                os.remove(file_path)
                return True
            return False
        except Exception as e:
            print(f"Failed to delete audio file: {e}")
            return False

    def list_audio_files(self) -> list:
        """List all audio files"""
        try:
            files = []
            for filename in os.listdir(self.audio_dir):
                if filename.endswith('.mp3'):
                    file_path = os.path.join(self.audio_dir, filename)
                    files.append({
                        "filename": filename,
                        "size": os.path.getsize(file_path)
                    })
            return files
        except Exception as e:
            print(f"Failed to list audio files: {e}")
            return []