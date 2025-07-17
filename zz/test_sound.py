# Save this as create_alert_sound.py and run it once
from gtts import gTTS
from pydub import AudioSegment
import os

def create_alert_sound(filename="alert.wav", text="Unattended bag detected!"):
    try:
        # Create gTTS object
        tts = gTTS(text=text, lang='en')
        temp_mp3 = "temp_alert.mp3"
        tts.save(temp_mp3)
        print(f"Temporary MP3 created: {temp_mp3}")

        # Convert MP3 to WAV using pydub
        audio = AudioSegment.from_mp3(temp_mp3)
        audio.export(filename, format="wav")
        print(f"Alert sound '{filename}' created successfully.")

        # Clean up temporary file
        os.remove(temp_mp3)
        print(f"Temporary MP3 removed: {temp_mp3}")

    except ImportError:
        print("Please install gTTS and pydub: pip install gTTS pydub")
        print("You might also need ffmpeg: 'conda install -c conda-forge ffmpeg' or 'sudo apt-get install ffmpeg'")
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Ensure you have an internet connection for gTTS to work, and ffmpeg for pydub to convert.")

if __name__ == "__main__":
    create_alert_sound()