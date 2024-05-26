import os
import wave
import json
from pydub import AudioSegment
from vosk import Model, KaldiRecognizer

# Function to convert audio to required format
def convert_audio(audio_path):
    sound = AudioSegment.from_file(audio_path)
    sound = sound.set_channels(1)
    sound = sound.set_frame_rate(16000)
    converted_path = audio_path.replace(".wav", "_16k.wav")
    sound.export(converted_path, format="wav")
    return converted_path

# Function to convert audio to text using Vosk with timestamps
def audio_to_text(audio_path):
    model_path = "MultiSpeech/FaceDetector/models/vosk-model-small-en-us-0.15"  # Update with the path to your Vosk model directory
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}, please download and unzip the model from https://alphacephei.com/vosk/models")
        return []
    
    model = Model(model_path)
    recognizer = KaldiRecognizer(model, 16000)
    recognizer.SetWords(True)

    audio_path = convert_audio(audio_path)

    wf = wave.open(audio_path, "rb")
    results = []

    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if recognizer.AcceptWaveform(data):
            results.append(json.loads(recognizer.Result()))
    
    results.append(json.loads(recognizer.FinalResult()))
    wf.close()

    transcriptions = []
    for result in results:
        if 'result' in result:
            for word_info in result['result']:
                transcriptions.append((word_info['start'], word_info['end'], word_info['word']))
    return transcriptions

if __name__ == "__main__":
    audio_file = "MultiSpeech/FaceDetector/videos/One_Plus_One_11s_clip.wav_modified.wav"
    
    if not os.path.exists(audio_file):
        print(f"Audio file not found at {audio_file}")
    else:
        transcriptions = audio_to_text(audio_file)
        if transcriptions:
            for start, end, word in transcriptions:
                print(f"{start}-{end}: {word}")
        else:
            print("No transcriptions found.")