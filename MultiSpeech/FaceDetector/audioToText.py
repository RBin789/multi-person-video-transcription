import os
import moviepy.editor as mp
import speech_recognition as sr

def audio_to_text(audio_file_path):
    recognizer = sr.Recognizer()
    
    with sr.AudioFile(audio_file_path) as source:
        audio_data = recognizer.record(source)  
    try:
        text = recognizer.recognize_google(audio_data, language='en-US')  
        return text
    except sr.UnknownValueError:
        return "Could not understand the audio."
    except sr.RequestError as e:
        return f"Could not access Google Speech Recognition service: {e}"

def find_audio_start_frame(video_clip, threshold=0.01):
    """Find the frame number where audio starts based on a threshold."""
    audio = video_clip.audio
    # Convert the audio waveform to an array where each value represents the amplitude at that time
    # fps is the frames per second of the audio, which may be different from the video fps
    fps = audio.fps
    audio_waveform = audio.to_soundarray(fps=fps)

    # Calculate the magnitude of the audio waveform (stereo or mono)
    magnitudes = (audio_waveform**2).sum(axis=1)**0.5

    # Find the first frame where the magnitude exceeds the threshold
    for i, magnitude in enumerate(magnitudes):
        if magnitude > threshold:
            return int(i / fps * video_clip.fps)  # Convert audio frame index to video frame index
    return None

if __name__ == "__main__":
    project_root = os.path.dirname(os.path.abspath(__file__)) 
    video_file_path = os.path.join(project_root, "videos", "Example Video 1 - Jordan Peterson Confronts Australian Politician on Gender Politics and Quotas Q&A - Trim.mp4")
    
    video = mp.VideoFileClip(video_file_path)
    video_name = os.path.splitext(os.path.basename(video_file_path))[0]  
    audio_file_path = os.path.join(project_root, f"{video_name}.wav")  
    
    video.audio.write_audiofile(audio_file_path)
    
    text = audio_to_text(audio_file_path)
    print("Converted speech to text:")
    print(text)