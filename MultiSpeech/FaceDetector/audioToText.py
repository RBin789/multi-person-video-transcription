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