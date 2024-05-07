import os
from moviepy.editor import VideoFileClip
import openai
from openai import OpenAI


class audioToText:

    def __init__(self, video_path):

        self.video_path = video_path
        # Load the video clip
        self.video_clip = VideoFileClip(self.video_path)
        # Extract the audio from the video clip
        self.audio_clip = self.video_clip.audio
        # Save the audio to a mp3 file
        self.audio_file_path = os.path.splitext(self.video_path)[0] + ".mp3"
        self.audio_clip.write_audiofile(self.audio_file_path)
        # Close the video and audio clips
        self.audio_clip.close()
        self.video_clip.close()

        self.transcribe_audio()

    def transcribe_audio(self):
        
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        audio_file= open(self.audio_file_path, "rb")
        transcript = client.audio.transcriptions.create(
        file=audio_file,
        model="whisper-1",
        response_format="verbose_json",
        timestamp_granularities=["segment"]
        )

        print(transcript.words)
    
    #     recognizer = sr.Recognizer()
        
    #     with sr.AudioFile(audio_file_path) as source:
    #         audio_data = recognizer.record(source)  
    #     try:
    #         text = recognizer.recognize_google(audio_data, language='en-US')  
    #         return text
    #     except sr.UnknownValueError:
    #         return "Could not understand the audio."
    #     except sr.RequestError as e:
    #         return f"Could not access Google Speech Recognition service: {e}"

    # def find_audio_start_frame(video_clip, threshold=0.01):
    #     """Find the frame number where audio starts based on a threshold."""
    #     audio = video_clip.audio
    #     # Convert the audio waveform to an array where each value represents the amplitude at that time
    #     # fps is the frames per second of the audio, which may be different from the video fps
    #     fps = audio.fps
    #     audio_waveform = audio.to_soundarray(fps=fps)

    #     # Calculate the magnitude of the audio waveform (stereo or mono)
    #     magnitudes = (audio_waveform**2).sum(axis=1)**0.5

    #     # Find the first frame where the magnitude exceeds the threshold
    #     for i, magnitude in enumerate(magnitudes):
    #         if magnitude > threshold:
    #             return int(i / fps * video_clip.fps)  # Convert audio frame index to video frame index
    #     return None

    # if __name__ == "__main__":
    #     project_root = os.path.dirname(os.path.abspath(__file__)) 
    #     video_file_path = os.path.join(project_root, "videos", "Example Video 1 - Jordan Peterson Confronts Australian Politician on Gender Politics and Quotas Q&A - Trim.mp4")
        
    #     video = mp.VideoFileClip(video_file_path)
    #     video_name = os.path.splitext(os.path.basename(video_file_path))[0]  
    #     audio_file_path = os.path.join(project_root, f"{video_name}.wav")  
        
    #     video.audio.write_audiofile(audio_file_path)
        
    #     text = audio_to_text(audio_file_path)
    #     print("Converted speech to text:")
    #     print(text)