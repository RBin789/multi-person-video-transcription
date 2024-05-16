from moviepy.editor import VideoFileClip
from pydub import AudioSegment
from pydub.silence import split_on_silence
import os

def extract_and_save_audio_clips(video_path):
    video_clip = VideoFileClip(video_path)
    audio_clip = video_clip.audio
    audio_path = "temp_audio.wav"
    audio_clip.write_audiofile(audio_path)

    sound = AudioSegment.from_wav(audio_path)

    # Split the audio based on silence
    audio_chunks = split_on_silence(sound, 
        min_silence_len=500,
        silence_thresh=sound.dBFS-14,
        keep_silence=500
    )

    # Create a directory to store the audio clips
    if not os.path.exists("audio_clips"):
        os.makedirs("audio_clips")

    for i, chunk in enumerate(audio_chunks):
        out_file = f"audio_clips/chunk{i}.wav"
        print(f"Exporting {out_file}")
        chunk.export(out_file, format="wav")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python Vid2Wav.py <video_file_path>")
    else:
        video_file_path = sys.argv[1]
        extract_and_save_audio_clips(video_file_path)
