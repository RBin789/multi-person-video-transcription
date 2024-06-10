import sys
import time
import cv2
import dlib
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from tkinter import *
from tkinter import messagebox
from tkinter import filedialog
sys.path.insert(0, 'MultiSpeech\FaceDetector')
import tensorflow as tf
from tensorflow import keras
from FaceDetector.Face2Vec import *
from FaceDetector.Person import *
from FaceDetector.FaceSequenceProcessor import *
from FaceDetector.Face import *
from FaceDetector.CreateTranscript import *
from ultralytics import YOLO
from PIL import Image, ImageTk, ImageOps
from moviepy.editor import VideoFileClip
from threading import Thread
import pygame
import tkinter as tk
from tkinter import Label, Button, Canvas, Frame
from vosk import Model, KaldiRecognizer
from pydub import AudioSegment
import wave
import json

all_faces = []
selected_file = None  # Initialize variable to store video path
lip_detection_model = tf.keras.models.load_model("MultiSpeech/FaceDetector/models/lip_detection_model.keras")

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
    model_path = "MultiSpeech/FaceDetector/models/vosk-model-small-en-us-0.15"
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


def process_video(video_path):

    # Open the video file
    video = cv2.VideoCapture(video_path)

    # Check if the video file was opened successfully
    if not video.isOpened():
        print(f"Video file does not exist: {video_path}")
        exit()

    face_detector = dlib.get_frontal_face_detector()
    landmark_predictor = dlib.shape_predictor("MultiSpeech/FaceDetector/models/shape_predictor_68_face_landmarks.dat")
    yolo_model = YOLO("MultiSpeech/FaceDetector/models/train4best.pt")
    current_frame_num = 1
    success, frame = video.read() # Read the first frame

    while success:
        print("-----------------------------------------------------------------------------------------------------------")
        print("Frame Number: " + str(current_frame_num))
        face2vec = Face2Vec(frame, current_frame_num, face_detector, landmark_predictor, yolo_model)
        face_features = face2vec.get_face_features()

        for faceid, face_features in enumerate(face_features):
            face = Face(face_features[0], face_features[1], face_features[2], face_features[3], face_features[4]) # Create a new person object (face vector, frame number, lip_seperation, bounding_box, face_coordinates)
            all_faces.append(face)
            print("face: " + str(faceid))
            print("lipSep: " + str(face.get_lip_seperation()))
            print("boundBox: " + str(face.get_bounding_box()))
            print()
        
        success, frame = video.read()
        current_frame_num += 1

    # Release the video file
    video.release()
    cv2.destroyAllWindows()

def run_gui():
    app = GUI()

class GUI:
    def __init__(self):
        self.selected_file = None
        self.output_video_path = None
        self.number_people = None
        self.transcriptions = []
        self.playing = False
        self.run_gui()

    def run_gui(self):
        self.MyWindow = tk.Tk()
        self.MyWindow.title("Multi Person Video Transcription")
        self.MyWindow.geometry('800x500')

        self.main_frame = Frame(self.MyWindow, bg='#f0f0f0')
        self.main_frame.pack(fill=BOTH, expand=True, padx=20, pady=20)

        self.top_frame = Frame(self.main_frame, bg='#f0f0f0', height=100)
        self.top_frame.pack(fill=X, side=TOP, expand=False)

        self.title_label = Label(self.top_frame, text="Convert a video to a transcript", font=("Arial", 20, "bold"), fg="#333333", bg='#f0f0f0')
        self.title_label.pack(side=TOP, pady=10)

        self.description_label = Label(self.top_frame, text="This tool allows you to convert a video to a transcript. To get started, press the open video button.", font=("Arial", 10), fg="#666666", bg='#f0f0f0')
        self.description_label.pack(side=TOP, pady=10)

        self.middle_frame = Frame(self.main_frame, bg='#f0f0f0', height=200)
        self.middle_frame.pack(fill=BOTH, expand=True, pady=20)

        self.open_video_btn = Button(self.middle_frame, text="Open Video", command=self.BtnOpen_Clicked, font=("Arial", 14), bg='#cccccc', fg='#333333')
        self.open_video_btn.pack(pady=10)

        self.people_input_label = Label(self.middle_frame, text="Enter the number of unique people who appear in the video:", font=("Arial", 10), fg='#666666', bg='#f0f0f0')
        self.people_input_label.pack(pady=10)

        self.number_entry = Entry(self.middle_frame, width=20, font=("Arial", 12), bg='#ffffff', fg='#333333')
        self.number_entry.pack(pady=10)
        self.number_entry.insert(0, "")

        self.start_button = Button(self.middle_frame, text="Start", state=DISABLED, command=self.BtnStart_Clicked, font=("Arial", 14), bg='#e0e0e0', fg='#999999')
        self.start_button.pack(pady=10)

        self.bottom_frame = Frame(self.main_frame, bg='#f0f0f0', height=50)
        self.bottom_frame.pack(fill=X, side=BOTTOM, pady=10, expand=False)

        self.next_button = Button(self.bottom_frame, text="Next", state=DISABLED, command=self.BtnNext_Clicked, font=("Arial", 14), bg='#e0e0e0', fg='#999999')
        self.next_button.pack(side=RIGHT, padx=10, pady=5)

        self.MyWindow.mainloop()

    def BtnOpen_Clicked(self):
        filetypes = [("Video files", "*.mp4")]
        file_path = filedialog.askopenfilename(filetypes=filetypes)
        print(file_path)

        if file_path:
            self.selected_file = file_path
            self.start_button.config(state=NORMAL, bg='#cccccc', fg='#333333')
            self.next_button.config(state=NORMAL, bg='#cccccc', fg='#333333')

    def BtnStart_Clicked(self):
        self.number_people = int(self.number_entry.get())
        print(f"Processing video with num people being: {self.number_people}")

        process_video(self.selected_file)

        face_sequence_processor = FaceSequenceProcessor(all_faces, self.number_people, lip_detection_model, self.selected_file)
        persons = face_sequence_processor.get_persons()

        create_transcript = CreateTranscript(self.selected_file, persons)
        
        # self.output_video_path = self.selected_file + "_modified.mp4"

        # Audio to text conversion
        # self.audio_path = self.selected_file.replace(".mp4", "_16k.wav")
        # video_clip = VideoFileClip(self.selected_file)
        # video_clip.audio.write_audiofile(self.audio_path.replace("_16k.wav", ".wav"))
        # self.transcriptions = audio_to_text(self.audio_path.replace("_16k.wav", ".wav"))

        messagebox.showinfo("Finished", "The video transcription has been completed.", parent=self.MyWindow)

    def BtnNext_Clicked(self):
        self.MyWindow.destroy()
        self.open_second_gui(self.output_video_path, self.number_people, self.transcriptions)

    def open_second_gui(self, video_path, number_people, transcriptions):
        SecondGUI(video_path, number_people, transcriptions)

class SecondGUI:
    def __init__(self, video_path, number_people, transcriptions):
        self.window = tk.Tk()
        self.window.title("Video Analysis Result")
        self.window.geometry("1600x900")

        self.playing = False
        self.cap = cv2.VideoCapture(video_path)
        self.current_frame = None
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.interval = int(1000 / self.fps)

        self.audio_thread = None
        self.audio_path = video_path.replace(".mp4", ".wav")
        video_clip = VideoFileClip(video_path)
        video_clip.audio.write_audiofile(self.audio_path)
        print(f"Audio path: {self.audio_path}")

        self.transcriptions = transcriptions
        self.transcriptions_index = 0

        self.main_frame = Frame(self.window, bg='#ffffff')
        self.main_frame.pack(fill=BOTH, expand=True, padx=20, pady=20)

        self.top_frame = Frame(self.main_frame, bg='#ffffff', height=200)
        self.top_frame.pack(fill=X, side=TOP, expand=False)

        self.modified_video_label = Label(self.top_frame, text="Modified Video", font=("Arial", 16, "bold"), fg="black", bg='#ffffff')
        self.modified_video_label.pack(side=LEFT, padx=20, anchor='nw')

        self.zoomed_face_label = Label(self.top_frame, text="Current Speaker", font=("Arial", 16, "bold"), fg="black", bg='#ffffff')
        self.zoomed_face_label.place(x=1220, y=0)

        self.middle_frame = Frame(self.main_frame, bg='#ffffff', height=600)
        self.middle_frame.pack(fill=BOTH, expand=True, pady=(10, 0))

        self.left_frame = Frame(self.middle_frame, width=1152, height=648, bg='#ffffff', relief='ridge', bd=2)
        self.left_frame.pack(side=LEFT, padx=20, pady=20, expand=True, fill=BOTH)
        self.canvas = Canvas(self.left_frame, width=1152, height=648, bg='#ffffff')
        self.canvas.pack(expand=True, fill=BOTH)

        self.right_frame = Frame(self.middle_frame, width=360, height=320, bg='#ffffff', relief='ridge', bd=2)
        self.right_frame.pack(side=RIGHT, padx=20, pady=20, expand=True, fill=BOTH)
        self.right_canvas = Canvas(self.right_frame, width=360, height=320, bg='#ffffff')
        self.right_canvas.pack(expand=True, fill=BOTH)

        self.text_box_frame = Frame(self.right_frame, width=360, height=320, bg='#ffffff', relief='ridge', bd=2)
        self.text_box_frame.pack(pady=10, expand=True, fill=BOTH)
        self.text_box_label = Label(self.text_box_frame, text="Transcription will appear here", font=("Arial", 16, "bold"), fg="black", bg='#ffffff')
        self.text_box_label.pack()

        self.bottom_frame = Frame(self.main_frame, bg='#ffffff', height=50)
        self.bottom_frame.pack(fill=X, side=BOTTOM, pady=10, expand=False)

        self.play_button = Button(self.bottom_frame, text="play", font=("Arial", 14), command=self.play_video, width=10, bg='#cccccc', fg='#333333')
        self.play_button.pack(side=LEFT, padx=20, pady=5, anchor='sw')
        self.stop_button = Button(self.bottom_frame, text="stop", font=("Arial", 14), command=self.stop_video, width=10, bg='#cccccc', fg='#333333')
        self.stop_button.pack(side=LEFT, padx=20, pady=5, anchor='sw')

        self.update_video_display()
        self.window.mainloop()

    def play_audio(self):
        pygame.mixer.music.load(self.audio_path)
        pygame.mixer.music.play()
        pygame.mixer.music.set_endevent(pygame.USEREVENT)
        print("Playing audio")

    def play_video(self):
        self.playing = True
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.start_time = time.time()

        if self.audio_thread is None or not self.audio_thread.is_alive():
            self.audio_thread = Thread(target=self.play_audio)
            self.audio_thread.start()
            print("Started audio thread")
        else:
            pygame.mixer.music.stop()
            pygame.mixer.music.play()
            self.start_time = time.time()
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            print("Rewind and play audio")

    def stop_video(self):
        self.playing = False
        pygame.mixer.music.stop()
        print("Stopped audio")
        if self.audio_thread is not None:
            self.audio_thread.join()
            self.audio_thread = None
            print("Joined audio thread")

    def update_video_display(self):
        for event in pygame.event.get():
            if event.type == pygame.USEREVENT:
                self.stop_video()
                return
        if self.playing:
            elapsed_time = time.time() - self.start_time
            frame_number = int(elapsed_time * self.fps)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = frame
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                canvas_width = self.canvas.winfo_width()
                canvas_height = self.canvas.winfo_height()
                frame_height, frame_width, _ = frame_rgb.shape
                scale_w = canvas_width / frame_width
                scale_h = canvas_height / frame_height
                scale = min(scale_w, scale_h)
                new_width = int(frame_width * scale)
                new_height = int(frame_height * scale)
                resized_frame = cv2.resize(frame_rgb, (new_width, new_height))
                img = Image.fromarray(resized_frame)
                imgtk = ImageTk.PhotoImage(image=img)
                self.canvas.create_image(0, 0, anchor=NW, image=imgtk)
                self.canvas.image = imgtk
                self.update_zoomed_face()
                self.update_transcription(elapsed_time)
            else:
                self.cap.release()
                self.playing = False
        self.window.after(self.interval, self.update_video_display)

    def update_zoomed_face(self):
        if self.current_frame is not None:
            for face in all_faces:
                if face.get_frame_number() == int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)):
                    x1, y1, x2, y2 = face.get_bounding_box()
                    zoomed_face = self.current_frame[y1:y2, x1:x2]
                    zoomed_face_rgb = cv2.cvtColor(zoomed_face, cv2.COLOR_BGR2RGB)
                    right_canvas_width = self.right_canvas.winfo_width()
                    right_canvas_height = self.right_canvas.winfo_height()
                    face_height, face_width, _ = zoomed_face_rgb.shape
                    scale_w = right_canvas_width / face_width
                    scale_h = right_canvas_height / face_height
                    scale = min(scale_w, scale_h)
                    new_face_width = int(face_width * scale)
                    new_face_height = int(face_height * scale)
                    resized_zoomed_face = cv2.resize(zoomed_face_rgb, (new_face_width, new_face_height))
                    zoomed_img = Image.fromarray(resized_zoomed_face)
                    self.zoomed_image = ImageTk.PhotoImage(image=zoomed_img)
                    self.right_canvas.create_image(0, 0, anchor=NW, image=self.zoomed_image)
                    self.right_canvas.image = self.zoomed_image
                    break

    def update_transcription(self, elapsed_time):
        current_time = elapsed_time
        current_transcription = ""
        for start, end, word in self.transcriptions:
            if start <= current_time <= end:
                current_transcription += word + " "
        self.text_box_label.config(text=current_transcription)
                    
# ---------------------------------------------------------------------------------------------------------------------------

def main():
    total_time = time.monotonic()

    gui = GUI()

    print("Number of Face Vectors: ", len(all_faces))
    print("Total Time taken: ", time.monotonic() - total_time)
    
if __name__ == "__main__":
    main()