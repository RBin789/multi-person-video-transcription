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
from FaceDetector.Sequence_Generation import *
from FaceDetector.Lip_Detection import *
from FaceDetector.audioToText import *
from FaceDetector.Person import *
from FaceDetector.CreateProcessedVideo import *
from ultralytics import YOLO
from PIL import Image, ImageTk, ImageOps
from moviepy.editor import VideoFileClip
from threading import Thread

all_persons = []
all_Sequences = []
selected_file = None  # Initialize variable to store video path
lip_detection_model = tf.keras.models.load_model("MultiSpeech/FaceDetector/models/lip_detection_model.keras")

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
        face2vec = Face2Vec(frame, current_frame_num, face_detector, landmark_predictor, yolo_model)
        face_features = face2vec.get_face_features()

        for face_features in face_features:
            person = Person(face_features[0], face_features[1], face_features[2], face_features[3], face_features[4]) # Create a new person object (face vector, frame number, lip_seperation, bounding_box, face_coordinates)
            all_persons.append(person)
        
        success, frame = video.read()
        current_frame_num += 1
        print("Frame Processed")
    # Release the video file
    video.release()
    cv2.destroyAllWindows()

def peform_kmeans_clustering(all_persons, num_people):
    # Extract vectors from the list
    vectors = [face_vector.get_face_vector().flatten() for face_vector in all_persons]

    # Convert the list of vectors to a NumPy array
    vector_array = np.array(vectors)

    # Create the KMeans model
    kmeans = KMeans(n_clusters=num_people)

    # Fit the model to the data (vectors only)
    kmeans.fit(vector_array)

    # Get the cluster labels for each vector
    cluster_labels = kmeans.labels_

    # Create a new list to store clustered data
    clustered_data = []

    # Assign cluster labels and preserve structure
    for item, label in zip(all_persons, cluster_labels):
        clustered_data.append([item.get_face_vector(), item.get_frame_number(), item.get_lip_seperation(), label])
        item.set_label(label)

    return clustered_data

def split_data_by_cluster(clustered_data):
    clustered_by_label = {}
    for item in clustered_data:
        cluster_label = item[3]  # Access the cluster label from the 4th element
        if cluster_label not in clustered_by_label:
            clustered_by_label[cluster_label] = []
        clustered_by_label[cluster_label].append(item)
    return clustered_by_label # A dictionary where keys are cluster labels and values are lists of data points belonging to that cluster.

def run_gui():
    app = GUI()

def process_clustered_data(clustered_by_label, model):
    for cluster_label, cluster_data in clustered_by_label.items():
        person_sequences = sequence_generation(cluster_label, cluster_data) # all of one persons sequences
        run_lip_detection(person_sequences, cluster_label, model)

def sequence_generation(face_vectors, cluster_label):
    # Generate sequences
    sequence_generation = Sequence_Generation(face_vectors, cluster_label)
    person_sequences = sequence_generation.get_person_sequences()
    return person_sequences

def run_lip_detection(person_sequences, cluster_label, model):
    for i, sequence in enumerate(person_sequences): # Loops though every sequence of a person
        if (len(sequence) == 0):
            continue
        lip_detection = Lip_Detection(sequence, cluster_label, model)
        all_Sequences.append(lip_detection.get_sequence_and_prediction())

def sort_Detected_Sequences(all_Sequences):
    all_Sequences.sort(key=lambda x: x[1])  # Sort by frame number
    
def update_persons(all_Sequences):
    for sequence in all_Sequences: # Loops though every sequence
        if sequence[2] == 1:
            for frame in sequence[1]: # Loops though every frame number in the sequence
                for person in all_persons: # Loops though every person
                    if (person.get_label() == sequence[0]) and (person.get_frame_number() == frame): # If the person label matches the sequence label
                        person.set_is_talking(2)
                    # else:
                    #     person.set_is_talking(1)


class GUI:

    def __init__(self):
        self.run_gui()
        self.selected_file = None  # Initialize variable to store video path
        self.output_video_path = None
        self.number_people = None

    def run_gui(self):
        # Create a Window.
        self.MyWindow = Tk()  # Create a window
        self.MyWindow.title("Multi Person Video Transcription")  # Change the Title of the GUI
        self.MyWindow.geometry('800x400')  # Set the size of the Windows

        # Create a new frame
        centered_frame = Frame(self.MyWindow, height=self.MyWindow.winfo_screenheight())
        centered_frame.pack(pady=(self.MyWindow.winfo_screenheight() - centered_frame.winfo_reqheight()) // 2)

        # Create GUI elements
        titleText = Label(centered_frame, text="Convert a video to a transcript", font=("Arial Bold", 20))
        titleText.pack(pady=20)

        descriptionText = Label(centered_frame, text="This tool allows you to convert a video to a transcript. To get started press the open video button.", font=("Arial Bold", 10))
        descriptionText.pack(pady=10)

        # Button to Open Video
        openVideoBtn = Button(centered_frame, text="Open Video", command=self.BtnOpen_Clicked)
        openVideoBtn.pack(pady=10)

        # Label for number of people
        peopleInputLabel = Label(centered_frame, text="Enter the amount of unique people who appear in the video:", font=("Arial Bold", 10))
        peopleInputLabel.pack(pady=10)

        # Text box for user input
        self.numberEntry = Entry(centered_frame, width=20)  # Create entry widget for number input
        self.numberEntry.pack(pady=10)
        self.numberEntry.insert(0, "")  # Set initial text in the entry

        # Start Button (initially disabled)
        self.startButton = Button(centered_frame, text="Start", state=DISABLED, command=self.BtnStart_Clicked)
        self.startButton.pack(pady=10)

        # Next Button (initially disabled)
        self.nextButton = Button(centered_frame, text="Next", state=DISABLED, command=self.BtnNext_Clicked)
        self.nextButton.pack(pady=10)

        # Calling the mainloop()
        self.MyWindow.mainloop()

    def BtnOpen_Clicked(self):
        # Specify video file types
        filetypes = [("Video files", "*.mp4")]

        file_path = filedialog.askopenfilename(filetypes=filetypes)
        print(file_path)

        if file_path:  # Check if a file was selected
            self.selected_file = file_path
            self.startButton.config(state=NORMAL)  # Enable Start button after selecting video
            self.nextButton.config(state=NORMAL)  # Enable Next button after selecting video

    def BtnStart_Clicked(self):
        # Access the entered number using self.numberEntry.get()
        self.number_people = int(self.numberEntry.get())
        print(f"Processing video with num people being: {self.number_people}")

        # Process the video
        process_video(self.selected_file)

        # Process Audio
        # audiototext = audioToText(self.selected_file)

        # K-means clustering on face vectors
        clustered_data = peform_kmeans_clustering(all_persons, self.number_people)
        clustered_by_label = split_data_by_cluster(clustered_data)

        # Generate sequences for each person and run lip detection
        process_clustered_data(clustered_by_label, lip_detection_model)

        
        # Sort all_Sequences by frame numbers
        sort_Detected_Sequences(all_Sequences)
        
        print("All Sequences sorted: ", all_Sequences)

        update_persons(all_Sequences) # Update the persons with the talking frames labels 

        create_processed_video = CreateProcessedVideo(self.selected_file, all_persons, all_Sequences)
        self.output_video_path = self.selected_file + "_modified.mp4"

        # Next Button (initially enabled)
        self.nextButton.config(state=NORMAL)  # Enable Next button after processing

        # Message after processing
        messagebox.showinfo("Finished", "The video transcription has been completed.", parent=self.MyWindow)
    
    def BtnNext_Clicked(self):
        self.MyWindow.destroy()
        self.open_second_gui(self.output_video_path, self.number_people)

    def open_second_gui(self, output_video_path, num_people):
        second_window = Tk()
        second_window.title("Video Analysis Result")
        second_window.geometry('1200x800')

        left_frame = Frame(second_window, width=840, height=800)
        left_frame.pack(side=LEFT, padx=10, pady=10)
        left_frame.pack_propagate(False)

        video_label = Label(left_frame, text="Video Player", font=("Arial Bold", 20))
        video_label.pack(pady=20)
        
        # Video display
        self.video_canvas = Canvas(left_frame, width=720, height=560)
        self.video_canvas.pack()

        self.cap = cv2.VideoCapture(output_video_path)

        control_frame = Frame(left_frame)
        control_frame.pack(pady=10)

        play_button = Button(control_frame, text="Play", command=self.play_video)
        play_button.grid(row=0, column=0, padx=5)

        pause_button = Button(control_frame, text="Pause", command=self.pause_video)
        pause_button.grid(row=0, column=1, padx=5)

        replay_button = Button(control_frame, text="Replay", command=self.replay_video)
        replay_button.grid(row=0, column=2, padx=5)

        right_frame = Frame(second_window, width=360, height=800)
        right_frame.pack(side=RIGHT, padx=10, pady=10)
        right_frame.pack_propagate(False)

        right_top_frame = Frame(right_frame, width=360, height=320)
        right_top_frame.pack(pady=(0, 10))
        right_top_frame.pack_propagate(False)

        right_middle_frame = Frame(right_frame, width=360, height=320)
        right_middle_frame.pack(pady=(0, 10))
        right_middle_frame.pack_propagate(False)

        # Placeholder images
        self.speaker_labels = []
        self.speaker_canvases = []

        for i in range(num_people):
            speaker_label = Label(right_top_frame if i == 0 else right_middle_frame, text=f"Speaker {i+1}", font=("Arial Bold", 15))
            speaker_label.pack(pady=10)
            self.speaker_labels.append(speaker_label)

            speaker_canvas = Canvas(right_top_frame if i == 0 else right_middle_frame, width=320, height=240)
            speaker_canvas.pack()
            self.speaker_canvases.append(speaker_canvas)

        self.load_placeholder_images(num_people)
        self.update_frame()

        second_window.mainloop()

    def load_placeholder_images(self, num_people):
        self.speaker_images = []

        for i in range(num_people):
            img = Image.open(f"placeholder_{i+1}.png")  # Placeholder image paths
            img_gray = ImageOps.grayscale(img)
            img_tk = ImageTk.PhotoImage(img_gray)
            self.speaker_images.append(img_tk)

        for i, canvas in enumerate(self.speaker_canvases):
            canvas.create_image(0, 0, anchor=NW, image=self.speaker_images[i])

    def play_video(self):
        global playing
        playing = True
        self.video_thread = Thread(target=self.update_frame)
        self.video_thread.start()
        self.audio_clip = VideoFileClip(self.output_video_path).audio
        self.audio_thread = Thread(target=self.audio_clip.preview)
        self.audio_thread.start()

    def pause_video(self):
        global playing
        playing = False

    def replay_video(self):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.play_video()

    def update_frame(self):
        global playing
        while playing:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (720, 560))  # Resize the frame to match the canvas size
                img = Image.fromarray(frame)
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_canvas.create_image(0, 0, anchor=NW, image=imgtk)
                self.video_canvas.image = imgtk

                # Update speaker images based on detected frames
                self.update_speakers(frame)
            else:
                break
            time.sleep(1/30)  # Control the frame rate

    def update_speakers(self, frame):
        # Logic to update speakers based on frame
        for person in all_persons:
            if person.get_is_talking() == 2:
                if self.cap.get(cv2.CAP_PROP_POS_FRAMES) == person.get_frame_number():
                    img = Image.open(f"speaker_{person.get_label()}.png")
                    img_tk = ImageTk.PhotoImage(img)
                    self.speaker_canvases[person.get_label()].create_image(0, 0, anchor=NW, image=img_tk)
                    self.speaker_canvases[person.get_label()].image = img_tk
                    
# ---------------------------------------------------------------------------------------------------------------------------

def main():
    total_time = time.monotonic()

    gui = GUI()

    print("Number of Face Vectors: ", len(all_persons))
    print("Total Time taken: ", time.monotonic() - total_time)
    
if __name__ == "__main__":
    main()