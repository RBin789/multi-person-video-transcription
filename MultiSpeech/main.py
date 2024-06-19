import sys
import time
import cv2
import dlib
from tkinter import *
import tkinter as tk
from tkinter import messagebox
from tkinter import filedialog
from ultralytics import YOLO
import tensorflow as tf
sys.path.insert(0, 'MultiSpeech\FaceDetector')
from FaceDetector.Face2Vec import *
from FaceDetector.Person import *
from FaceDetector.FaceSequenceProcessor import *
from FaceDetector.Face import *
from FaceDetector.CreateTranscript import *


all_faces = []
number_of_people = None
annotated_video_path = None
transcript_path = None
selected_file = None  # Initialize variable to store video path
start_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
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
    return current_frame_num

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
        self.MyWindow.geometry('800x450')

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

        self.view_annotated_video_button = Button(self.bottom_frame, text="View Annotated Video", state=tk.DISABLED, command=self.BtnViewAnnotatedVideo_Clicked, font=("Arial", 14), bg='#e0e0e0', fg='#999999')
        self.view_annotated_video_button.grid(row=1, column=0, padx=10, pady=5)

        self.view_transcript_button = Button(self.bottom_frame, text="View Transcript", state=tk.DISABLED, command=self.BtnViewTranscript_Clicked, font=("Arial", 14), bg='#e0e0e0', fg='#999999')
        self.view_transcript_button.grid(row=1, column=1, padx=10, pady=5)

        self.bottom_frame.grid_columnconfigure(0, weight=1)
        self.bottom_frame.grid_columnconfigure(1, weight=1)

        self.MyWindow.mainloop()

    def BtnOpen_Clicked(self):
        filetypes = [("Video files", "*.mp4")]
        file_path = filedialog.askopenfilename(filetypes=filetypes)
        print(file_path)

        if file_path:
            self.selected_file = file_path
            self.start_button.config(state=NORMAL, bg='#cccccc', fg='#333333')

    def BtnStart_Clicked(self):

        # sys.stdout = DualLogger(f"{self.selected_file[:-4]}_log_{start_time}.txt", 'w')

        self.number_people = int(self.number_entry.get())
        print(f"Processing video with num people being: {self.number_people}")
        
        global number_of_people 
        number_of_people = self.number_people

        num_of_frames = process_video(self.selected_file)
        
        face_sequence_processor = FaceSequenceProcessor(all_faces, self.number_people, lip_detection_model, self.selected_file, num_of_frames, start_time)
        
        global annotated_video_path 
        annotated_video_path = face_sequence_processor.get_annotated_video_path()
        persons = face_sequence_processor.get_persons()
        
        global transcript_path
        create_transcript = CreateTranscript(self.selected_file, persons, start_time)
        transcript_path = create_transcript.get_transcript_path()
        
        self.view_annotated_video_button.config(state=NORMAL, bg='#cccccc', fg='#333333')
        self.view_transcript_button.config(state=NORMAL, bg='#cccccc', fg='#333333')

        messagebox.showinfo("Finished", "The annotated video and transcription have been made, they have been saved in the same directorty as the original video.", parent=self.MyWindow)

    def BtnViewAnnotatedVideo_Clicked(self):
        global annotated_video_path
        os.startfile(annotated_video_path)
    
    def BtnViewTranscript_Clicked(self):
        global transcript_path
        os.startfile(transcript_path)
                    
# ---------------------------------------------------------------------------------------------------------------------------

class DualLogger:
    def __init__(self, filepath, mode='a'):
        self.terminal = sys.__stdout__
        self.log = open(filepath, mode)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # Flush the I/O buffer to both the terminal and the file.
        self.terminal.flush()
        self.log.flush()


def main():

    total_time = time.monotonic()

    gui = GUI()

    print("Number of Face Vectors: ", len(all_faces))
    print("Total Time taken: ", time.monotonic() - total_time)
    
if __name__ == "__main__":
    main()
