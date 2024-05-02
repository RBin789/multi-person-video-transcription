import sys
import time
import cv2
import dlib
sys.path.insert(0, 'MultiSpeech\FaceDetector')
import tensorflow as tf
from tensorflow import keras
from FaceDetector.Face2Vec import *
from FaceDetector.Sequence_Generation import *
from FaceDetector.GUI import *
from FaceDetector.Lip_Detection import *

all_Face_Vectors = []

def process_video(video_path):
    # Open the video file
    video = cv2.VideoCapture(video_path)

    # Check if the video file was opened successfully
    if not video.isOpened():
        print(f"Video file does not exist: {video_path}")
        exit()

    face_detector = dlib.get_frontal_face_detector()
    landmark_predictor = dlib.shape_predictor("MultiSpeech\FaceDetector\shape_predictor_68_face_landmarks.dat")
    current_frame_num = 1
    success, frame = video.read() # Read the first frame

    while success:
        face2vec = Face2Vec(frame, current_frame_num, face_detector, landmark_predictor)
        face_vectors = face2vec.get_face_vectors()
        all_Face_Vectors.extend(face_vectors)  # Final Format of all_Face_Vectors: [[vectors], frame_num, lip_sep]
        success, frame = video.read()
        current_frame_num += 1
        print("Frame Processed")

    # Release the video file
    video.release()

def run_gui():
    app = GUI()

def sequence_generation(face_vectors):
    # Generate sequences
    sequence_generation = Sequence_Generation(face_vectors)
    person_sequences = sequence_generation.get_person_sequences()
    return person_sequences

def run_lip_detection(person_sequences, model):
    for i, sequence in enumerate(person_sequences):
        lip_detection = Lip_Detection(sequence, model)

def main():
    total_time = time.monotonic()

    model = tf.keras.models.load_model("MultiSpeech\FaceDetector\models\model.keras")
    # video_path = "MultiSpeech/FaceDetector/videos/One_Plus_One_1s_clip.mp4"
    video_path = "MultiSpeech/FaceDetector/videos/One_Plus_One_5s_clip.mp4"

    # Process the video
    process_video(video_path)

    # Run the GUI
    # run_gui()

    # Generate sequences
    person_sequences = sequence_generation(all_Face_Vectors)

    # Run the lip detection for each sequence of a person
    run_lip_detection(person_sequences, model)

    print("Number of Face Vectors: ", len(all_Face_Vectors))
    # print(all_Face_Vectors[0])
    print("Total Time taken: ", time.monotonic() - total_time)
    
    
    

if __name__ == "__main__":
    main()
