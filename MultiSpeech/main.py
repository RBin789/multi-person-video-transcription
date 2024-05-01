import sys
import time
import cv2
from tensorflow.keras.models import load_model
sys.path.insert(0, 'MultiSpeech\FaceDetector')
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

def main():
    total_time = time.monotonic()
    video_path = r"C:\Users\dexte\Github-Repositories\multi-person-video-transcription\MultiSpeech\FaceDetector\videos\One_Plus_One_1s_clip.mp4"
    # video_path = r"C:\Users\dexte\Visual Studio Code Projects\DL & CNN Assingment 3\TrainModels\Video datasets\My Data\stock-footage-face-portrait-close-up-beautiful-little-girl-looking-at-the-camera-and-smiling-kindly-furniture.webm"
    process_video(video_path)
    # run_gui() # Run the GUI

    # Generate sequences
    # This should run in a for loop for each person in the video
    # sequence_generation = Sequence_Generation(all_Face_Vectors)
    # person_sequences = sequence_generation.get_person_sequences()

    # Run the lip detection for each sequence of a person
    # model = load_model(r"MultiSpeech\FaceDetector\models\1_64_True_True_0.0_lip_motion_net_model.h5")
    # for i, sequence in enumerate(person_sequences):
    #     lip_detection = Lip_Detection(sequence, model)

    print("Number of Face Vectors: ", len(all_Face_Vectors))
    # print(all_Face_Vectors[0])
    print("Total Time taken: ", time.monotonic() - total_time)
    
    
    

if __name__ == "__main__":
    main()


