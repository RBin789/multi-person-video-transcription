import sys
import time
import cv2
sys.path.insert(0, 'MultiSpeech\FaceDetector')
from FaceDetector.Face2Vec import *
from FaceDetector.Sequence_Generation import *
from FaceDetector.GUI import *

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
        all_Face_Vectors.extend(face_vectors)
        success, frame = video.read()
        current_frame_num += 1
        print("Frame Processed")

    # Release the video file
    video.release()    

def process_image():
    face_detector = dlib.get_frontal_face_detector()
    landmark_predictor = dlib.shape_predictor("MultiSpeech\FaceDetector\shape_predictor_68_face_landmarks.dat")
    # Read in the image
    image_path = "MultiSpeech\FaceDetector\images\One+One_frame.png"
    image = cv2.imread(image_path)
    face2vec = Face2Vec(image, 1, face_detector, landmark_predictor)
    face_vectors = face2vec.get_face_vectors()
    all_Face_Vectors.extend(face_vectors)


def main():
    total_time = time.monotonic()
    video_path = r"C:\Users\dexte\Github-Repositories\multi-person-video-transcription\MultiSpeech\FaceDetector\videos\One_Plus_One_1s_clip.mp4"
    process_video(video_path)
    # process_image()
    # run_gui() # Run the GUI from GUI.py


    print("Number of Face Vectors: ", len(all_Face_Vectors))
    print("Total Time taken: ", time.monotonic() - total_time)
    
    # print(all_Face_Vectors)
    

if __name__ == "__main__":
    main()


