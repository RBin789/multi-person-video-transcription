import sys
import time
import cv2
sys.path.insert(0, 'MultiSpeech\FaceDetector')
from FaceDetector.Face2Vec import *

all_Face_Vectors = []

def process_video(video_path):
    # Open the video file
    video = cv2.VideoCapture(video_path)

    # Check if the video file was opened successfully
    if not video.isOpened():
        print("Error opening video file")
        exit()

    success, frame = video.read() # Read the first frame

    while success:
        face2vec = Face2Vec(frame)
        face_vectors = face2vec.get_face_vectors()
        all_Face_Vectors.extend(face_vectors)
        success, frame = video.read()

    # Release the video file
    video.release()    

def process_image():
    # Read in the image
    image_path = "MultiSpeech\FaceDetector\images\One+One_frame.png"
    face2vec = Face2Vec(image_path)
    face_vectors = face2vec.get_face_vectors()
    all_Face_Vectors.extend(face_vectors)

def main():
    monotonic_time = time.monotonic()
    # process_video()
    process_image()
    print("Time taken: ", time.monotonic() - monotonic_time)
    #print(all_Face_Vectors)
    

if __name__ == "__main__":
    main()


