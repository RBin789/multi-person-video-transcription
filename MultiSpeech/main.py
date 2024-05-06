import sys
import time
import cv2
import dlib
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
sys.path.insert(0, 'MultiSpeech\FaceDetector')
import tensorflow as tf
from tensorflow import keras
from FaceDetector.Face2Vec import *
from FaceDetector.Sequence_Generation import *
from FaceDetector.GUI import *
from FaceDetector.Lip_Detection import *
from FaceDetector.audioToText import *

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

def peform_kmeans_clustering(all_Face_Vectors, num_people):
    # Extract vectors from the list
    vectors = [face_vector[0].flatten() for face_vector in all_Face_Vectors]

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
    for item, label in zip(all_Face_Vectors, cluster_labels):
        clustered_data.append([item[0], item[1], item[2], label])

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
        person_sequences = sequence_generation(cluster_data) # all of one persons sequences
        run_lip_detection(person_sequences, model)


def sequence_generation(face_vectors):
    # Generate sequences
    sequence_generation = Sequence_Generation(face_vectors)
    person_sequences = sequence_generation.get_person_sequences()
    return person_sequences

def run_lip_detection(person_sequences, model):
    for i, sequence in enumerate(person_sequences): # Loops though every sequence of a person
        lip_detection = Lip_Detection(sequence, model)

def main():
    total_time = time.monotonic()

    model = tf.keras.models.load_model("MultiSpeech\FaceDetector\models\model.keras")
    # video_path = "MultiSpeech/FaceDetector/videos/One_Plus_One_1s_clip.mp4"
    video_path = "MultiSpeech/FaceDetector/videos/One_Plus_One_1s_clip.mp4"

    # Run the GUI
    # run_gui()

    # Process the video
    process_video(video_path)

    # Process Audio
    audiototext = audioToText(video_path)

    # K-means clustering on face vectors
    num_people = int(input("Enter the number of people in the video: "))
    clustered_data = peform_kmeans_clustering(all_Face_Vectors, num_people)
    clustered_by_label = split_data_by_cluster(clustered_data)

    # Generate sequences for each person and run lip detection
    process_clustered_data(clustered_by_label, model)


    print("Number of Face Vectors: ", len(all_Face_Vectors))
    # print(all_Face_Vectors[0])
    print("Total Time taken: ", time.monotonic() - total_time)
    
    
    

if __name__ == "__main__":
    main()
