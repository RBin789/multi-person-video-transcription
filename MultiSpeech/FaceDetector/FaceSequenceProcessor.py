import sys
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
sys.path.insert(0, 'MultiSpeech\FaceDetector')
from FaceDetector.Sequence_Generation import *
from FaceDetector.Lip_Detection import *
from FaceDetector.Person import *
from FaceDetector.Face import *
from FaceDetector.CreateProcessedVideo import *

class FaceSequenceProcessor:
    
    def __init__(self, all_faces, num_people, lip_detection_model, selected_file):
        self.all_faces = all_faces
        self.num_people = num_people
        self.lip_detection_model = lip_detection_model
        self.selected_file = selected_file

        self.persons = []
        self.all_sequences = []

        self.run_functions(all_faces, num_people)

    def run_functions(self, all_faces, num_people):
        self.normalize_lip_seperation(all_faces)
        
        clustered_data = self.peform_kmeans_clustering(all_faces, num_people)
        clustered_by_label = self.split_data_by_cluster(clustered_data)
        self.process_clustered_data(clustered_by_label, self.lip_detection_model)
        self.sort_Detected_Sequences()

        self.update_faces()
        self.create_persons(num_people)
        self.print_sequences()
        self.create_annotated_video(all_faces, self.all_sequences)
    
    def normalize_lip_seperation(self, all_faces):
    
        max_lip_sep = 0
        frame_occurence = 0

        for face_num, face in enumerate(all_faces):
            if face.get_lip_seperation() > max_lip_sep:
                max_lip_sep = face.get_lip_seperation()
                frame_occurence = face.get_frame_number()
    
        print("Max lip sep: " + str(max_lip_sep))
        print("Frame occurence: " + str(frame_occurence))

        for face_num, face in enumerate(all_faces):
            face.set_lip_seperation(face.get_lip_seperation() / max_lip_sep)
                

    def peform_kmeans_clustering(self, all_faces, num_people):
        # Extract vectors from the list
        vectors = [face_vector.get_face_vector() for face_vector in all_faces]

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
        for item, label in zip(all_faces, cluster_labels):
            clustered_data.append([item.get_face_vector(), item.get_frame_number(), item.get_lip_seperation(), label])
            item.set_label(label)

        # Plotting the clusters
        # plt.scatter(vector_array[:, 0], vector_array[:, 1], c=cluster_labels, cmap='viridis')
        # centers = kmeans.cluster_centers_
        # plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
        # plt.show() 

        return clustered_data
    
    def split_data_by_cluster(self, clustered_data):
        clustered_by_label = {}
        for item in clustered_data:
            cluster_label = item[3]  # Access the cluster label from the 4th element
            if cluster_label not in clustered_by_label:
                clustered_by_label[cluster_label] = []
            clustered_by_label[cluster_label].append(item)
        return clustered_by_label # A dictionary where keys are cluster labels and values are lists of data points belonging to that cluster.

    def process_clustered_data(self, clustered_by_label, lip_detection_model):
        for cluster_label, cluster_data in clustered_by_label.items():
            sequence_generation = Sequence_Generation(cluster_label, cluster_data) # all of one persons sequences
            person_sequences = sequence_generation.get_person_sequences()
            
            self.run_lip_detection(person_sequences, cluster_label, lip_detection_model)

    def run_lip_detection(self, person_sequences, cluster_label, lip_detection_model):
        for i, sequence in enumerate(person_sequences): # Loops though every sequence of a person
            if (len(sequence) == 0):
                continue
            lip_detection = Lip_Detection(sequence, cluster_label, lip_detection_model)
            self.all_sequences.append(lip_detection.get_sequence_and_prediction())

    def sort_Detected_Sequences(self):
        self.all_sequences.sort(key=lambda x: x[1])  # Sort by frame number
        
    def update_faces(self):
        for sequence in self.all_sequences: # Loops though every sequence
                for frame in sequence[1]: # Loops though every frame number in the sequence
                    for face in self.all_faces: # Loops though every person
                        if (face.get_label() == sequence[0]) and (face.get_frame_number() == frame): # If the person label matches the sequence label
                            if sequence[2] == 1:# else:
                                face.set_is_talking(2)  # person is talking
                            if sequence[2] == 0:# else:
                                face.set_is_talking(1)  # person not talking
    
    def create_persons(self, num_people):
        for i in range(num_people):
            person = Person(i)
            for face in self.all_faces:
                if face.get_label() == i:
                    person.add_face(face)
            self.persons.append(person)

    def print_sequences(self):
        for sequence in self.all_sequences:
            print(sequence)

    def create_annotated_video(self, all_faces, all_sequences):
        create_processed_video = CreateProcessedVideo(self.selected_file, all_faces, all_sequences)

    def get_persons(self):
        return self.persons