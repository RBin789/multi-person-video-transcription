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
    
    def __init__(self, all_faces, num_people, lip_detection_model, selected_file, num_of_frames, current_time):
        self.all_faces = all_faces
        self.num_people = num_people
        self.lip_detection_model = lip_detection_model
        self.selected_file = selected_file
        self.num_of_frames = num_of_frames
        self.current_time = current_time
        self.annotated_video_path = None

        self.persons = []
        self.all_sequences = []
        self.max_lip_seperations_outer = []
        self.max_lip_seperations_inner = []

        self.run_functions(self.all_faces)

    def run_functions(self, all_faces):
        self.find_max_lip_sep_per_frame(all_faces, self.num_of_frames)
        
        clustered_data = self.peform_kmeans_clustering(all_faces, self.num_people)
        clustered_by_label = self.split_data_by_cluster(clustered_data)
        self.process_clustered_data(clustered_by_label, self.lip_detection_model, self.max_lip_seperations_outer, self.max_lip_seperations_inner)
        self.sort_Detected_Sequences()

        self.update_faces()
        self.create_persons(self.num_people)
        self.print_sequences()
        self.create_annotated_video(all_faces, self.all_sequences, self.current_time)

    def find_max_lip_sep_per_frame(self, all_faces, num_of_frames):
        """Create a List of the maximum lip seperation for each frame in the video."""

        for frame in range(1, num_of_frames + 1): 
            max_lip_sep = 0
            for face_num, face in enumerate(all_faces):
                if face.get_frame_number() == frame:
                    if face.get_lip_seperation()[0] > max_lip_sep:
                        max_lip_sep = face.get_lip_seperation()[0]
            self.max_lip_seperations_outer.append(max_lip_sep)

        for frame in range(1, num_of_frames + 1): 
            max_lip_sep = 0
            for face_num, face in enumerate(all_faces):
                if face.get_frame_number() == frame:
                    if face.get_lip_seperation()[1] > max_lip_sep:
                        max_lip_sep = face.get_lip_seperation()[1]
            self.max_lip_seperations_inner.append(max_lip_sep)

    def peform_kmeans_clustering(self, all_faces, num_people):
        """Perform KMeans clustering on the face vectors & assign labels to the faces."""
        
        vectors = [face_vector.get_face_vector() for face_vector in all_faces] # Extract vectors from the list

        vector_array = np.array(vectors)
        kmeans = KMeans(n_clusters=num_people) # Create the KMeans model
        kmeans.fit(vector_array) # Fit the model to the data
        cluster_labels = kmeans.labels_ + 1 # Get the cluster labels for each vector

        clustered_data = []

        # Assign labels to the faces
        for item, label in zip(all_faces, cluster_labels):
            item.set_label(label)
        
        labels_and_avg_x_coord = []

        # Create a list of tuples containing the label and the average x coordinate of the faces with that label
        for label in range(1, num_people + 1):
            num_faces = 0
            avg_x_coord = 0
            for face in all_faces:
                if face.get_label() == label:
                    avg_x_coord += face.get_bounding_box()[0]
                    num_faces += 1
            avg_x_coord = avg_x_coord / num_faces
            labels_and_avg_x_coord.append((label, avg_x_coord))

        # Sort the list of tuples by the average x coordinate
        labels_and_avg_x_coord = sorted(labels_and_avg_x_coord, key=lambda x: x[1])
        
        # Create a mapping of old labels to new labels
        label_mapping = {label: index for index, (label, _) in enumerate(labels_and_avg_x_coord, start=1)}

        # Update the labels of the faces
        for face in all_faces:
            old_label = face.get_label()
            if old_label in label_mapping:
                new_label = label_mapping[old_label]
                face.set_label(new_label)

        # Create a list of the clustered data
        for item in all_faces:
            clustered_data.append([item.get_face_vector(), item.get_frame_number(), item.get_lip_seperation(), item.get_label()])               
        
        # Plotting the clusters
        # plt.scatter(vector_array[:, 0], vector_array[:, 1], c=cluster_labels, cmap='viridis')
        # centers = kmeans.cluster_centers_
        # plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
        # plt.show() 

        return clustered_data
    
    def split_data_by_cluster(self, clustered_data):
        """Split the clustered data by cluster label."""

        clustered_by_label = {}
        
        for item in clustered_data:
            cluster_label = item[3]  # Access the cluster label from the 4th element
            if cluster_label not in clustered_by_label:
                clustered_by_label[cluster_label] = [] 
            clustered_by_label[cluster_label].append(item)
        
        return clustered_by_label # A dictionary where keys are cluster labels and values are lists of data points belonging to that cluster.

    def process_clustered_data(self, clustered_by_label, lip_detection_model, max_lip_seperations_outer, max_lip_seperations_inner):
        """Process the clustered data by running the lip detection on each cluster."""
        
        for cluster_label, cluster_data in clustered_by_label.items():
            sequence_generation = Sequence_Generation(cluster_label, cluster_data) # Generate all of one persons sequences
            person_sequences = sequence_generation.get_person_sequences() # Get all of one persons sequences
            
            self.run_lip_detection(person_sequences, cluster_label, lip_detection_model, max_lip_seperations_outer, max_lip_seperations_inner) 

    def run_lip_detection(self, person_sequences, cluster_label, lip_detection_model, max_lip_seperations_outer, max_lip_seperations_inner):
        """Run the lip detection on each sequence of a person."""

        for i, sequence in enumerate(person_sequences):
            if (len(sequence) == 0):
                continue
            lip_detection_outer = Lip_Detection(sequence, cluster_label, lip_detection_model, max_lip_seperations_outer, lip_index=0)
            lip_detection_inner = Lip_Detection(sequence, cluster_label, lip_detection_model, max_lip_seperations_inner, lip_index=1)
            
            if (lip_detection_outer.get_sequence_and_prediction()[2] == 1):
                self.all_sequences.append(lip_detection_outer.get_sequence_and_prediction()) # Append the sequence and prediction to the list of all sequences
            else:
                self.all_sequences.append(lip_detection_inner.get_sequence_and_prediction())

    def sort_Detected_Sequences(self):
        """Sort the detected sequences by frame number."""
        
        self.all_sequences.sort(key=lambda x: x[1])  # Sort by frame number
        
    def update_faces(self):
        """Update the faces with the is_talking() attribute."""

        for sequence in self.all_sequences: # Loops though every sequence
                for frame in sequence[1]: # Loops though every frame number in the sequence
                    for face in self.all_faces: # Loops though every person
                        if (face.get_label() == sequence[0]) and (face.get_frame_number() == frame): # If the person label matches the sequence label
                            if sequence[2] == 1:
                                face.set_is_talking(2)  # Person is talking
                            if sequence[2] == 0:
                                face.set_is_talking(1)  # Person not talking
    
    def create_persons(self, num_people):
        """Create a Person object for each person in the video."""
        
        for i in range(1, num_people + 1):
            person = Person(i)
            for face in self.all_faces:
                if face.get_label() == i: 
                    person.add_face(face)
            self.persons.append(person)

    def print_sequences(self):
        """Print the detected sequences."""

        for sequence in self.all_sequences:
            print(sequence)

    def create_annotated_video(self, all_faces, all_sequences, current_time):
        """Create a video with the detected sequences annotated."""
        
        create_processed_video = CreateProcessedVideo(self.selected_file, all_faces, all_sequences, current_time)
        self.annotated_video_path = create_processed_video.annotated_video_path

    def get_persons(self):
        """Return the list of Person objects."""

        return self.persons
    
    def get_annotated_video_path(self):
        """Return the path to the annotated video."""

        return self.annotated_video_path
