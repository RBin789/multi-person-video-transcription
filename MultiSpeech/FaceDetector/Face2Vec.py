import sys
import os
import glob
import time
import cv2
import dlib
import numpy as np
from sklearn.preprocessing import MaxAbsScaler

"""
This code is used to detect faces in an image and extract the facial keypoints and turn them into vectors
This code is based on the code from the following link: https://github.com/N2ITN/Face2Vec/tree/master specifically the file identify.py file
"""

class Face2Vec:
    def __init__(self, frame, current_frame_num, face_detector, landmark_predictor, yolo_model):
        self.img = frame
        self.current_frame_num = current_frame_num
        self.face_detector = face_detector
        self.landmark_predictor = landmark_predictor
        self.yolo_model = yolo_model
        
        self.bounding_boxs = []
        self.heads = []
        self.face_keypoints = []
        self.face_features = []
        self.lip_seperation = []
        self.enforced_face_frame_height = 480 # This is the enforced height of a face frame.
        
        self.detect_faces()
        self.detect_keypoints()
        self.convert_to_vectors()
        self.show_keypoints() # Display the keypoints on the faces.  Comment out if not needed
    
    def detect_faces(self):
        """Detects faces in the image and crops the image to the face bounding box."""
        
        faces = []
        results = self.yolo_model(self.img)
        
        # Extract the bounding boxes from the results
        for result in results:
            boxes = result.boxes.cpu().numpy()
            xyxys = boxes.xyxy
            for xyxy in xyxys:
                x1, y1, x2, y2 = xyxy

                faces.append((int(x1), int(y1), int(x2), int(y2)))
        
        for (x1, y1, x2, y2) in faces:
            self.crop_image(x1, y1, x2, y2)

    def crop_image(self, x1, y1, x2, y2):
        """Crops the image to the face bounding box."""
        
        # Grow the bounding box by 1/8 of the width and height
        growx = (x2 - x1)/8
        growy = (y2 - y1)/8

        # Grow the bounding box
        x1 = int(x1 - growx)
        x2 = int(x2 + growx)
        y1 = int(y1 - growy)
        y2 = int(y2 + growy)

        # Set negative values to 0
        x1 = max(0, x1)
        x2 = max(0, x2)
        y1 = max(0, y1)
        y2 = max(0, y2)

        self.heads.append(self.img[y1:y2, x1:x2])
        self.bounding_boxs.append((x1, y1, x2, y2))

    def calculate_Lip_Seperation(self, keypoints):
        """Calculates the distance between the top and bottom lip"""

        point50 = keypoints[50] # To understand the points see the following link: https://github.com/sachinsdate/lip-movement-net/tree/master
        point58 = keypoints[58]
        difference1 = abs(point50[1] - point58[1])

        point51 = keypoints[51]
        point57 = keypoints[57]
        difference2 = abs(point51[1] - point57[1])

        point52 = keypoints[52]
        point56 = keypoints[56]
        difference3 = abs(point52[1] - point56[1])
    
        outer_avg_distance = (difference1 + difference2 + difference3) / 3

        point61 = keypoints[61] # To understand the points see the following link: https://github.com/sachinsdate/lip-movement-net/tree/master
        point67 = keypoints[67]
        difference4 = abs(point61[1] - point67[1])

        point62 = keypoints[62]
        point66 = keypoints[66]
        difference5 = abs(point62[1] - point66[1])

        point63 = keypoints[63]
        point65 = keypoints[65]
        difference6 = abs(point63[1] - point65[1])

        inner_avg_distance = (difference4 + difference5 + difference6) / 3
        
        return (outer_avg_distance, inner_avg_distance)

    def detect_keypoints(self):
        """For self.face_keypoints data is stored in the form [(68 points), (68 points), (68 points)] every entry is a face"""  

        for i, image in enumerate(self.heads):
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self.face_detector(gray, 1)
            
            # If no faces are found, append an empty list to the face_keypoints and lip_seperation lists
            if len(faces) == 0:
                self.face_keypoints.append([])
                self.lip_seperation.append([])
        
            # If faces are found, extract the facial keypoints and lip seperation
            for face in faces:
                landmarks_for_face = self.landmark_predictor(gray, face)
                landmarks = []
                for j in range(0, landmarks_for_face.num_parts):
                    x = landmarks_for_face.part(j).x
                    y = landmarks_for_face.part(j).y
                    landmarks.append((x, y))

                min_point = min(landmarks, key=lambda pair: pair[1]) # This is to get the top most point of the face
                max_point = max(landmarks, key=lambda pair: pair[1]) # This is to get the bottom most point of the face
                height_of_face = max_point[1] - min_point[1]
                face_height_scaling = self.enforced_face_frame_height/height_of_face  # This is to ensure that the lip seperation is the same for all face frame sizes
                
                lip_sepration = (self.calculate_Lip_Seperation(landmarks)[0] * face_height_scaling, self.calculate_Lip_Seperation(landmarks)[1] * face_height_scaling) # Scale the lip seperation based on the face height
                
                print("Face Height: ", height_of_face)
                print ("Orig Lip Seperation: ", self.calculate_Lip_Seperation(landmarks))
                print ("New Lip Seperation: ", lip_sepration)
                
                self.lip_seperation.append(lip_sepration)
                self.face_keypoints.append(landmarks)
                break  # This is to ensure that the landmark predictor only gets the first face even if it finds multiple (just take the first)    

    def show_keypoints(self):
        """Displays the frame with the facial keypoints plotted."""

        cv2.putText(self.img, "Frame: " + str(self.current_frame_num), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA) # Display the frame number

        # Loop through each face and plot the bounding box and facial keypoints
        for head in range(len(self.heads)):
            x1, y1, x2, y2 = self.bounding_boxs[head]
            cv2.rectangle(self.img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
            if self.face_keypoints[head] != []:
                for point in range(len(self.face_keypoints[head])):
                    x, y = self.face_keypoints[head][point] # Get the x, y coords of the point 
                    color = (0, 0, 255) if point in [61, 62, 63, 65, 66, 67] else (0, 255, 255) # Set color the lips to red and the rest to yellow
                    cv2.circle(self.img, (x+x1, y+y1), 1, color, 2) # Plot the point on the image

            cv2.imshow("Image with Landmarks", self.img) 
        cv2.waitKey(1)

    def euc_center(self, keypoints):
        """Calculates the Euclidean distance between the center point of the face and each keypoint."""
        
        centerPoint = keypoints[30] # The center point of the face
        euclidian = [self.distance_2D(centerPoint, point) for point in keypoints] # Calculate the Euclidean distance between the center point and each keypoint
        euclidian = np.array(euclidian).reshape(-1, 1)
        norm = MaxAbsScaler().fit_transform(euclidian) 
        self.euclidian = norm

    def euc_xy(self, keypoints):
        """ This function calculates the separate X and Y components of the Euclidean distance between the center point and each keypoint """
        
        euclidian1D = []
        centerPoint = keypoints[30] # The center point of the face

        [euclidian1D.append(self.distance_1D(centerPoint, point)) for point in keypoints] # Calculate the X and Y distances between the center point and each keypoint

        x, y = [x for x in zip(*euclidian1D)] # Unzip the list of tuples into two separate lists

        x = np.array(x).reshape(-1, 1)
        y = np.array(y).reshape(-1, 1)
        x = MaxAbsScaler().fit_transform(x)
        y = MaxAbsScaler().fit_transform(y)

        self.euclidianX = x
        self.euclidianY = y

    def calculate_angles(self, keypoints):
        """Calculates the angle between the center point and each keypoint."""
        
        centerPoint = keypoints[30] # The center point of the face
        angles = [self.calculate_angle(centerPoint, point) for point in keypoints]
        angles = np.array(angles).reshape(-1, 1)
        norm = MaxAbsScaler().fit_transform(angles)
        self.angle_values = norm

    def normalized_coordinates(self, keypoints, bounding_box):
        """Calculates the normalized coordinates of each keypoint relative to the bounding box."""
        
        x_min, y_min, x_max, y_max = bounding_box 
        width = x_max - x_min
        height = y_max - y_min
        norm_coords = [((x - x_min) / width, (y - y_min) / height) for (x, y) in keypoints]
        norm_coords = np.array(norm_coords).reshape(-1, 2)
        self.norm_coords = norm_coords

    def aspect_ratio(self, keypoints):
        """Calculates the aspect ratio of the bounding box enclosing the face keypoints."""
        
        x_coords = [point[0] for point in keypoints]
        y_coords = [point[1] for point in keypoints]
        width = max(x_coords) - min(x_coords)
        height = max(y_coords) - min(y_coords)
        self.aspect_ratio_values = width / height

    def all_euclidian(self, keypoints, bounding_box):
        """Calculates all the necessary Euclidean distance information and stores it in a tensor """
        
        x_offset = np.full(68, bounding_box[0]).reshape(-1, 1)

        self.euc_xy(keypoints)
        self.euc_center(keypoints)
        self.calculate_angles(keypoints)
        self.normalized_coordinates(keypoints, bounding_box)
        self.aspect_ratio(keypoints)
        aspect_ratios = np.full(68, self.aspect_ratio_values).reshape(-1, 1)

        # tensor = np.rot90(np.hstack((x_offset, self.euclidianX, self.euclidianY, self.euclidian, self.angle_values, self.norm_coords, aspect_ratios)))
        tensor = np.rot90(np.hstack((self.euclidianX, self.euclidianY, self.euclidian, self.angle_values, self.norm_coords, aspect_ratios)))
        return tensor

    def calculate_angle(self, center, point):
        """Calculates the angle between the horizontal axis and the line connecting the center point and a keypoint."""
        
        x1, y1 = center
        x2, y2 = point
        angle = np.arctan2(y2 - y1, x2 - x1)
        
        return angle

    def distance_1D(self, a, b):
        """ This function calculates the X and Y distances between two points (a and b) """
        
        x1, y1 = a
        x2, y2 = b
        x = x1 - x2
        y = y1 - y2
        
        return x, y
    
    def distance_2D(self, a, b):
        """ This function calculates the Euclidean distance between two points (a and b) """
        
        x1, y1 = a
        x2, y2 = b
        a = np.array((x1, y1))
        b = np.array((x2, y2))
        dist = np.linalg.norm(a - b)
        
        return dist
    
    def convert_to_vectors(self):
        """Convert each face to a vector and create a list object that holds all information about the face"""

        for j, keypoints in enumerate(self.face_keypoints):
            if len(keypoints) != 0: # This is to check for head but no face
                tensor = self.all_euclidian(keypoints, self.bounding_boxs[j])
                tensor = tensor.flatten()
                try:
                    self.face_features.append((tensor, self.current_frame_num, self.lip_seperation[j], self.bounding_boxs[j], self.face_keypoints[j]))
                except:
                    print("Error in Face2Vec.py! Check the convert_to_vectors function.")

    def get_face_features(self):
        """Get the face features."""

        return self.face_features