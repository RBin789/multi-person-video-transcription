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
This code is heavily based on the code from the following link: https://github.com/N2ITN/Face2Vec/tree/master specifically the file identify.py file
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
        self.enforced_face_frame_height = 640 # This is the enforced height of a face frame.
        self.min_lip_threshold = 0.005 # This is the minimum lip threshold for the lip seperation
        

        self.detect_faces()
        self.detect_keypoints()
        self.convert_to_vectors()
        
        self.show_keypoints() # Display the keypoints on the faces.  Comment out if not needed
        # self.print_vectors() # Print the number of vectors.  Comment out if not needed
    

    def detect_faces(self):
        # Detects faces in the image
        # Might be changed to a more accurate model just simple for now
        faces = []

        results = self.yolo_model(self.img)
        
        # Display the results
        for result in results:
            boxes = result.boxes.cpu().numpy()
            # x1, y1, x2, y2 = box[0]
            xyxys = boxes.xyxy
            for xyxy in xyxys:
                x1, y1, x2, y2 = xyxy

                faces.append((int(x1), int(y1), int(x2), int(y2)))
        
        for (x1, y1, x2, y2) in faces:
            self.crop_image(x1, y1, x2, y2)

    def crop_image(self, x1, y1, x2, y2):
        # Convert face coordinates to rectangle corner points
        growx = (x2 - x1)/8
        growy = (y2 - y1)/8

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
        point61 = keypoints[61] # To understand the points see the following link: https://github.com/sachinsdate/lip-movement-net/tree/master
        point67 = keypoints[67]
        difference1 = abs(point61[1] - point67[1])

        point62 = keypoints[62]
        point66 = keypoints[66]
        difference2 = abs(point62[1] - point66[1])

        point63 = keypoints[63]
        point65 = keypoints[65]
        difference3 = abs(point63[1] - point65[1])

        avg_distance = (difference1 + difference2 + difference3) / 3
        return avg_distance


    def detect_keypoints(self):
        """For self.face_keypoints data is stored in the form [(68 points), (68 points), (68 points)] every entry is a face"""  

        for i, image in enumerate(self.heads):
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self.face_detector(gray, 1)
            
          
            if len(faces) == 0:
                self.face_keypoints.append([])
                self.lip_seperation.append([])
        
            for face in faces:
                landmarks_for_face = self.landmark_predictor(gray, face)
                landmarks = []
                for j in range(0, landmarks_for_face.num_parts):
                    x = landmarks_for_face.part(j).x
                    y = landmarks_for_face.part(j).y
                    landmarks.append((x, y))

                min_point = min(landmarks, key=lambda pair: pair[1])
                max_point = max(landmarks, key=lambda pair: pair[1])
                # bounding_box_height_scaling = self.enforced_face_frame_height/(self.bounding_boxs[i][3] - self.bounding_boxs[i][1]) # y2 - y1 # This is to ensure that the lip seperation is the same for all face frame sizes
                # bounding_box_height_scaling = self.enforced_face_frame_height/(max_point[1] - min_point[1]) # This is to ensure that the lip seperation is the same for all face frame sizes
                
                lip_sepration = self.calculate_Lip_Seperation(landmarks)
                height_of_face = max_point[1] - min_point[1]
                
                if (lip_sepration / height_of_face) < self.min_lip_threshold:
                    lip_sepration = 0
                
                print("Face Height: ", max_point[1] - min_point[1])
                print ("Orig Lip Seperation: ", self.calculate_Lip_Seperation(landmarks))
                print ("New Lip Seperation: ", lip_sepration)
                
                # lip_sepration = self.calculate_Lip_Seperation(landmarks)
                self.lip_seperation.append(lip_sepration)
                self.face_keypoints.append(landmarks)
                break  # This is to ensure that the landmark predictor only gets the first face even if it finds multiple (just take the first)    

    def show_keypoints(self):
        """ Shows the first face found with the keypoints drawn on it. """

        cv2.putText(self.img, "Frame: " + str(self.current_frame_num), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        for head in range(len(self.heads)):
            x1, y1, x2, y2 = self.bounding_boxs[head]
            cv2.rectangle(self.img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
            if self.face_keypoints[head] != []:
                for point in range(len(self.face_keypoints[head])):
                    x, y = self.face_keypoints[head][point]
                    color = (0, 0, 255) if point in [61, 62, 63, 65, 66, 67] else (0, 255, 255)
                    cv2.circle(self.img, (x+x1, y+y1), 1, color, 2)

            cv2.imshow("Image with Landmarks", self.img)
        cv2.waitKey(1)

# -----------------------------------------------------------------------------------------------

    def euc_center(self, keypoints):
        """ Calculates the Euclidean distance between the center point of the face (defined in geom) and each keypoint """
        centerPoint = keypoints[30]

        euclidian = [self.distance_2D(centerPoint, point) for point in keypoints]
        euclidian = np.array(euclidian).reshape(-1, 1)
        norm = MaxAbsScaler().fit_transform(euclidian)
        self.euclidian = norm # Definition of self.euclidian

    def euc_xy(self, keypoints):
        """ This function calculates the separate X and Y components of the Euclidean distance between the center point and each keypoint """
        euclidian1D = []
        centerPoint = keypoints[30]

        [euclidian1D.append(self.distance_1D(centerPoint, point)) for point in keypoints]

        x, y = [x for x in zip(*euclidian1D)]

        x = np.array(x).reshape(-1, 1)
        y = np.array(y).reshape(-1, 1)
        x = MaxAbsScaler().fit_transform(x)
        y = MaxAbsScaler().fit_transform(y)

        self.euclidianX = x
        self.euclidianY = y

    def all_euclidian(self, keypoints, x_offset):
        """Calculates all the necessary Euclidean distance information and stores it in a tensor """
        xoffset = np.full(68, x_offset).reshape(-1, 1)

        self.euc_xy(keypoints)
        self.euc_center(keypoints)
        tensor = np.rot90(np.hstack((xoffset, self.euclidianX, self.euclidianY, self.euclidian)))
        # print(tensor)
        # print(tensor.shape)
        return tensor

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
        # for i in range(len(self.face_keypoints)):
            # print(self.face_keypoints[i])
            
        for j, keypoints in enumerate(self.face_keypoints):
            if len(keypoints) != 0: # This is to check for head but no face
                tensor = self.all_euclidian(keypoints, self.bounding_boxs[j][0])
                tensor = tensor.flatten()
                try:
                    self.face_features.append((tensor, self.current_frame_num, self.lip_seperation[j], self.bounding_boxs[j], self.face_keypoints[j]))
                except:
                    print("Error in Face2Vec.py! Check the convert_to_vectors function.")

    def get_face_features(self):
        return self.face_features    

    def print_vectors(self):
        print("Faces found: " + str(len(self.face_keypoints)))
        print("Length of vector " + str(len(self.face_features)))
        # print(self.face_features)