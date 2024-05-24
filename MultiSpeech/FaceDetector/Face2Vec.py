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
                # cv2.rectangle(self.img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                # cv2.imshow('image', np.array(self.img))
                # cv2.waitKey(0)

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
                # print("Faces found 2: " + str(len(faces)))
                self.face_keypoints.append([])
                self.lip_seperation.append([])
                # print(len(self.face_keypoints[-1]))
                # self.heads.remove(image)
        
            for face in faces:
                landmarks_for_face = self.landmark_predictor(gray, face)
                landmarks = []
                for i in range(0, landmarks_for_face.num_parts):
                    x = landmarks_for_face.part(i).x
                    y = landmarks_for_face.part(i).y
                    landmarks.append((x, y))
                self.lip_seperation.append(self.calculate_Lip_Seperation(landmarks))
                self.face_keypoints.append(landmarks)      

    def show_keypoints(self):
        """ Shows the first face found with the keypoints drawn on it. """

        for head in range(len(self.heads)):
            # print(self.face_keypoints[head])
        
            if self.face_keypoints[head] != []:
                # x, y = self.face_keypoints[imgnum][61]                            # If you want to see mouth points use this code
                # cv2.circle(self.heads[0], (x, y), 1, (0, 0, 255), 2)
                # x, y = self.face_keypoints[imgnum][67]
                # cv2.circle(self.heads[0], (x, y), 1, (0, 0, 255), 2)
                # x, y = self.face_keypoints[imgnum][62]
                # cv2.circle(self.heads[0], (x, y), 1, (0, 0, 255), 2)
                # x, y = self.face_keypoints[imgnum][66]
                # cv2.circle(self.heads[0], (x, y), 1, (0, 0, 255), 2)
                # x, y = self.face_keypoints[imgnum][63]
                # cv2.circle(self.heads[0], (x, y), 1, (0, 0, 255), 2)
                # x, y = self.face_keypoints[imgnum][65]
                # cv2.circle(self.heads[0], (x, y), 1, (0, 0, 255), 2)
                for point in range(len(self.face_keypoints[head])):               # If you want to see all points use this code
                    x, y = self.face_keypoints[head][point]
                    cv2.circle(self.heads[head], (x, y), 1, (0, 0, 255), 2)

                cv2.imshow("Image with Landmarks", self.heads[head])
                cv2.waitKey(1)
                # cv2.destroyAllWindows()


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

    def all_euclidian(self, keypoints):
        """Calculates all the necessary Euclidean distance information and stores it in a tensor """
        self.euc_xy(keypoints)
        self.euc_center(keypoints)
        tensor = np.rot90(np.hstack((self.euclidianX, self.euclidianY, self.euclidian)))

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
                tensor = self.all_euclidian(keypoints)
                try:
                    self.face_features.append((tensor, self.current_frame_num, self.lip_seperation[j], self.bounding_boxs[j], self.face_keypoints[j]))
                except:
                    self.face_features.append((tensor, self.current_frame_num, [], [], self.lip_seperation[j]))

    def get_face_features(self):
        return self.face_features    

    def print_vectors(self):
        print("Faces found: " + str(len(self.face_keypoints)))
        print("Length of vector " + str(len(self.face_features)))
        # print(self.face_features)