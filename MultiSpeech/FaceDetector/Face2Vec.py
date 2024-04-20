import sys
import os
import glob
import cv2
import dlib
import numpy as np

class Face2Vec:
    def __init__(self, image):
        self.img = cv2.imread(image)
        self.processed_image = None
        self.cropped_faces = []
        self.face_keypoints = []
        self.face_vectors = []

        self.detect_faces()
        self.detect_keypoints()
        self.show_keypoints()
        # self.convert_to_vector()
    

    def detect_faces(self):
        # Detects faces in the image
        # Might be changed to a more accurate model just simple for now
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        for (x, y, w, h) in faces:
            self.crop_image(x, y, w, h)


    def crop_image(self, x, y, w, h):
        # Convert face coordinates to rectangle corner points
        x1, y1 = x, y
        x2, y2 = x + w, y + h
        grow = h / 7

        x1 = int(x1 - grow)
        x2 = int(x2 + grow)
        y1 = int(y1 - grow)
        y2 = int(y2 + grow)

        # Set negative values to 0
        x1 = max(0, x1)
        x2 = max(0, x2)
        y1 = max(0, y1)
        y2 = max(0, y2)

        self.cropped_faces.append(self.img[y1:y2, x1:x2])

    def detect_keypoints(self):
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor("MultiSpeech\FaceDetector\shape_predictor_68_face_landmarks.dat")
        for face in self.cropped_faces:
            gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            rects = detector(gray, 0)
            for rect in rects:
                shape = predictor(gray, rect)
                self.face_keypoints.append(shape)


    def show_keypoints(self):
        print("Faces found: " + str(len(self.cropped_faces)))

        # Create a blank canvas to display all the faces
        canvas = np.zeros((500, 500, 3), dtype=np.uint8)

        for idx, face in enumerate(self.cropped_faces):
            for shape in self.face_keypoints:
                for i in range(68):
                    x = shape.part(i).x
                    y = shape.part(i).y
                    cv2.circle(face, (x, y), 2, (0, 255, 0), -1)
            
            # Calculate the position to display the face on the canvas
            row = idx // 5
            col = idx % 5
            x_start = col * 250
            y_start = row * 250
            x_end = x_start + 250
            y_end = y_start + 250

            # Resize the face to fit in the canvas
            resized_face = cv2.resize(face, (250, 250))

            # Place the resized face on the canvas
            canvas[y_start:y_end, x_start:x_end] = resized_face

        # Display the canvas with all the faces
        cv2.imshow('Faces', canvas)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# -----------------------------------------------------------------------------------------------


# return self.face_vectors