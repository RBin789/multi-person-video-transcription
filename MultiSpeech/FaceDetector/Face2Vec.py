import cv2
import dlib

class Face2Vec:
    def __init__(self, image):
        self.img = cv2.imread(image)
        self.processed_image = None
        cropped_faces = []
        face_keypoints = []
        face_vectors = []

        self.process_image()
        self.detect_faces()
        self.detect_keypoints()
        self.convert_to_vector()
    

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

        self.cropped_faces.append(self.img[y1:y2, x1:x2])
    

    def detect_keypoints(self):
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        for face in self.cropped_faces:
            gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            rects = detector(gray, 0)
            for rect in rects:
                shape = predictor(gray, rect)
                self.face_keypoints.append(shape)


    def show_keypoints(self):
        #don't know if this will work
        for face in self.cropped_faces:
            for shape in self.face_keypoints:
                for i in range(68):
                    x = shape.part(i).x
                    y = shape.part(i).y
                    cv2.circle(face, (x, y), 2, (0, 255, 0), -1)
            cv2.imshow('Face', face)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

# -----------------------------------------------------------------------------------------------


return self.face_vectors