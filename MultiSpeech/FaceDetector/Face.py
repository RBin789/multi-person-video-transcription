class Face:
    def __init__(self, face_vector, frame_number, lip_separation, bounding_box, face_coordinates, talking=0, label=None):
        if not face_coordinates:
            face_coordinates = [(0, 0)] * 68

        self.face_vector = face_vector
        self.frame_number = frame_number
        self.bounding_box = bounding_box
        self.face_coordinates = face_coordinates
        self.lip_separation = lip_separation
        self.label = label
        self.talking = talking

    def get_face_vector(self):
        return self.face_vector

    def get_frame_number(self):
        return self.frame_number

    def get_label(self):
        return self.label

    def get_bounding_box(self):
        return self.bounding_box

    def get_face_coordinates(self):
        return self.face_coordinates

    def get_lip_separation(self):
        return self.lip_separation

    def is_talking(self):
        return self.talking

    def set_face_vector(self, face_vector):
        self.face_vector = face_vector

    def set_frame_number(self, frame_number):
        self.frame_number = frame_number

    def set_label(self, label):
        self.label = label

    def set_bounding_box(self, bounding_box):
        self.bounding_box = bounding_box

    def set_face_coordinates(self, face_coordinates):
        self.face_coordinates = face_coordinates

    def set_lip_separation(self, lip_separation):
        self.lip_separation = lip_separation

    def set_is_talking(self, talking):
        self.talking = talking
