class Person:

    def __init__(self, label=None):
        self.label = label
        self.faces = []
    
    def get_label(self):
        return self.label
    
    def set_label(self, label):
        self.label = label
    
    def add_face(self, face):
        self.faces.append(face)
    
    def get_faces(self):
        return self.faces
    
    
