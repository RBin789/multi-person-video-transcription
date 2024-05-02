import numpy as np
from sklearn.preprocessing import MinMaxScaler


class Lip_Detection:

    def __init__(self, sequence, model):
        self.sequence = sequence
        self.model = model
        self.X_data = None

        self.prepare_sequences()
        self.detect_lip_movement()


    def prepare_sequences(self):
        f = []
        for i, item in enumerate(self.sequence): # Looping though the sequence which looks like this: [[vectors], frame_num, lip_sep]
            f.append(item[2])
            # print(item[2]) # Check it's working

        f = np.array(f).reshape(-1, 1)  # Reshape the array to 2D
        self.scalar = MinMaxScaler()
        arr = self.scalar.fit_transform(f)

        self.X_data = np.array([arr])

    def detect_lip_movement(self): # the code to look at is line 543 in lip_movement_net.py
        print(self.X_data)
        y_pred = self.model.predict(self.X_data)  # Something like this

        print(y_pred)  # Check it's working


        '''
        Mabye make a new data structure to store the lip movement data [first_frame, last_frame, lip_movement]
        this would be returned for every sequence
        '''