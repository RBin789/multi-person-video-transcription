import numpy as np
from sklearn.preprocessing import MinMaxScaler


class Lip_Detection:

    def __init__(self, sequence):
        self.sequence = sequence

        f = []
        for i, item in enumerate(sequence): # Looping though the sequence which looks like this: [[vectors], frame_num, lip_sep]
            f.append(item[2])
            print(item[2]) # Check it's working

        f = np.array(f).reshape(-1, 1)  # Reshape the array to 2D
        self.scalar = MinMaxScaler()
        arr = self.scalar.fit_transform(f)

        self.X_data = np.array([arr])

    # def detect_lip_movement(self):
        # y_pred = self.model.predict(self.X_data)  # Something llike this