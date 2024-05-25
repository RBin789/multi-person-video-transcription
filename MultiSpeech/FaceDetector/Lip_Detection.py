import numpy as np
from sklearn.preprocessing import MinMaxScaler


class Lip_Detection:  # This class takes in 25 values of lip separation and predicts the lip movement

    def __init__(self, sequence, cluster_label, model):
        self.sequence = sequence  # this list looks like this: [[vectors], frame_num, lip_sep]
        self.model = model
        self.cluster_label = cluster_label
        self.X_data = None
        self.sequence_and_prediction = None


        self.prepare_sequences()
        self.detect_lip_movement()
        self.get_sequence_and_prediction()


    def prepare_sequences(self):
        f = []
        for i, item in enumerate(self.sequence): # Looping though the sequence which looks like this: [[vectors], frame_num, lip_sep]
            f.append(item[2])  # Append the lip separation value to the list
        if f:  # If the list is not empty
            f = np.array(f).reshape(-1, 1)  # Reshape the array to 2D
            self.scalar = MinMaxScaler() 
            arr = self.scalar.fit_transform(f)
            self.X_data = np.array([arr]) 
        else:
            raise ValueError("The sequence attribute doesn't contain the expected data.")
        
    def detect_lip_movement(self): # the code to look at is line 543 in lip_movement_net.py
        y_pred = self.model.predict_on_batch(self.X_data)  # Predict the lip movement
        y_pred_max = y_pred[0].argmax()
        
        unique_sequence = self.get_unique_values(self.sequence)

        self.sequence_and_prediction = [self.cluster_label, unique_sequence, y_pred_max] # Set the cluster label, start frame, end frame, and prediction to the list
        

        # print("The prediction this sequence is " + str(y_pred_max))
    
    def get_unique_values(self, sequence):
        frame_nums = []
        for i, item in enumerate(sequence):
            frame_nums.append(item[1])
        return list(set(frame_nums))  # This could end up making the list lose it's order
    
    def get_sequence_and_prediction(self):
        return self.sequence_and_prediction
