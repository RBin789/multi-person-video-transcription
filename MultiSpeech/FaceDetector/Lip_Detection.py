import numpy as np


class Lip_Detection:  # This class takes in 25 values of lip separation and predicts the lip movement

    def __init__(self, sequence, cluster_label, model, all_lip_seperations, lip_index):
        self.sequence = sequence  # this list looks like this: [[vectors], frame_num, lip_sep]
        self.model = model
        self.cluster_label = cluster_label
        self.X_data = None
        self.sequence_and_prediction = None
        self.all_lip_seperations = all_lip_seperations
        self.lip_index = lip_index  # 0 = outer lip, 1 = inner lip``

        self.prepare_sequences()
        self.detect_lip_movement()
        self.get_sequence_and_prediction()

    def prepare_sequences(self):
        """Prepare the sequences for prediction."""

        start_frame = self.sequence[0][1]  # Get the start frame number of the sequence
        end_tuple = max(self.sequence, key=lambda x: x[1])  # Get the tuple with the highest frame number in the sequence
        end_frame = end_tuple[1]  # Get the end frame number of the sequence

        if start_frame == end_frame:  # If the start frame and end frame are the same
            max_lip_sep = self.all_lip_seperations[start_frame]  # Get the lip separation value at the start frame
        else:
            max_lip_sep = max(self.all_lip_seperations[start_frame:end_frame])  # Get the maximum lip separation value in the sequence

        if self.lip_index == 0:
            lip_threshold = 0.9 # 0.4 for outer lip
        else:
            lip_threshold = 0.4 # 0.2 for inner lip
        
        # print("Max lip sep: " + str(max_lip_sep))
        # print("Start frame: " + str(start_frame) + ", End frame: " + str(end_frame) + " len: " + str(len(self.sequence)))
        # print()

        f = []

        for i, item in enumerate(self.sequence):
            f.append(item[2][self.lip_index])  # Append the lip separation value to the list
        
        if f:  # If the list is not empty
            f = np.array(f).reshape(-1, 1)  # Reshape the array to 2D
            
            min_val = np.min(f)
            max_val = np.max(f)

            if (max_val / max_lip_sep) < lip_threshold:
                scaled_data = (f - min_val) / (max_lip_sep - min_val) # Scale the data (This is a manual min-max scaling)

            else:
                scaled_data = (f - min_val) / (max_val - min_val) # Scale the data (This is a manual min-max scaling)
                
            self.X_data = np.array([scaled_data])
            
            # if (self.sequence[0][1] >= 0 and self.sequence[0][1]  <= 2000):
            #     print("Person: " + str(self.cluster_label)  + "  Start Frame: " + str(self.sequence[0][1]) + " Using lip index: " + str(self.lip_index) + ", lip sep data: " + str(f))
            #     print("Person: " + str(self.cluster_label)  + "  Start Frame: " + str(self.sequence[0][1]) + " Using lip index: " + str(self.lip_index) + ", lip sep data: " + str(scaled_data))
        else:
            raise ValueError("The sequence attribute doesn't contain the expected data.")
        
    def detect_lip_movement(self):
        """Detect the lip movement."""

        y_pred = self.model.predict_on_batch(self.X_data)  # Predict the lip movement
        y_pred_max = y_pred[0].argmax() # Get the index of the highest value in the prediction
        
        unique_sequence = self.get_unique_values(self.sequence)
        # frame_nums = []
        # for i, item in enumerate(self.sequence):
        #     frame_nums.append(item[1])

        self.sequence_and_prediction = [self.cluster_label, unique_sequence, y_pred_max] # Set the cluster label, start frame, end frame, and prediction to the list
        # self.sequence_and_prediction = [self.cluster_label, frame_nums, y_pred_max] # Set the cluster label, start frame, end frame, and prediction to the list
    
    def get_unique_values(self, sequence):
        """Get the unique values from the sequence."""

        frame_nums = []
        for i, item in enumerate(sequence):
            frame_nums.append(item[1])
        
        frame_nums = list(set(frame_nums))
        frame_nums.sort()
        return frame_nums
    
    def get_sequence_and_prediction(self):
        """Return the sequence and prediction."""

        return self.sequence_and_prediction
