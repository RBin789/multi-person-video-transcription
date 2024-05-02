class Sequence_Generation:

    def __init__(self, person_vectors):
        self.person_vectors = person_vectors
        self.person_sequences = []  # Final Format of person_sequences: [[[[vectors], frame_num, lip_sep], [[vectors], frame_num, lip_sep], ...], [[vectors], frame_num, lip_sep], [[vectors], frame_num, lip_sep], ...]] the are frames in order 

        self.generate_sequences()
        # self.print_sequences()
        

    def generate_sequences(self):
        current_sublist = []
        for i, item in enumerate(self.person_vectors):
            if len(current_sublist) == 25:
                self.person_sequences.append(current_sublist)
                current_sublist = []
            # Check if it's the first item or the current item is different from the previous one
            if i == 0 or (item[1] - 1) == self.person_vectors[i-1][1]:
                current_sublist.append(item)
            else:
                self.person_sequences.append(current_sublist)
                current_sublist = []

        # Append the last sublist if it has any elements
        if current_sublist:
            self.person_sequences.append(current_sublist)
        
        for sequence in self.person_sequences: # Add extra fake frames to the sequence so that all sequences have a length of 25
            while len(sequence) < 25:
                sequence.append(sequence[-1])

    def print_sequences(self):
        print(self.person_sequences)
        for i, sequence in enumerate(self.person_sequences):
            for j, item in enumerate(sequence):
                print(str(item[1]) + ", ", end="")

    def get_person_sequences(self):
        return self.person_sequences