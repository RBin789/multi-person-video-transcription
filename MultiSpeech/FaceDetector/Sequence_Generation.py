class Sequence_Generation:

    def __init__(self, person_vectors):
        self.person_vectors = person_vectors
        self.person_sequences = []

        self.generate_sequences()
        self.print_sequences()
        

    def generate_sequences(self):
        current_sublist = []
        for i, item in enumerate(self.person_vectors):
            # print(str(item[1]) + " " + str(self.person_vectors[i-1][1]))
            # Check if it's the first item or the current item is different from the previous one
            if i == 0 or (item[1] - 1) == self.person_vectors[i-1][1]:
                current_sublist.append(item)
            else:
                self.person_sequences.append(current_sublist)
                current_sublist = []

        # Append the last sublist if it has any elements
        if current_sublist:
            self.person_sequences.append(current_sublist)

    def print_sequences(self):
        for i, sequence in enumerate(self.person_sequences):
            for j, item in enumerate(sequence):
                print(str(item[1]) + ", ", end="")

    def get_person_sequences(self):
        return self.person_sequences 