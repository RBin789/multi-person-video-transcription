class Sequence_Generation:

    def __init__(self, person_label, person_vectors):
        self.person_vectors = person_vectors
        self.person_label = person_label
        self.person_sequences = []  # Final Format of person_sequences: [[[[vectors], frame_num, lip_sep], [[vectors], frame_num, lip_sep], ...], [[vectors], frame_num, lip_sep], [[vectors], frame_num, lip_sep], ...]] the are frames in order 

        if self.person_label == 0:
            self.person_label = "A"
        else:
            self.person_label = "B"

        self.generate_sequences()
        # self.print_sequences()
        

    def generate_sequences(self):
        current_sublist = []
        for i, item in enumerate(self.person_vectors):

            if (len(current_sublist) == 0) or ((item[1]) == (self.person_vectors[i-1][1]+1)):           # Add current frame to curent frame list, if first item or current frame is one more than previous i.e. continuous segment
                current_sublist.append(item)
                                                
            else:
                current_sublist = current_sublist * (25 // len(current_sublist) + 1)                    # If user segment changes, then repeat the list so far to extend
                current_sublist = current_sublist[:25]                                                  # Trim the list to 25 frames

            if len(current_sublist) == 25:
                self.person_sequences.append(current_sublist)                                           # Appends the processed sublist to main list
                current_sublist = []                                                                    # Create a new sublist
  
        if len(current_sublist) != 0:                                                                   # After loop is finished, append close off sublist, if it is non empty sublistAppend the last sublist if it has any elements
            current_sublist = current_sublist * (25 // len(current_sublist) + 1)
            current_sublist = current_sublist[:25]                                                      # Trim the list to 25 frames
            self.person_sequences.append(current_sublist)                                               # Appends the processed sublist to main list

    def print_sequences(self):
        for i, sequence in enumerate(self.person_sequences):
            for j, item in enumerate(sequence):
                print(self.person_label + str(item[1]) + ", ", end="")
            print(" length=" + str(len(sequence)), end="")
            print("\n")

    def get_person_sequences(self):
        return self.person_sequences