# Import Required Libraries

from tkinter import *
from tkinter import messagebox
from tkinter import filedialog
# import cv2 <-- To use openCV function/methods

# Create a Window.
MyWindow = Tk() # Create a window
MyWindow.title("First GUI") # Change the Title of the GUI
MyWindow.geometry('400x200') # Set the size of the Windows


# Create the GUI Component but dont display or add them to the window yet.
MyLabel = Label(text = "Click to Open an Image", font=("Arial Bold", 10))

ClassficationResultLabel = Label(text = "Classification Result: ", font=("Arial Bold", 10))

InfoLabel = Label(text = "Enter Your Name:", font=("Arial Bold", 10)) # Addind information on the Text Entry box.
userEntry = Entry(width = 20) # Allows to Enter single line of text


# Create the Custom Methods for Processing the Images/Video using DL model

# Open Image Function using OpenCV
def openImg(filename):
    messagebox.showinfo("Image to Show", filename)
    # Open the image using OPENCV
    # img = imread(filename)
    # cv.imshow(img)



# Create Event Methods attached to the button etc.
def BttnOpen_Clicked():
    messagebox.showinfo("Info", "Open Button Clicked")
    # Use the File Dialog component to Open the Dialog box to select files
    file = filedialog.askopenfilename(filetypes = (("Images files","*.jpg"),("Video Files","*.mp4"),("all files","*.*")))
    messagebox.showinfo("File Selected", file)
    openImg(file) # Passing the file to openImg method to show is using opencv (imread, imshow)
    

def BttnProcess_Clicked():
    messagebox.showinfo("Info", "Process Button Clicked")
    # Read and process images/frame using your DL model here <--
    # Testing
    #messagebox.showwarning("Invalid Input","Image is having an invalid format") # Showing Warning not very Critcal 
    #messagebox.showerror("Invalid Input","Image is having an invalid format") # Showing Error, very Critcal 
    #classifcationResult = "CAT"
    #messagebox.showinfo("Classfication Result", classifcationResult)
    result = "DOG" # model.predict(file) for example 
    resultText = "Classification Result:" + result  # Concatenate the result class to the Label on the Window
    ClassficationResultLabel.configure(text = resultText)  # Update the Label text on the Window
    
    

# Add the Components create previsously to the window

MyLabel.grid(column=0, row=1) # Adding the Label
openBttn = Button(text="Open Image", command=BttnOpen_Clicked)
openBttn.grid(column=1, row=1) # Adding the Open Button
openProcess = Button(text="Process Image", command=BttnProcess_Clicked)
openProcess.grid(column=2, row=1) # Adding the Process Button
ClassficationResultLabel.grid(column=0, row=2) # Adding the label to display classfication result
MyLabel.grid(column=0, row=3) # Adding label for information on the Text Entry box
userEntry.grid(column=1, row=3) # Adding the Text entry Widget

# Calling the maninloop()
MyWindow.mainloop()
