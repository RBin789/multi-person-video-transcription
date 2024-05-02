# Import Required Libraries

from tkinter import *
from tkinter import messagebox
from tkinter import filedialog

class GUI:

    def __init__(self):
        self.run_gui()

    def run_gui(self):

        # Create a Window.
        MyWindow = Tk() # Create a window
        MyWindow.title(" Multi Person Video Transcription") # Change the Title of the GUI
        MyWindow.geometry('800x400') # Set the size of the Windows


        # Create Event Methods attached to the button etc.
        def BtnOpen_Clicked():
            messagebox.showinfo("Open Video", "Open Video Button Clicked")
            # file_path = filedialog.askopenfilename()
            # print(file_path)
            # Call the Process Video Method
            # process_video(file_path)

        # Create a new frame
        centered_frame = Frame(MyWindow, height=MyWindow.winfo_screenheight())
        centered_frame.pack(pady=(MyWindow.winfo_screenheight() - centered_frame.winfo_reqheight()) // 2)


        # Create GUI elements
        titleText = Label(centered_frame, text="Convert a video to a transcript", font=("Arial Bold", 20))
        titleText.pack(pady=20)

        descriptionText = Label(centered_frame, text="This tool allows you to convert a video to a transcript.  To get started open an video.", font=("Arial Bold", 10))
        descriptionText.pack(pady=10)

        openVideoBtn = Button(centered_frame, text="Open Video", command=BtnOpen_Clicked)
        openVideoBtn.pack(pady=10)

        # Calling the maninloop()
        MyWindow.mainloop()
