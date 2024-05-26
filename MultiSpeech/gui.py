import sys
import time
import cv2
import threading
import subprocess
from tkinter import *
from tkinter import messagebox
from tkinter import filedialog
from tkinter import ttk
from PIL import Image, ImageTk
from moviepy.editor import VideoFileClip
import moviepy.editor as mp

class GUI:

    def __init__(self):
        self.selected_file = None
        self.main_window = Tk()
        self.main_window.title("Multi Person Video Transcription")
        self.main_window.geometry('800x400')
        self.setup_first_window()
        self.main_window.mainloop()

    def setup_first_window(self):
        centered_frame = Frame(self.main_window)
        centered_frame.pack(pady=50)

        title_text = Label(centered_frame, text="Convert a video to a transcript", font=("Arial Bold", 20))
        title_text.pack(pady=20)

        description_text = Label(centered_frame, text="This tool allows you to convert a video to a transcript. To get started press the open video button.", font=("Arial Bold", 10))
        description_text.pack(pady=10)

        open_video_btn = Button(centered_frame, text="Open Video", command=self.btn_open_clicked)
        open_video_btn.pack(pady=10)

        people_input_label = Label(centered_frame, text="Enter the number of unique people who appear in the video:", font=("Arial Bold", 10))
        people_input_label.pack(pady=10)

        self.number_entry = Entry(centered_frame, width=20)
        self.number_entry.pack(pady=10)
        self.number_entry.insert(0, "")

        self.start_button = Button(centered_frame, text="Start", state=DISABLED, command=self.btn_start_clicked)
        self.start_button.pack(pady=10)

        cancel_button = Button(centered_frame, text="Cancel", command=self.btn_cancel_clicked)
        cancel_button.pack(pady=10)

    def btn_open_clicked(self):
        filetypes = [("Video files", "*.mp4")]
        file_path = filedialog.askopenfilename(filetypes=filetypes)
        if file_path:
            self.selected_file = file_path
            self.start_button.config(state=NORMAL)

    def btn_start_clicked(self):
        number_people = int(self.number_entry.get())
        loading_thread = threading.Thread(target=self.show_loading_screen)
        loading_thread.start()
        processing_thread = threading.Thread(target=self.process_video_and_show_second_window)
        processing_thread.start()

    def show_loading_screen(self):
        global loading_window
        loading_window = Toplevel(self.main_window)
        loading_window.title("Processing")
        loading_window.geometry('200x100')
        loading_label = Label(loading_window, text="Processing, please wait...", font=("Arial", 10))
        loading_label.pack(pady=20)
        progress_bar = ttk.Progressbar(loading_window, mode='indeterminate')
        progress_bar.pack(expand=True)
        progress_bar.start()

    def process_video_and_show_second_window(self):
        # Run Vid2Wav.py to extract audio clips
        subprocess.run(['python', 'Vid2Wav.py', self.selected_file])
        loading_window.destroy()
        self.show_second_window()

    def show_second_window(self):
        self.second_window = Toplevel(self.main_window)
        self.second_window.title("Video Analysis Result")
        self.second_window.geometry('1200x800')

        left_frame = Frame(self.second_window, width=720, height=800)
        left_frame.pack(side=LEFT, padx=10, pady=10)
        left_frame.pack_propagate(False)

        video_label = Label(left_frame, text="Video Player", font=("Arial Bold", 20))
        video_label.pack(pady=20)
        
        # Video display
        self.video_canvas = Canvas(left_frame, width=720, height=560)
        self.video_canvas.pack()

        self.play_button = Button(left_frame, text="Play", command=self.play_video)
        self.play_button.pack(side=LEFT, padx=5, pady=5)

        self.pause_button = Button(left_frame, text="Pause", command=self.pause_video)
        self.pause_button.pack(side=LEFT, padx=5, pady=5)

        self.replay_button = Button(left_frame, text="Replay", command=self.replay_video)
        self.replay_button.pack(side=LEFT, padx=5, pady=5)

        self.cap = cv2.VideoCapture(self.selected_file)

        right_frame = Frame(self.second_window, width=480, height=800)
        right_frame.pack(side=RIGHT)
