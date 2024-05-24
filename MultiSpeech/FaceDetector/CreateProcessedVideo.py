from moviepy.editor import VideoFileClip, VideoClip
import numpy as np
from PIL import Image, ImageDraw

class CreateProcessedVideo:
    def __init__(self, video_path, all_persons, all_sequences):
        self.video_path = video_path
        self.all_persons = all_persons
        self.all_sequences = all_sequences

        self.process_video()

    def process_video(self):
        # Load the video file
        video = VideoFileClip(self.video_path)
        
        # Define a function to process each frame
        def process_frame(get_frame, t):
            frame = get_frame(t)
            current_frame_num = int(t * video.fps) + 1
            
            for person in self.all_persons:
                if person.get_frame_number() == current_frame_num:
                    frame = self.draw_bounding_box(frame, person.get_bounding_box(), person.get_face_coordinates(), person.is_talking())
                    
            return frame

        # Create a new video with the processed frames
        processed_video = video.fl(process_frame)
        processed_video.write_videofile(self.video_path + "_modified.mp4", codec='libx264')

    def draw_bounding_box(self, frame, bounding_box, face_coordinates, talking):
        # Convert frame to PIL Image
        frame = Image.fromarray(frame)
        draw = ImageDraw.Draw(frame)
        
        if talking:
            x, y, w, h = bounding_box
            self.draw_rectangle(draw, x, y, w, h, color=(255, 0, 0))

            for i, point in enumerate(face_coordinates):
                color = (255, 0, 0) if i in [61, 62, 63, 65, 66, 67] else (255, 255, 0)
                self.draw_circle(draw, point[0] + x, point[1] + y, color)
        
        else:
            x, y, w, h = bounding_box
            self.draw_rectangle(draw, x, y, w, h, color=(0, 255, 0))

        # Convert frame back to numpy array
        return np.array(frame)
    
    def draw_rectangle(self, draw, x, y, w, h, color):
        draw.rectangle([x, y, w, h], outline=color, width=2)
    
    def draw_circle(self, draw, x, y, color):
        r = 2  # radius
        draw.ellipse([x-r, y-r, x+r, y+r], fill=color, outline=color)
