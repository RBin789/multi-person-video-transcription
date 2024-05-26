from moviepy.editor import VideoFileClip, VideoClip
import numpy as np
from PIL import Image, ImageDraw, ImageFont

class CreateProcessedVideo:
    def __init__(self, video_path, all_persons, all_sequences):
        self.video_path = video_path
        self.all_persons = all_persons
        self.all_sequences = all_sequences
        self.current_frame_num = 0

        self.process_video()

    def process_video(self):
        # Load the video file
        video = VideoFileClip(self.video_path)
        
        # Define a function to process each frame
        def process_frame(get_frame, t):
            frame = get_frame(t)
            self.current_frame_num = int(t * video.fps) + 1

            for person in self.all_persons:                
                if person.get_frame_number() == self.current_frame_num:
                    frame = self.draw_bounding_box(frame, person.get_bounding_box(), person.get_face_coordinates(), person.is_talking(), person.get_label())
                    
            return frame

        # Create a new video with the processed frames
        processed_video = video.fl(process_frame)
        processed_video.write_videofile(self.video_path + "_modified.mp4", codec='libx264')

    def draw_bounding_box(self, frame, bounding_box, face_coordinates, talking, label):
        # Convert frame to PIL Image
        frame = Image.fromarray(frame)
        draw = ImageDraw.Draw(frame)
        label = "Person: " + str(label)
        
        # Draw frame counter
        self.draw_text(draw, 20, 20, "Frame: " + str(self.current_frame_num), (255, 0, 0))

        x1, y1, x2, y2 = bounding_box
        if talking == 2:
            self.draw_rectangle(draw, x1, y1, x2, y2, color=(255, 0, 0))
            self.draw_text(draw, x1, y1 - 40, label, color=(255, 0, 0))

            for i, point in enumerate(face_coordinates):
                color = (255, 0, 0) if i in [61, 62, 63, 65, 66, 67] else (255, 255, 0)
                self.draw_circle(draw, point[0] + x1, point[1] + y1, color)
        
        elif talking == 1:
            self.draw_rectangle(draw, x1, y1, x2, y2, color=(0, 255, 0))
            self.draw_text(draw, x1, y1 - 40, label, color=(0, 255, 0))

            for i, point in enumerate(face_coordinates):
                color = (255, 0, 0) if i in [61, 62, 63, 65, 66, 67] else (255, 255, 0)
                self.draw_circle(draw, point[0] + x1, point[1] + y1, color)         

        # Convert frame back to numpy array
        return np.array(frame)
    
    def draw_rectangle(self, draw, x, y, w, h, color):
        draw.rectangle([x, y, w, h], outline=color, width=2)
    
    def draw_circle(self, draw, x, y, color):
        r = 2  # radius
        draw.ellipse([x-r, y-r, x+r, y+r], fill=color, outline=color)

    def draw_text(self, draw, x, y, label, color):
        try:
            font = ImageFont.truetype("arial.ttf", 30)
            draw.text((x, y), label, fill=color, font=font)
        except Exception as e:
            # print(f"Error loading font: {e}")
            draw.text((x, y), label, fill=color)