from moviepy.editor import VideoFileClip, VideoClip
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime

class CreateProcessedVideo:
    def __init__(self, video_path, all_faces, all_sequences, current_time):
        self.video_path = video_path
        self.all_faces = all_faces
        self.all_sequences = all_sequences
        self.current_time = current_time
        self.annotated_video_path = None
        self.current_frame_num = 0
        self.video_width = 0
        self.video_height = 0

        self.process_video()

    def process_video(self):
        """Create a new version of the video with all information about the faces plotted"""

        # Load the video file
        video = VideoFileClip(self.video_path)
        width, height = video.size
        self.video_width = width
        self.video_height = height
        
        def process_frame(get_frame, t):
            """Process each frame of the video."""

            frame = get_frame(t)
            self.current_frame_num = int(t * video.fps) + 1

            for face in self.all_faces:                
                if face.get_frame_number() == self.current_frame_num:
                    frame = self.draw_bounding_box(frame, face.get_bounding_box(), face.get_face_coordinates(), face.is_talking(), face.get_label())

            return frame
        
        processed_video = video.fl(process_frame)
        # processed_video.write_videofile(self.video_path + "_modified.mp4", codec='libx264')
        self.annotated_video_path = self.video_path[:-4] + "_annotated_" + str(self.current_time) + ".mp4"
        processed_video.write_videofile(self.video_path[:-4] + "_annotated_" + str(self.current_time) + ".mp4", codec='libx264')

    def draw_bounding_box(self, frame, bounding_box, face_coordinates, talking, label):
        """Draw the bounding box around the face and the face landmarks."""

        frame = Image.fromarray(frame)
        draw = ImageDraw.Draw(frame)
        label = "Person: " + str(label)
        
        self.draw_text(draw, (self.video_width - 200), 20, "Frame: " + str(self.current_frame_num), (255, 0, 0)) # Draw frame counter

        x1, y1, x2, y2 = bounding_box

        if talking == 2: # If the person is talking
            self.draw_rectangle(draw, x1, y1, x2, y2, color=(255, 0, 0))
            self.draw_text(draw, x1, y1 - 40, label, color=(255, 0, 0))

            for i, point in enumerate(face_coordinates):
                color = (255, 0, 0) if i in [61, 62, 63, 65, 66, 67] else (255, 255, 0) # Red for lips, yellow for others
                self.draw_circle(draw, point[0] + x1, point[1] + y1, color)
        
        elif talking == 1: # If the person is not talking
            self.draw_rectangle(draw, x1, y1, x2, y2, color=(0, 255, 0))
            self.draw_text(draw, x1, y1 - 40, label, color=(0, 255, 0))

            for i, point in enumerate(face_coordinates):
                color = (255, 0, 0) if i in [61, 62, 63, 65, 66, 67] else (255, 255, 0) # Red for lips, yellow for others
                self.draw_circle(draw, point[0] + x1, point[1] + y1, color)

        return np.array(frame)
    
    def draw_rectangle(self, draw, x, y, w, h, color):
        """Draw a rectangle on the image."""

        draw.rectangle([x, y, w, h], outline=color, width=2)
    
    def draw_circle(self, draw, x, y, color):
        """Draw a circle on the image."""

        r = 2  # radius
        draw.ellipse([x-r, y-r, x+r, y+r], fill=color, outline=color)

    def draw_text(self, draw, x, y, label, color):
        """Draw text on the image."""

        try:
            font = ImageFont.truetype("arial.ttf", 30)
            draw.text((x, y), label, fill=color, font=font)
        except Exception as e:
            print(f"Error loading font: {e}")
            draw.text((x, y), label, fill=color)

    def get_annotated_video_path(self):
        """Return the path to the annotated video."""

        return self.annotated_video_path
