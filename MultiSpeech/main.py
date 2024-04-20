import sys
sys.path.insert(0, 'MultiSpeech\FaceDetector')
from Face2Vec import *

def process_video():
    # Read in the image
    image_path = "MultiSpeech\FaceDetector\images\One+One_frame.png"
    face2vec = Face2Vec(image_path)
    face_vectors = face2vec.get_face_vectors()

def main():
    process_video()
    

if __name__ == "__main__":
    main()