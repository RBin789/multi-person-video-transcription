import sys
sys.path.insert(0, 'MultiSpeech\FaceDetector')
from Face2Vec import *

def process_video():
    # Read in the image
    image_path = "MultiSpeech\FaceDetector\images\istockphoto-1368965646-612x612.jpg"
    instance = Face2Vec(image_path)
    # instance.__init__(image)

def main():
    process_video()
    

if __name__ == "__main__":
    main()