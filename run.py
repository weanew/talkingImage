from talkingImage.generator import generate

IMAGE_PATH = './images/face2.jpeg'
MUSIC_PATH = './music/Juice.mp3'
OUTPUT_PATH = './output/'

if __name__ == '__main__':
    generate(IMAGE_PATH, MUSIC_PATH, OUTPUT_PATH)
