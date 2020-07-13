import argparse
from pydub import AudioSegment
from pydub.playback import play

parser = argparse.ArgumentParser()
parser.add_argument('code', type=int,
                    help='A required music positional argument')


def main(code):
    song = AudioSegment.from_mp3('./voice/Voice/' + str(code) + '.mp3')
    play(song)


"""
second choice
"""
#
# from playsound import playsound
# playsound('C:\\Users\\NeuHeLium\\Downloads\\FirstMp3.mp3')

if __name__ == '__main__':
    args = parser.parse_args()
    main(args.code)
