import os
import shutil
import re
from pydub import AudioSegment

pattern = ".flac"
root = r"LibriSpeech/train-clean-100"
root2 = r"LibriSpeech/test-clean"
root3 = r"LibriSpeech/dev-clean"
aim_path = r"LibriSpeech_wav/train"
aim_path2 = r"LibriSpeech_wav/test"
aim_path3 = r"LibriSpeech_wav/val"

if not os.path.exists(aim_path):os.makedirs(aim_path)
if not os.path.exists(aim_path2):os.makedirs(aim_path2)
if not os.path.exists(aim_path3):os.makedirs(aim_path3)
def flac_to_wav(flac_path, wav_path):
    song = AudioSegment.from_file(flac_path)
    song.export(wav_path[:-4]+'wav', format="wav")


def search(path, mypath):
    children = os.listdir(path)
    for i in children:
        p = os.path.join(path, i)
        if os.path.isdir(p):
            search(p, mypath)
        else:
            if re.match(pattern,os.path.splitext(i)[1],re.I):
                d = os.path.join(mypath,i)
                # shutil.move(p,d)
                flac_to_wav(p,d)

if __name__ == '__main__':
    search(root, aim_path)
    search(root2, aim_path2)
    search(root3, aim_path3)