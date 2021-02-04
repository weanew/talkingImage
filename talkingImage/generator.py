import os
import uuid
from multiprocessing import Process, Manager

import cv2
import ffmpeg
import matplotlib.pyplot as plt
import numpy as np
from spleeter.audio.adapter import AudioAdapter
from spleeter.separator import Separator

from talkingImage.detect import LandmarkFace
from talkingImage.transformation import LipsTransform


def generate(image_path, music_path, output_path):
    decimals = 1
    FPS = 30
    sample_rate = 44100

    def generate_images(decimals, lipsT):
        imgs = {}
        for i in range(10 ** decimals):
            coef = i / (10 ** decimals)
            imgs[str(coef)] = lipsT.transformate(coef)
        return imgs

    def image_processing(image_pth, ret_value):
        img = plt.imread(image_pth)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (1024, 1024), interpolation=cv2.INTER_LINEAR)
        lface = LandmarkFace.estimate(img)
        lips = LipsTransform(lface)
        imgs = generate_images(decimals, lips)
        ret_value['images'] = imgs
        return imgs

    def music_processing(music_pth, ret_value):
        separator = Separator(params_descriptor='spleeter:2stems')

        audio_adapter = AudioAdapter.get('spleeter.audio.ffmpeg.FFMPEGProcessAudioAdapter')
        waveform, _ = audio_adapter.load(
            music_path,
            dtype=np.float32,
            sample_rate=22050
        )
        sources = separator.separate(waveform=waveform, audio_descriptor=music_pth)
        vocals = sources['vocals']
        ret_value['vocals'] = vocals
        return vocals

    manager = Manager()
    ret_dict = manager.dict()
    procs = [
        Process(target=music_processing, args=(music_path, ret_dict)),
        Process(target=image_processing, args=(image_path, ret_dict))
    ]

    for p in procs:
        p.start()

    for p in procs:
        p.join()

    voc = ret_dict['vocals']
    images = ret_dict['images']

    out = cv2.VideoWriter('temp_video.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), FPS, (1024, 1024))
    time = voc.shape[0] * voc.shape[1] / sample_rate
    frames_count = time * FPS

    for i in range(1, int(frames_count)):
        coeffs = [np.max(voc[j]) for j in range(
            (i-1) * int(sample_rate / FPS / 2),
            (i * int(sample_rate / FPS / 2)) - 1)]

        if len(coeffs) != 0:
            coeff = round(abs(max(coeffs)), decimals)
        else:
            coeff = 0

        img_to_write = images[str(coeff)]
        out.write(img_to_write)

    out.write(images['0.0'])
    out.release()

    input_video = ffmpeg.input('temp_video.avi')
    input_audio = ffmpeg.input(music_path)
    ffmpeg.concat(input_video, input_audio, v=1, a=1).output(output_path + str(uuid.uuid4()) + '.mp4').run()
    os.remove('temp_video.avi')
