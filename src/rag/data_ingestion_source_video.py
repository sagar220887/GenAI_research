
# Install libraries
# pip install -r requirements.txt
# pip install git+https://github.com/openai/CLIP.git

from moviepy.editor import VideoFileClip
from pathlib import Path
import speech_recognition as sr
from pytube import YouTube
from pprint import pprint
import cv2 
from PIL import Image
import matplotlib.pyplot as plt
import os
import sys

VIDEO_URL = "https://youtu.be/3dhcmeOTZ_Q"
VIDEO_OUTPUT_PATH = "./video_data/"
OUTPUT_FOLDER = "./output_data/"

VIDEO_FILE_NAME = 'input_video.mp4'
VIDEO_FILE = os.path.join(VIDEO_OUTPUT_PATH,VIDEO_FILE_NAME)
IMAGE_FILE = os.path.join(OUTPUT_FOLDER,'frame%04d.png')


AUDIO_FILE_NAME = "output_audio.wav"
AUDIO_FILE = os.path.join(OUTPUT_FOLDER, AUDIO_FILE_NAME)

TEXT_FILE_NAME = "audio_to_text.txt"
TEXT_FILE = os.path.join(OUTPUT_FOLDER, TEXT_FILE_NAME)

VECTOR_DB_DIRECTORY = './vectorDB'


def download_video(video_url,video_output_path):
    try:
        if not video_url.startswith('http'):
            sys.exit('Could not found youtube URL')
        print('video_url - ', video_url)
        yt = YouTube(video_url)
        metadata={
            "author": yt.author,
            "title": yt.title,
            "views": yt.views
        }
        if not os.path.exists(VIDEO_OUTPUT_PATH):
            os.makedirs(VIDEO_OUTPUT_PATH)

        yt.streams.get_highest_resolution().download(video_output_path, VIDEO_FILE_NAME)
        print(f'========= 1. Video created in the directory {video_output_path} as :: {VIDEO_FILE_NAME}')
        return metadata
    except Exception as e:
        print(e)
        return None
    

def convert_video_to_images(video_path, image_path_to_save):
    try: 
        if not os.path.exists(video_path):
            sys.exit('Could not found video in the specified path => ', video_path)
        cam = cv2.VideoCapture(video_path) 
        currentframe = 0

        if not os.path.exists(image_path_to_save):
            os.makedirs(image_path_to_save)
        
        while(True): 
            ret,frame = cam.read() 
            if ret: 
                name = os.path.join(image_path_to_save, f'{currentframe}.jpg')
                print ('Creating...' + name) 
                cv2.imwrite(name, frame) 
                currentframe += 1
            else: 
                break
        cam.release() 
        cv2.destroyAllWindows()
        print('========= 2. Successfully converted images from video. Saved images in => ', image_path_to_save) 

    except Exception as e:
        print('Exception inside convert_video_to_images - ', e)
        return None
    

def convert_video_to_audio(video_path, audio_path_to_save):
    if not os.path.exists(video_path):
        sys.exit('Could not found video in the specified path => ', video_path)
    clip = VideoFileClip(video_path)
    audio = clip.audio

    audio.write_audiofile(audio_path_to_save)
    print('========= 3. Successfully converted audio from video. Saved audio in => ', audio_path_to_save) 

def save_text_to_file(content, filepath):
    
    file1 = open(filepath, 'w')
    file1.writelines(content)
    file1.close()
    
    file1 = open(filepath, 'r')
    print(file1.read())
    file1.close()



def convert_audio_to_text(audio_path):
    if not os.path.exists(audio_path):
        sys.exit('Could not found audio in the specified path => ', audio_path)
    print('audio_path => ', audio_path)
    recognizer = sr.Recognizer()
    audio = sr.AudioFile(audio_path)

    with audio as source:
        audio_data = recognizer.record(source)

    try:
        audio_text = recognizer.recognize_whisper(audio_data)
        save_text_to_file(audio_text, TEXT_FILE)
    except sr.UnknownValueError:
        print("Could not understand audio")

    print('========= 4. Successfully converted audio to text. Saved text file in => ', audio_path) 
    return audio_text


def extract_audio_and_text_from_video():
    download_video(VIDEO_URL, VIDEO_OUTPUT_PATH)
    convert_video_to_images(VIDEO_FILE, OUTPUT_FOLDER)
    convert_video_to_audio(VIDEO_FILE, AUDIO_FILE)
    convert_audio_to_text(AUDIO_FILE)


if __name__ == '__main__':
    extract_audio_and_text_from_video()