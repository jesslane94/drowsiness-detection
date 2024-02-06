# Goal: detect drowsiness in students during online learning. can be extrapolated to detecting drowsiness in real time.

import os
import dlib
import cv2
import video_processing
from video import VideoRecorder
from collections import defaultdict
from flask import Flask, render_template, request


app = Flask(__name__)

@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/video_start')
def video_start():
    recording = VideoRecorder()
    recording.video_processing()
    return render_template('results.html', message = "placeholder")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='localhost', port=port)