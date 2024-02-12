# Goal: detect drowsiness in students during online learning. can be extrapolated to detecting drowsiness in real time.

import os
from video import VideoRecorder
from flask import Flask, render_template, request
import time

global recording

app = Flask(__name__)

@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/video_start')
def video_start():
    recording = VideoRecorder()
    recording.video_processing()
    print(recording.get_totals())
    return render_template('results.html', totals = recording.get_totals())

""" @app.route('/video_stop')
def video_stop():
    # load buffer
    time.sleep(0.5)
    totals = {}
    totals = recording.get_totals()
    del recording
    return render_template('results.html', totals = totals) """
    

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='localhost', port=port)