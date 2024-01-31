# Goal: detect drowsiness in students during online learning. can be extrapolated to detecting drowsiness in real time.

import os
import dlib
import cv2
from collections import defaultdict
from flask import Flask, render_template, request


app = Flask(__name__)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='localhost', port=port)