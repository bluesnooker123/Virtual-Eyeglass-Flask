import os
import shutil
from flask import Flask, render_template, request, Response, redirect, url_for, jsonify
from camera import Camera
from datetime import datetime
import json
import cv2 as cv

from lib.overlay_accessory import overlay_accessory
from lib.landmark_detection import LandmarkDetector
from lib.image_object import ImageObject
from lib.emotion_detect.emotion_detector import EmotionDetector

app = Flask(__name__)
camera = None

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
PREDICTOR_PATH_LOCAL = os.path.join(ROOT_PATH, 'lib/shape_predictor_68_face_landmarks.dat')
OBJECT_PATH_LOCAL = os.path.join(ROOT_PATH, 'objects/objects.json')

Expressions = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Sad", 5: "Surprise", 6: "Neutral"}

landmark_detector = LandmarkDetector(predictor_path=PREDICTOR_PATH_LOCAL)
emotion_detector = EmotionDetector()

with open(OBJECT_PATH_LOCAL) as json_file:
    data = json.load(json_file)
    obj_keys = data.keys()

if len(obj_keys) == 0:
    print("Failed to load Overlay Object")

cache = {
    'overlay_obj_index': 0,
    'emotion': []
}


def get_camera():
    global camera
    if not camera:
        camera = Camera()

    return camera


@app.route('/')
def root():
    return redirect(url_for('index'))


@app.route('/index/')
def index():
    return render_template('start.html')


def gen(camera):
    prev_time = datetime.now()

    while True:
        frame = camera.get_feed()

        if len(list(obj_keys)) > 0:
            obj_id = list(obj_keys)[cache['overlay_obj_index']]
            frame = face_overlay_accessory(src_img=frame, accessory_obj_id=obj_id)

            current_time = datetime.now()
            if (current_time - prev_time).total_seconds() > 0.5:
                prev_time = current_time
                emotion_frame = frame
                detect_result = emotion_detector.detect(emotion_frame)
                if detect_result:
                    [x, y, emotion_result] = detect_result
                    cache['emotion'].append(1 if emotion_result == "Happy" else 0)

        if frame is not None:
            ret, jpeg = cv.imencode('.jpg', frame)
            frame_bytes = jpeg.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/video_feed/')
def video_feed():
    camera = get_camera()
    return Response(gen(camera),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/emotion_feed/')
def emotion_feed():
    def generate():
        with app.app_context():
            emotion_result = cache['emotion']
            if len(emotion_result) > 0:
                positive_emotions = len([emotion for emotion in emotion_result if emotion == 1])
                negative_emotions = len([emotion for emotion in emotion_result if emotion == 0])
                cache['emotion'] = []
                if positive_emotions > negative_emotions:
                    resp = 'suggest'
                else:
                    resp = 'change'
            else:
                resp = 'none'
            print("Suggest Result : ", resp)
            yield resp
    return Response(generate(), mimetype='text')


@app.route('/change_spectacle/')
def change_spectacle():
    change_method = request.args.get('method')
    print("======================")
    print(change_method)
    print("======================")
    if change_method == 'start':
        cache['overlay_obj_index'] = 0   
    elif change_method == 'glass_1':
        cache['overlay_obj_index'] = 0
    elif change_method == 'glass_2':
        cache['overlay_obj_index'] = 1
    elif change_method == 'glass_3':
        cache['overlay_obj_index'] = 2
    elif change_method == 'glass_4':
        cache['overlay_obj_index'] = 3
    elif change_method == 'glass_5':
        cache['overlay_obj_index'] = 4
    elif change_method == 'glass_6':
        cache['overlay_obj_index'] = 5
    elif change_method == 'glass_7':
        cache['overlay_obj_index'] = 6
    elif change_method == 'glass_8':
        cache['overlay_obj_index'] = 7
    elif change_method == 'glass_9':
        cache['overlay_obj_index'] = 8
    elif change_method == 'glass_10':
        cache['overlay_obj_index'] = 9
    else:
        cache['overlay_obj_index'] = 0
    return render_template('index.html')


@app.route('/capture/')
def capture():
    camera = get_camera()
    stamp = camera.capture()
    return redirect(url_for('show_capture', timestamp=stamp))


def stamp_file(timestamp):
    return 'captures/' + timestamp + ".jpg"


def load_accessory_obj(obj_id):
    with open(OBJECT_PATH_LOCAL) as json_file:
        data = json.load(json_file)
        obj_json = data[obj_id]
        acc_img = cv.imread(os.path.join(ROOT_PATH, obj_json['path']), cv.IMREAD_UNCHANGED)
        # acc_img = cv2.imread('glasses2.png')
        acc_info = obj_json['info']
        return acc_img, acc_info


def face_overlay_accessory(src_img, accessory_obj_id=None):
    """ FACE OVERLAY WITH ACCESSORY """
    margin = 1.5
    width = 500
    ratio = 1

    # Load the image
    face_landmarks = landmark_detector.detect(img=src_img)
    face_obj = ImageObject(img=src_img, landmarks=face_landmarks, type='face')
    # Crop the image
    face_obj.crop(margin, ratio, width)

    glass_img, info = load_accessory_obj(obj_id=accessory_obj_id)

    accessory_obj = ImageObject(img=glass_img, landmarks=info, type='accessory', sub_type='glasses')

    if not face_obj.has_face():
        object_overlay_img = face_obj.data
    else:
        # Face overlay with object
        object_overlay_img = overlay_accessory(face_obj, accessory_obj)

    return object_overlay_img


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
