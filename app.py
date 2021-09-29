
from PIL import Image
from Mask_Detection import MaskDetection
from flask import Flask, jsonify, request, render_template
import os

app=Flask(__name__)
md=MaskDetection()

@app.route('/')
def home():
    return render_template('home.html')
@app.route('/train')
def train():
    md.__init__()
    md.train()
    return render_template('train.html')

@app.route('/evaluate')
def evaluate():
    return render_template('evaluate.html')


@app.route('/predict')
def predict():
    return render_template('predict.html')

@app.route('/live')
def live():
    md.predict_live()
    return render_template('result.html')

@app.route('/img_html')
def img_html():
    return render_template('img.html')


@app.route('/img', methods=['GET', 'POST'])
def img():
    if request.method == 'POST':
        f = request.files['image']
        path = os.path.join('photos_to_predict', f.filename)
        f.save(path)
        md.predict_img(path)
    return  render_template('result.html')
    

@app.route('/eval')
def eval():
    arr= md.evaluate_model()
    return render_template('evaluate.html', traina=round(arr[0], 2),
     testa=round(arr[2], 2), trainl=round(arr[1], 2), testl=round(arr[3],2))


@app.route('/video_html')
def video_html():
    return render_template('video.html')


@app.route('/video', methods=['GET', 'POST'])
def video():
    if request.method == 'POST':
        f = request.files['video']
        path = os.path.join('videos_to_predict', f.filename)
        f.save(path)
        md.predict_video(path)


    return render_template('result.html')


app.run()
