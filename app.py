import os
import cv2
from flask import Flask, render_template, request,jsonify
from PIL import Image
from detect import License_plate
import json

UPLOAD_FOLDER = r'static'
app = Flask(__name__,static_folder='static')
app.config['UPLOAD_FOLDER'] = 'static'


with open('config.json') as data_file:
    cred = json.load(data_file)
lp=License_plate(cred['labelsPath'],cred['cfgpath'],cred['wpath'])
port = cred['port']
local_ip = cred['IP_port']
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/show', methods=['POST','GET'])
def show():
    text=""
    if request.method == "GET":
        return "Welcome to License plate detection Demo"
    elif request.method == "POST":
        file = request.files['image']
    try:
        f = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(f)
        img,text = lp.main(app.config['UPLOAD_FOLDER']+"/"+file.filename)
    except:
        pass
    return render_template('show.html',data=text,file_image=file.filename,name=file.filename)

def main():
    app.run(host=str(local_ip),port=int(port),threaded=True)

if __name__ == "__main__":
    main()    
