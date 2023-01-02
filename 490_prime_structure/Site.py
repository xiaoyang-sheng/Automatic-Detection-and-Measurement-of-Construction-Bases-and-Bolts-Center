ROUTE = '/'

import os
from flask import Flask, flash, request, redirect, render_template, Response, stream_with_context, request
from werkzeug.utils import secure_filename
import zipfile
import sys
import glob
import time
import socket
import subprocess


def get_host_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    finally:
        s.close()
    return ip


app = Flask(__name__)

app.secret_key = "secret key"
app.config['MAX_CONTENT_LENGTH'] = 128 * 1024 * 1024

# Get current path
path = os.getcwd()

# Allowed extension you can set your own
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route(ROUTE)
def upload_form():
    return render_template('upload.html')


@app.route(ROUTE, methods=['POST'])
def upload_file():
    if request.method == 'POST':
        if 'files[]' not in request.files:
            flash('No file part')
            return redirect(request.url)
        files = request.files.getlist('files[]')
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join(sys.path[0], "process", filename))
        zipFilePath = os.path.join(sys.path[0], "test.zip")
        zipFinalPath = os.path.join(sys.path[0], "Receive", "test.zip")
        zipFile = zipfile.ZipFile(zipFilePath, "w", zipfile.ZIP_DEFLATED)
        files = glob.glob('./process/*')
        for idx, file in enumerate(files):
            zipFile.write(file, str(idx) + '.jpg')
            os.remove(file)
        zipFile.close()
        if os.path.exists(zipFinalPath):
            os.remove(zipFinalPath)
        os.rename(zipFilePath, zipFinalPath)

        def generate():
            yield 'File(s) successfully uploaded and compressed <br> '
            result_name = "coord.txt"
            result_path = os.path.join(sys.path[0], "Receive", result_name)
            start_flag = False
            while not start_flag:
                if os.path.exists(result_path):
                    start_flag = True
                else:
                    time.sleep(2)
                    yield 'processing <br> '
            all_the_text = open(result_path).read()
            os.remove(result_path)
            yield 'Process finish <br>'
            yield all_the_text

        return Response(stream_with_context(generate()))
        # return redirect('/')


if __name__ == "__main__":
    ip = get_host_ip()
    print(ip)
    port = int(input('请输入服务器端口：'))
    print("网址: %s:%d" % (ip, port) + ROUTE)
    images = glob.glob('./process/*')
    for fname in images:
        os.remove(fname)
    images = glob.glob('./Receive/*')
    for fname in images:
        os.remove(fname)
    process1 = subprocess.Popen(['python', 'Server.py'])
    app.run(host=ip, port=port, debug=False, threaded=True)
    process1.wait()
