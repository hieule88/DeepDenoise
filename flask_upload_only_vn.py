import numpy as np
from flask import Flask, request, render_template, url_for
import argparse
import os
import sys
import numpy as np
import timeit
import scipy.io.wavfile as wav_


app = Flask(__name__)

@app.route('/')
def upload_form():
	return render_template('upload_templates/upload.html', audio_path = 'select file to predict!')

@app.route('/', methods=['POST'])
def get_prediction():
    print('PREDICT MODE')
    if request.method == 'POST':
        _file = request.files['file']
        if _file.filename == '':
            return upload_form()
        print('\n\nfile uploaded:',_file.filename)
        _file.save(os.path.join('static/uploaded', _file.filename))
        print('Write file success!')
        start = timeit.default_timer()
	
        end_load = timeit.default_timer()
 
        #predict_grapheme = p2g(predict)
        end_predict = timeit.default_timer()
        print('Load audio time:',end_load-start)
        print('Predict time:',end_predict-end_load)

        return render_template('upload_templates/upload.html', audio_path=os.path.join('static/uploaded', _file.filename))

if __name__ == '__main__':
    #load_model()  # load model at the beginning once only

    app.run(host='localhost', port=5000, debug=True)
