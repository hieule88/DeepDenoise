from flask import Flask, request, render_template, send_file
import flask
import utils 
import torchaudio
import io
import os 
# khoi tao model
global model 

model = None

dns_home = "/storage/hieuld/SpeechEnhancement/DeepDenoise"
#dns_home = "/home/hieule/DeepDenoise"

# khoi tao flask app
app = Flask(__name__)

# Khai báo các route 1 cho API

@app.route('/')
def upload_form():
	return render_template('upload_templates/upload.html', audio_path = 'select file to predict!')

@app.route('/', methods=['POST'])
def get_prediction():
    print('PREDICT MODE')
    dns_home = "/storage/hieuld/SpeechEnhancement/DeepDenoise"
    #dns_home = "/home/hieule/DeepDenoise"
    if request.method == 'POST':
        _file = request.files['file']
        if _file.filename == '':
            return upload_form()
        print('\n\nfile uploaded:',_file.filename) 
        _file.save(os.path.join(dns_home,'static/upload', _file.filename)) 
        print('Write file success!')

        denoise_flt = utils._process(os.path.join(dns_home,'static/upload',_file.filename), model)
        
        #denoise_flt = utils.combine_Out(batch, len_input,model)
        
        torchaudio.save(os.path.join(dns_home, 'denoised','denoised_' + _file.filename), 
                            denoise_flt, sample_rate = 16000)

        print('Done')        
        try :
            return send_file(os.path.join(dns_home, 'denoised','denoised_' + _file.filename), as_attachment=True)
        except Exception as e:
            return str(e)

    #        return render_template('upload_templates/upload.html', audio_path=os.path.join(dns_home, 'static/upload', _file.filename))


if __name__ == "__main__":
    print("App run!")

	#load model
    port = 5000
    host = '0.0.0.0'
    model = utils._load_model()    
    app.run(debug=False, port = port, host= host, threaded=False)

