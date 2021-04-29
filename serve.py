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

        batch, len_input = utils._preprocess(os.path.join(dns_home,'static/upload',_file.filename))
        
        denoise_flt = utils.combine_Out(batch, len_input,model)
        
        torchaudio.save(os.path.join(dns_home, 'denoised', _file.filename +'.wav'), 
                            denoise_flt, sample_rate = 16000)

        print('Done')        
        try :
            return send_file(os.path.join(dns_home, 'denoised', _file.filename +'.wav'))
        except Exception as e:
            return str(e)

    #        return render_template('upload_templates/upload.html', audio_path=os.path.join(dns_home, 'static/upload', _file.filename))

# def _predict():

#     file = request.files["wav"]
#     file.save(os.path.join(dns_home,'upload',file.filename))

#     denoise = []
#     data = {"success": False}  
    
#     batch = utils._preprocess(os.path.join(dns_home,'upload',file.filename))
#     print(batch.shape)
#     exit()
#     for i in range(len(batch)) :
#         denoise[i].append(model(batch[i])[1])
#         torchaudio.save(os.path.join(dns_home, 'denoise_wav_' + i +'.wav'), 
#                         denoise[i].unsqueeze(0), sample_rate = 16000)
#     data["success"] = True
    
#     return 'Done'
if __name__ == "__main__":
    print("App run!")

    #load model
    model = utils._load_model()    
    app.run(debug=True, threaded=False)
#curl -X POST -F wav='/storage/hieuld/SpeechEnhancement/DeepComplexCRN/test.wav' 'http://127.0.0.1:5000/predict'
