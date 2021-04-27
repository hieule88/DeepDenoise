from flask import Flask, request
import flask
import utils 
import torchaudio

# khoi tao model
global model 
model = None

# khoi tao flask app
app = Flask(__name__)

# Khai báo các route 1 cho API
@app.route("/", methods=["GET"])
# Khai báo hàm xử lý dữ liệu.
def _hello_world():
	return "Hello world"

# khai bao cac route 2 cho API
@app.route("/predict", methods=["POST"])
# khai bao ham xu ly du lieu
def _predict():
    
    file = request.files["wav"]
    wav = io.BytesIO(file)
    print(wav)
    return file.filename
    '''
    dns_home = "/storage/hieuld/SpeechEnhancement/DeepComplexCRN"
    denoise = []
    data = {"success": False}
    print(request.files["wav"])
    
    
    if request.files.get("wav"):
        batch = _preprocess(request.files["wav"])
        for i in range(len(batch)) :
            denoise[i].append(model(batch[i])[1])
            torchaudio.save(os.path.join(dns_home, 'denoise_wav_' + i +'.wav'), 
                            denoise[i].unsqueeze(0), sample_rate = 16000)
        data["success"] = True
    '''
if __name__ == "__main__":
    print("App run!")
    #load model
    model = utils._load_model()    
    app.run(debug=True, threaded=False)
#curl -X POST -F wav='/storage/hieuld/SpeechEnhancement/DeepComplexCRN/test.wav' 'http://127.0.0.1:5000/predict'
