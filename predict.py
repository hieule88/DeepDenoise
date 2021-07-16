import sys
import utils 
import torchaudio
import time
from argparse import ArgumentParser

# argparser = ArgumentParser('test_manager.py -ts test_suite')
# argparser.add_argument('-ts', '--test_suite', default=None, help='Ban nho nhap ten test_suite ')
# args = argparser.parse_args()
# test_suite_name = args.test_suite
# runner(test_suite_name)
# model_path = 
# model = utils._load_model(model_path) 
def get_prediction(wav_file, model, save_folder = None):
    start = time.perf_counter()
    denoise_flt = utils._process(wav_file, model)
    # torchaudio.save(os.path.join(dns_home, 'denoised','denoised_' + '.wav'), 
    #                     denoise_flt, sample_rate = 16000)
    
    return denoise_flt

if __name__ == '__main__':
    file = sys.argv[1]
    model_path = sys.argv[2]
    model = utils._load_model(model_path) 
    # save_folder = sys.argv[3]
    get_prediction(file, model)
