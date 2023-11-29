# import sys,os
# sys.path.append(os.getcwd())
# import VAD_inference

from .inference import *  # noqa


# def main(file,VAD):    
#     audio_file ="VAD/Speechbrain_VAD/vad-crdnn-libriparty"
#     VAD = VAD.from_hparams(source=audio_file, savedir="pretrained_models/vad-crdnn-libriparty")
#     boundaries = VAD.get_speech_segments(file) 
#     print(boundaries)
#     return boundaries



# if __name__ =='__main__':
#     # file = input('audio file:')
#     main("whisper_finetune/TedTalk/ted_test.wav",VAD(activation_th=0.7, deactivation_th=0.25)) #"git_file/whisper_finetune/TedTalk/commonvoice_tw.wav"
