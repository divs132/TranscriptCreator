# TranscriptCreator
Transcript Creation Project for Video to Text


Usage:


from INNO_TranscriptCreator import TranscriptCreator
abc = INNO_TranscriptCreator.TranscriptCreator(videopath = "S2.mp4")
#Using pyannote-audio pipeline and pretrained model
abc.diarization()

#separates tracks based on above diarization
abc.separatetracks(max_pause=0.5)

#Using the HUBERT CTC model (https://huggingface.co/docs/transformers/model_doc/hubert#transformers.HubertForCTC)
#and https://huggingface.co/patrickvonplaten/wav2vec2-base-100h-with-lm/tree/main/language_model 
#pre trained 4 gram language model and 
#beam decoder CTC from https://github.com/kensho-technologies/pyctcdecode to convert waveform to text
abc.speechtotext()
#Using the Nemo NLP model from https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/punctuation_en_bert
abc.full_punctuate()

#Using the BERT MLM model from https://huggingface.co/bert-base-cased
abc.correct_by_masking()

#Using the neuspell library and swordbert checkpoint from https://github.com/neuspell/neuspell#List-of-neural-models-in-the-toolkit
abc.spellcheck()

abc1 = INNO_TranscriptCreator.TranscriptCreator(videopath = "S3.mp4")
#use below function to create transcript directly without doing diarization
#use create_transcript(diarize=True) to do diarization before converting speech to text
abc1.create_transcript()

