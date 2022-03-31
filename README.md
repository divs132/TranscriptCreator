# TranscriptCreator
Transcript Creation Project for Video to Text

#Data and Checkpoints available at
https://drive.google.com/drive/folders/12lbjthkwMM3quCmC4vPZciYY6C4A7Y-R?usp=sharing

Conda environment.yml file also available at gdrive folder

Current Word Error Metrics

Data: Video provided by InnoDatatics for transcript generation

Base Model: 33.1

Final Output: 39.4


Dataset:Timit_ASR

Base Model: 2.5

Final Output: 2.1

Datset Ami Corpus(To be revised)

Base Model: 7.1

Final Output: 5.9

Dataset Mozilla Common Voice Dataset:

Base Model: 28.44

Final Output: 28.66


Error Metrics are calculated with punctuated text instead of unpunctuated text.

This is due to the requirements by question answer models using punctuations as context information


Usage:


from INNO_TranscriptCreator import TranscriptCreator

Directory Structure created is

-Main Directory
- Checkpoints
- ---Bert-Base-Cased-MLM
- ---Diarization
- ---HUBERT_Audio_to_Text
- ---punctuation_en_bert
- ---subwordbert-probwordnoise
- ---Wav2Vec2ProcessorwithLM
- ---Voice_Activity_Detection
- Topic Name
- ---Beam Lm
- ---Bert MLM
- ---HuBERT
- ---Punctuation
- ---SpellCheck
- Video Name
- ---Video
- ---Audio
- ---Tracks based on Diarization
- ---Pickle Files for faster reloading

abc = INNO_TranscriptCreator.TranscriptCreator(videopath = "S2.mp4",topic = 'Computer Science)

#Using pyannote-audio pipeline and pretrained model

Pipelines from pyannote-audio: 'Diarization' and 'Voice Activity Detection'

https://github.com/pyannote/pyannote-audio

abc.diarization()
abc.vad_detection()

https://github.com/pyannote/pyannote-audio/tree/develop/pyannote/audio/pipelines

#separates tracks based on above diarization
abc.separatetracks(max_pause=0.5)

Separate Tracks based on Annotation Segments accessed by
output_of_pipeline.itertracks()


#Using the HUBERT CTC model (https://huggingface.co/docs/transformers/model_doc/hubert#transformers.HubertForCTC)

#and https://huggingface.co/patrickvonplaten/wav2vec2-base-100h-with-lm/tree/main/language_model 

#https://arxiv.org/pdf/2106.07447.pdf page 6 gives WER metrics of the needed

#pre trained 4 gram language model and 

#beam decoder CTC from https://github.com/kensho-technologies/pyctcdecode to convert waveform to text

abc.speechtotext()

#Using the Nemo NLP model from https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/punctuation_en_bert

abc.full_punctuate()

#Using the BERT MLM model from https://huggingface.co/bert-base-cased

abc.correct_by_masking()

#Using the neuspell library and swordbert checkpoint from https://github.com/neuspell/neuspell#List-of-neural-models-in-the-toolkit

abc.spellcheck()

abc = INNO_TranscriptCreator.TranscriptCreator(videopath = "S3.mp4")

#use below function to create transcript directly without doing diarization

#use create_transcript(diarize=True) to do diarization before converting speech to text

abc.create_transcript()



Using Batch Creation:

You can use Batch_TranscriptCreator by giving a list of videopaths and corresponding video Ids to create transcripts of a batch of videos

Then call create_transcript() on above batch creator to create the transcripts

This will create the transcripts and will be available in:

self.basic_model for  HuBERT output

self.punctuated_op for Punctuated Output of BERT MLM model

self.mlm_op for BERT MLM model

self.spell_checked_op for Spell Checked Output of MLM model(only if do_spell_check=True in create_transcript())

Using Evaluator for Metrics

This only works for huggingface datasets

just call the class with the dataset name (must be one from ['timit_asr','superb','librispeech_asr'])

after creation of all transcripts

call evaluate_metrics to get a list of all metrics

INNO_FineTuner contains the finetuning functions for the entire pipeline

After calling Fine_Tuner

It will automatically extract completed human reviews from the direbase db

Then it will fine tune the individual topic models and update the models by calling start_finetune on the Fine_Tuner object

It will print the before and after training metrics from the finetuning operation


For Human Review System

https://aispry-174a7.web.app/

Go to this link

Listen to the audio on top

Write in the word you listen at the [MASK] token into the text box

Click Submit




