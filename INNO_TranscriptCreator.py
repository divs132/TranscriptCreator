import os
import shutil
from typing import Tuple
import pickle

import soundfile as sf
import math
import torch
import torchaudio
from transformers import AutoProcessor, HubertForCTC

import regex as re
import glob
from nltk.tokenize import sent_tokenize

from itertools import zip_longest 
from transformers import BertTokenizer, BertForMaskedLM
from transformers import pipeline
import nltk
from itertools import repeat
import time
from datasets import load_dataset
import jiwer
from pydub import AudioSegment
import json   
        



class TranscriptCreator_Evaluator:
    def __init__(self,dataset_name = 'timit_asr',
        config_name='', video_topic = 'Computer Science',
        baserootpath = '/home/divyansh/Documents/Capstone',delete_folders = True,
        diarize = False,do_spell_check = False,max_test=0,
        minimize_size=False,
        upload_to_db=False,load_models = True):
        if not dataset_name in ['timit_asr','superb','librispeech_asr','ami','common']:
            raise NotImplementedError("Please use one of the specified test dbs for the evaluator:'timit_asr','superb','librispeech_asr','ami','common'",)

        if dataset_name == 'timit_asr':
            dbset = load_dataset('timit_asr',split='test')
        elif dataset_name == 'superb':
            dbset = load_dataset('superb','asr',split='test')
        elif dataset_name == 'ami':
            dbset = load_dataset('ami','headset-multi',split='test')
        elif dataset_name == 'common':
            dbset = load_dataset('csv', data_files=['/mnt/spare2/cv-corpus-8.0-2022-01-19-en.tar/cv-corpus-8.0-2022-01-19-en/cv-corpus-8.0-2022-01-19/en/test.csv'],split='train')
        else:
            dbset = load_dataset('librispeech_asr','clean',split='test')


        if not dataset_name=='common':
            vpaths = list(i['audio']['path'] for i in dbset)
            if dataset_name=='ami':
                gtruths = list(" ".join(i['words']) for i in dbset)
            else:
                gtruths = list(i['text'] for i in dbset)
            v_ids = list(range(len(vpaths)))
            v_ids = [str(a)+'_' + dataset_name for a in v_ids]
            
        else:
            vpaths=[]
            gtruths=[]
            v_ids=[]

            cmn_base_path = '/mnt/spare2/cv-corpus-8.0-2022-01-19-en.tar/cv-corpus-8.0-2022-01-19-en/cv-corpus-8.0-2022-01-19/en/clips'
            for i in range(0,dbset.num_rows-1):
                path = os.path.join(cmn_base_path,dbset[i]['path'])
                if os.path.exists(path) and not dbset[i]['sentence'] == None:
                    vpaths.append(path)
                    gtruths.append(dbset[i]['sentence'])
                    v_ids.append(str(i) + '_common')




        self.do_spell_check = do_spell_check
        if not max_test ==0:
            vpaths =vpaths[0:max_test]
            gtruths =gtruths[0:max_test]
            v_ids =v_ids[0:max_test]

        self.batcher = Batch_TranscriptCreator(videopaths = vpaths,video_ids = v_ids,groundtruths = gtruths,diarize=diarize,delete_folders=delete_folders,video_topic=video_topic,baserootpath=baserootpath)
        self.batcher.create_transcript(do_spell_check=self.do_spell_check,upload_to_db=upload_to_db,load_model = load_models)
        self.metrics = self.evaluate_metrics(self.batcher)
        

    def evaluate_metrics(self,btr,transformation = ''):
        transformationcstm = jiwer.Compose([
            jiwer.ToLowerCase(),
            jiwer.RemovePunctuation(),
            jiwer.RemoveWhiteSpace(replace_by_space=True),
            jiwer.RemoveMultipleSpaces(),
            jiwer.ReduceToListOfListOfWords(word_delimiter=" ")
        ]) 

        if transformation=='':
            transformation=transformationcstm


        self.wer1 = jiwer.wer(btr.ground_truth, btr.basic_model,truth_transform=transformation,hypothesis_transform=transformation)
        self.mer1 = jiwer.mer(btr.ground_truth, btr.basic_model,truth_transform=transformation,hypothesis_transform=transformation)
        self.wil1 = jiwer.wil(btr.ground_truth, btr.basic_model,truth_transform=transformation,hypothesis_transform=transformation)

        #measures1 = jiwer.compute_measures(ground_truth, hypothesis)

        self.wer2 = jiwer.wer(btr.ground_truth, btr.mlm_op,truth_transform=transformation,hypothesis_transform=transformation)
        self.mer2 = jiwer.mer(btr.ground_truth, btr.mlm_op,truth_transform=transformation,hypothesis_transform=transformation)
        self.wil2 = jiwer.wil(btr.ground_truth, btr.mlm_op,truth_transform=transformation,hypothesis_transform=transformation)

        #measures2 = jiwer.compute_measures(ground_truth, hypothesis)

        self.wer3 = jiwer.wer(btr.ground_truth, btr.punctuated_op,truth_transform=transformation,hypothesis_transform=transformation)
        self.mer3 = jiwer.mer(btr.ground_truth, btr.punctuated_op,truth_transform=transformation,hypothesis_transform=transformation)
        self.wil3 = jiwer.wil(btr.ground_truth, btr.punctuated_op,truth_transform=transformation,hypothesis_transform=transformation)

        #measures3 = jiwer.compute_measures(ground_truth, hypothesis)
        if self.do_spell_check:
            self.wer4 = jiwer.wer(btr.ground_truth, btr.spell_checked_op,truth_transform=transformation,hypothesis_transform=transformation)
            self.mer4 = jiwer.mer(btr.ground_truth, btr.spell_checked_op,truth_transform=transformation,hypothesis_transform=transformation)
            self.wil4 = jiwer.wil(btr.ground_truth, btr.spell_checked_op,truth_transform=transformation,hypothesis_transform=transformation)

            #measures4 = jiwer.compute_measures(ground_truth, hypothesis)

            return dict(Hubert_WER = self.wer1,Punctuated_WER = self.wer3,MLM_WER = self.wer2,SpellChecked_WER = self.wer4)
        else:
            return dict(Hubert_WER = self.wer1,Punctuated_WER = self.wer3,MLM_WER = self.wer2)




class Batch_TranscriptCreator():
    def __init__(self,videopaths,video_ids,video_topic = 'Computer Science',baserootpath = '/home/divyansh/Documents/Capstone',delete_folders = True,diarize = False,groundtruths = [],upload_to_db=True):

        print('Converting ' + str(len(videopaths)) + ' Tracks')
        self.base_path = baserootpath
        self.initialize_env()
        self.initialize_topic(video_topic)
        self.ground_truth = groundtruths
        self.basic_model = []
        self.mlm_op = []
        self.punctuated_op = []
        self.spell_checked_op=[]
        self.do_diarization=diarize
        self.videopaths = videopaths
        self.video_topic = video_topic
        self.video_ids = video_ids

    
        


    def create_transcript(self,do_spell_check=False,minimize_size=False,upload_to_db=True,load_model=True):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.do_diarization:
            from pyannote.audio import Pipeline
            if load_model:
                diarize_model = Pipeline.from_pretrained(self.diarize_model_path)
            else:
                diarize_model= 1
            i=0
            for videopath in self.videopaths:
                temp_tcr = TranscriptCreator(videopath,video_topic=self.video_topic,video_id=self.video_ids[i],minimize_size=minimize_size,upload_to_db=upload_to_db)
                temp_tcr.diarization(diarize_model)
                temp_tcr.separatetracks(temp_tcr.diarized)
                i=i+1
                del temp_tcr

            #print('Unloading Model Cuda Memory Allocated:' , torch.cuda.memory_allocated(0))
            del diarize_model
            torch.cuda.empty_cache()
            #print('Unloaded Model Cuda Memory Allocated:' , torch.cuda.memory_allocated(0))
            
        else:
            from pyannote.audio import Pipeline
            if load_model:
                vad_model = Pipeline.from_pretrained(self.vad_model_path)
            else:
                vad_model= 1
            
            i=0
            for videopath in self.videopaths:
                temp_tcr = TranscriptCreator(videopath,video_topic=self.video_topic,video_id=self.video_ids[i],minimize_size=minimize_size,upload_to_db=upload_to_db)
                temp_tcr.vad_detection(vad_model)
                temp_tcr.separatetracks(temp_tcr.vad)
                i=i+1
                del temp_tcr

            #print('Unloading Model Cuda Memory Allocated:' , torch.cuda.memory_allocated(0))
            del vad_model
            torch.cuda.empty_cache()
            #print('Unloaded Model Cuda Memory Allocated:' , torch.cuda.memory_allocated(0))

               
        if load_model:
            transcribe_model = HubertForCTC.from_pretrained(self.audio_to_text_model_path).to(device)
            decoder = AutoProcessor.from_pretrained(self.audio_to_text_tokenizer_path)
        else:
            transcribe_model = 1
            decoder = 1


        i=0
        for videopath in self.videopaths:
            temp_tcr = TranscriptCreator(videopath,video_topic=self.video_topic,video_id=self.video_ids[i],minimize_size=minimize_size,upload_to_db=upload_to_db)
            self.basic_model.append(temp_tcr.speechtotext(transcribe_model,decoder))
            i=i+1
            del temp_tcr
        #print('Unloading Model Cuda Memory Allocated:' , torch.cuda.memory_allocated(0))
        del transcribe_model,decoder
        torch.cuda.empty_cache()
        #print('Unloaded Model Cuda Memory Allocated:' , torch.cuda.memory_allocated(0))
        from nemo.utils.exp_manager import exp_manager
        from nemo.collections import nlp as nemo_nlp
        if load_model:
            punctuate_model = nemo_nlp.models.PunctuationCapitalizationModel.restore_from(self.punctuate_model_path)
        else:
            punctuate_model = 1
        
        i=0
        for videopath in self.videopaths:
            temp_tcr = TranscriptCreator(videopath,video_topic=self.video_topic,video_id=self.video_ids[i],minimize_size=minimize_size,upload_to_db=upload_to_db)
            self.punctuated_op.append(temp_tcr.full_punctuate(punctuate_model))
            i=i+1
            del temp_tcr
        #print('Unloading Model Cuda Memory Allocated:' , torch.cuda.memory_allocated(0))
        #del punctuate_model
        torch.cuda.empty_cache()
        #print('Unloaded Model Cuda Memory Allocated:' , torch.cuda.memory_allocated(0))

        if load_model:
            tokenizer = BertTokenizer.from_pretrained(self.mlm_model_path)
            model = BertForMaskedLM.from_pretrained(self.mlm_model_path)
            unmasker = pipeline(task = 'fill-mask', model=model,tokenizer = tokenizer,device=-1,top_k=1)
        else:
            tokenizer = 1
            model = 1
            unmasker = 1
        
        i=0
        for videopath in self.videopaths:
            temp_tcr = TranscriptCreator(videopath,video_topic=self.video_topic,video_id=self.video_ids[i],minimize_size=minimize_size,upload_to_db=upload_to_db)
            self.mlm_op.append(temp_tcr.correct_by_masking(unmasker,punctuate_model_path= punctuate_model))
            i=i+1
            del temp_tcr
        #print('Unloading Model Cuda Memory Allocated:' , torch.cuda.memory_allocated(0))
        del tokenizer,model,unmasker,punctuate_model
        torch.cuda.empty_cache()
        #print('Unloaded Model Cuda Memory Allocated:' , torch.cuda.memory_allocated(0))
        


        if do_spell_check:
            from neuspell import available_checkers, BertChecker
            if load_model:
                spell_checker_model = BertChecker(device="cuda")
                spell_checker_model._from_pretrained(ckpt_path = self.spellcheck_model_path,vocab_path = os.path.join(self.spellcheck_model_path,'vocab.pkl'))
            else:
                spell_checker_model = 1
            
            i=0
            for videopath in self.videopaths:
                temp_tcr = TranscriptCreator(videopath,video_topic=self.video_topic,video_id=self.video_ids[i],minimize_size=minimize_size,upload_to_db=upload_to_db)
                self.spell_checked_op.append(temp_tcr.spellcheck(query = '' ,spellcheck_model_path= spell_checker_model))
                i=i+1
                del temp_tcr
            #print('Unloading Model Cuda Memory Allocated:' , torch.cuda.memory_allocated(0))
            del spell_checker_model
            torch.cuda.empty_cache()
            #print('Unloaded Model Cuda Memory Allocated:' , torch.cuda.memory_allocated(0))



    def initialize_env(self):
        if not os.path.exists(self.base_path):
            raise NotImplementedError("Do not currently support completely new environment")
        
        if not os.path.exists(os.path.join(self.base_path,'Diarization/pipeline_checkpoint')):
            shutil.copytree(os.path.join(self.base_path,'Checkpoints/Diarization'), os.path.join(self.base_path,'Diarization'))

        if not os.path.exists(os.path.join(self.base_path,'Voice_Activity_Detection/pipeline_checkpoint')):
            shutil.copytree(os.path.join(self.base_path,'Checkpoints/Voice_Activity_Detection'), os.path.join(self.base_path,'Voice_Activity_Detection'))
        
        self.diarize_model_path = os.path.join(self.base_path,'Diarization')
        self.diarize_model_path = os.path.join(self.diarize_model_path,'pipeline_checkpoint')
        self.vad_model_path = os.path.join(self.base_path,'Voice_Activity_Detection')
        self.vad_model_path = os.path.join(self.vad_model_path,'pipeline_checkpoint')

    def initialize_topic(self,topicname):
        
        self.topic_path = os.path.join(self.base_path,topicname)
        self.audio_to_text_model_path = os.path.join(self.topic_path,'HuBERT')
        self.audio_to_text_tokenizer_path = os.path.join(self.topic_path,'Beam_LM')
        self.mlm_model_path = os.path.join(self.topic_path,'BERT_MLM')
        self.spellcheck_model_path = os.path.join(self.topic_path,'SpellCheck')
        self.punctuate_model_path = os.path.join(self.topic_path,'Punctuation','punctuation_en_bert.nemo')   

        if not os.path.exists(self.topic_path):
            os.mkdir(self.topic_path)
            shutil.copytree(os.path.join(self.base_path,'Checkpoints/HuBERT_Audio_to_Text'), os.path.join(self.topic_path,'HuBERT'))
            shutil.copytree(os.path.join(self.base_path,'Checkpoints/Wav2Vec2processorwithLM'), os.path.join(self.topic_path,'Beam_LM'))
            shutil.copytree(os.path.join(self.base_path,'Checkpoints/Bert-base-Cased-MLM'), os.path.join(self.topic_path,'BERT_MLM'))
            shutil.copytree(os.path.join(self.base_path,'Checkpoints/subwordbert-probwordnoise'), os.path.join(self.topic_path,'SpellCheck'))
            shutil.copytree(os.path.join(self.base_path,'Checkpoints/punctuation_en_bert'), os.path.join(self.topic_path,'Punctuation'))
            while not os.path.exists(self.punctuate_model_path):
                print('Initializing Topic Models')
                time.sleep(1000)









class TranscriptCreator():
    def __init__(self,videopath,savedirectory='',video_topic='Computer Science',baserootpath = '/home/divyansh/Documents/Capstone',video_id = '',minimize_size=False,make_new=False,upload_to_db = True):
        self.base_path=baserootpath
        self.original_videopath = videopath
        self.upload_to_db = upload_to_db
        vdname = os.path.basename(videopath).replace(".","")
        if not video_id=='':
            vdname = str(video_id) +'_' + vdname

        self.video_id = vdname

        dirname = os.path.join(self.base_path, 'Videos')
        dirname = os.path.join(dirname, vdname)
        
        if(savedirectory==''):
            self.directory = dirname
        else:
            self.directory = savedirectory
            
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)
        
        self.initialize_env()
        self.initialize_topic(video_topic)
        self.readinfofromfiles(make_new)
        
        if not minimize_size:
            self.videopath = os.path.join(self.directory,os.path.basename(videopath))  

            if not os.path.exists(self.videopath):
                shutil.copyfile(self.original_videopath, self.videopath)
        else:
            self.videopath = videopath

        self.audiopath = os.path.join(self.directory,"audio.wav")

        
        
        
        
        if not os.path.exists(self.audiopath):
            if '.wav' in self.videopath.lower():
                shutil.copyfile(self.videopath, self.audiopath)
            elif '.mp3' in self.videopath.lower():
                sound = AudioSegment.from_mp3(self.videopath)
                sound.export(self.audiopath, format="wav")
            else:
                #print(self.videopath)
                import moviepy.editor as mp
                clip = mp.VideoFileClip(self.videopath)
                clip.audio.write_audiofile(self.audiopath)
                

    def readpkl(self,filepath,usedir=True):
        
        if usedir:
            fpath = os.path.join(self.directory,filepath)
        else:
            fpath = filepath
        
        if os.path.exists(fpath):
            #print(fpath)
            with open(fpath, 'rb') as inp:
                return pickle.load(inp)
        else:
            return None


    def writepkl(self,object,filepath,usedir=True):
        if usedir:
            fpath = os.path.join(self.directory,filepath)
        else:
            fpath = filepath

        if os.path.exists(fpath):
            os.remove(fpath)
        with open(fpath, 'wb') as outp:
            pickle.dump(object, outp, pickle.HIGHEST_PROTOCOL)




    def readinfofromfiles(self,make_new=False):
        if not make_new:
            self.vad = self.readpkl(self.vad_path)
            self.diarize=self.readpkl(self.diarize_path)
            self.speaker_dict = self.readpkl(self.speaker_dict_path)
            self.transcribe_op = self.readpkl(self.transcribe_op_path)
            self.transcribe_full = self.readpkl(self.transcribe_full_path)
            self.punctuate_full = self.readpkl(self.punctuate_full_path)
            self.word_op = self.readpkl(self.word_op_path)
            self.errors_mlm = self.readpkl(self.errors_mlm_path)
            self.mlm_full = self.readpkl(self.mlm_full_path)
        else:
            self.vad = None
            self.diarize=None
            self.speaker_dict = None
            self.transcribe_op = None
            self.transcribe_full = None
            self.punctuate_full = None
            self.word_op = None
            self.errors_mlm = None  



    def load_diarize_model(self,model_path):
        from pyannote.audio import Pipeline
        if not hasattr(self,'diarize'):
            if type(model_path)==str and model_path!='':
                self.diarize_model = Pipeline.from_pretrained(model_path)
            elif not type(model_path)==str:
                self.diarize_model = model_path
            else:
                self.diarize_model = Pipeline.from_pretrained(self.diarize_model_path)
 
    def load_vad_model(self,model_path):
        from pyannote.audio import Pipeline
        if not hasattr(self,'vad_model'):
            if type(model_path)==str and model_path!='':
                self.vad_model = Pipeline.from_pretrained(model_path)
            elif not type(model_path)==str:
                self.vad_model = model_path
            else:
                self.vad_model = Pipeline.from_pretrained(self.vad_model_path)


    def load_transcribe_model(self,model_path):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if not hasattr(self,'transcribe_model'):
            if type(model_path)==str and model_path!='':
                self.transcribe_model = HubertForCTC.from_pretrained(model_path).to(device)
            elif not type(model_path)==str:
                self.transcribe_model = model_path
            else:
                self.transcribe_model = HubertForCTC.from_pretrained(self.audio_to_text_model_path).to(device)

    def load_transcribe_decoder(self,model_path):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if not hasattr(self,'decoder'):
            if type(model_path)==str and model_path!='':
                self.decoder = HubertForCTC.from_pretrained(model_path).to(device)
            elif not type(model_path)==str:
                self.decoder = model_path
            else:
                self.decoder = AutoProcessor.from_pretrained(self.audio_to_text_tokenizer_path)
  
    def load_punctuate_model(self,model_path):
        from nemo.utils.exp_manager import exp_manager
        from nemo.collections import nlp as nemo_nlp
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if not hasattr(self,'punctuate_model'):
            if type(model_path)==str and model_path!='':
                self.punctuate_model = nemo_nlp.models.PunctuationCapitalizationModel.restore_from(model_path)
            elif not type(model_path)==str:
                self.punctuate_model = model_path
            else:
                self.punctuate_model = nemo_nlp.models.PunctuationCapitalizationModel.restore_from(self.punctuate_model_path)

    def load_spell_checker_model(self,model_path):
        from neuspell import available_checkers, BertChecker
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if not hasattr(self,'spell_checker_model'):
            if type(model_path)==str and model_path!='':
                self.spell_checker_model = BertChecker(device=device)._from_pretrained(ckpt_path = model_path,vocab_path = os.path.join(model_path,'vocab.pkl'))
            elif not type(model_path)==str:
                self.spell_checker_model = model_path
            else:
                self.spell_checker_model = BertChecker(device=device)._from_pretrained(ckpt_path = self.spellcheck_model_path,vocab_path = os.path.join(self.spellcheck_model_path,'vocab.pkl'))
    
    def load_mlm_pipeline(self,model_path):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if not hasattr(self,'mlm_unmasker'):
            if type(model_path)==str and model_path!='':
                self.mlm_tokenizer = BertTokenizer.from_pretrained(model_path)
                self.mlm_model = BertForMaskedLM.from_pretrained(model_path).to(device)
                self.mlm_unmasker = pipeline(task = 'fill-mask', model=self.mlm_model,tokenizer = self.mlm_tokenizer,device=-1,top_k=1)
            elif not type(model_path)==str:
                self.mlm_unmasker = model_path
            else:
                self.mlm_tokenizer = BertTokenizer.from_pretrained(self.mlm_model_path)
                self.mlm_model = BertForMaskedLM.from_pretrained(self.mlm_model_path)
                self.mlm_unmasker = pipeline(task = 'fill-mask', model=self.mlm_model,tokenizer = self.mlm_tokenizer,device=-1,top_k=1)




    def create_transcript(self,diarize=False,delete_folder = False):
        if diarize:
            self.diarization()
            torch.cuda.empty_cache()
            self.separatetracks(self.diarized)
            torch.cuda.empty_cache()
            self.speechtotext()
            torch.cuda.empty_cache()
            self.full_punctuate()
            torch.cuda.empty_cache()
            self.correct_by_masking()
            torch.cuda.empty_cache()
            #self.spellcheck()
            torch.cuda.empty_cache()
        else:
            self.vad_detection()
            torch.cuda.empty_cache()
            self.separatetracks(self.vad)
            self.speechtotext()
            torch.cuda.empty_cache()
            self.full_punctuate()
            torch.cuda.empty_cache()
            self.correct_by_masking()
            torch.cuda.empty_cache()
            #self.spellcheck()
            torch.cuda.empty_cache()
        
        if delete_folder:
            os.removedirs(self.directory)


        return dict(Basic_Transcript = self.transcribe_op,
            Corrected_Transcript = self.mlm_full,
            Full_Punctuated = self.punctuate_full )


    def initialize_env(self):
        self.transcript_path = os.path.join(self.directory,"transcript.pkl")
        self.spell_checked_path_1 = os.path.join(self.directory,'spell_checked_1.pkl')
        self.full_punctuate_path = os.path.join(self.directory,"full_punctuate.pkl")
        self.vad_path = os.path.join(self.directory,'vad.pkl')
        self.diarize_path = os.path.join(self.directory,'diarized.pkl')
        self.corrected_path = os.path.join(self.directory,"corrected.pkl")
        self.human_path = os.path.join(self.directory,"human.pkl")
        self.speaker_dict_path = os.path.join(self.directory,"speaker_dict.pkl")
        self.transcribe_op_path = os.path.join(self.directory,"transcribe_op.pkl")
        self.transcribe_full_path = os.path.join(self.directory,"transcribe_full.pkl")
        self.word_op_path = os.path.join(self.directory,"words_op.pkl")
        self.punctuate_full_path = os.path.join(self.directory,"punctuate_full.pkl")
        self.errors_mlm_path = os.path.join(self.directory,"errors_mlm.pkl")
        self.mlm_full_path = os.path.join(self.directory,'mlm_full.pkl')


        if not os.path.exists(self.base_path):
            raise NotImplementedError("Do not currently support completely new environment")
        
        if not os.path.exists(os.path.join(self.base_path,'Diarization/pipeline_checkpoint')):
            shutil.copytree(os.path.join(self.base_path,'Checkpoints/Diarization'), os.path.join(self.base_path,'Diarization'))

        if not os.path.exists(os.path.join(self.base_path,'Voice_Activity_Detection/pipeline_checkpoint')):
            shutil.copytree(os.path.join(self.base_path,'Checkpoints/Voice_Activity_Detection'), os.path.join(self.base_path,'Voice_Activity_Detection'))
        
        self.diarize_model_path = os.path.join(self.base_path,'Diarization')
        self.diarize_model_path = os.path.join(self.diarize_model_path,'pipeline_checkpoint')
        self.vad_model_path = os.path.join(self.base_path,'Voice_Activity_Detection')
        self.vad_model_path = os.path.join(self.vad_model_path,'pipeline_checkpoint')

    def initialize_topic(self,topicname):

        self.topic_path = os.path.join(self.base_path,topicname)
        self.audio_to_text_model_path = os.path.join(self.topic_path,'HuBERT')
        self.audio_to_text_tokenizer_path = os.path.join(self.topic_path,'Beam_LM')
        self.mlm_model_path = os.path.join(self.topic_path,'BERT_MLM')
        self.spellcheck_model_path = os.path.join(self.topic_path,'SpellCheck')
        self.punctuate_model_path = os.path.join(self.topic_path,'Punctuation','punctuation_en_bert.nemo')   

        if not os.path.exists(self.topic_path):
            os.mkdir(self.topic_path)
            shutil.copytree(os.path.join(self.base_path,'Checkpoints/HuBERT_Audio_to_Text'), os.path.join(self.topic_path,'HuBERT'))
            shutil.copytree(os.path.join(self.base_path,'Checkpoints/Wav2Vec2processorwithLM'), os.path.join(self.topic_path,'Beam_LM'))
            shutil.copytree(os.path.join(self.base_path,'Checkpoints/Bert-base-Cased-MLM'), os.path.join(self.topic_path,'BERT_MLM'))
            shutil.copytree(os.path.join(self.base_path,'Checkpoints/subwordbert-probwordnoise'), os.path.join(self.topic_path,'SpellCheck'))
            shutil.copytree(os.path.join(self.base_path,'Checkpoints/punctuation_en_bert'), os.path.join(self.topic_path,'Punctuation'))
            while not os.path.exists(self.punctuate_model_path):
                print('Initializing Topic Models')
                time.sleep(1000)

    def diarization(self,model = ''):
        
        if not os.path.exists(self.diarize_path):
            self.load_diarize_model(model_path=model)
            #print("Doing Diarization")
            self.diarized = self.diarize_model(self.audiopath)
            self.writepkl(self.diarized,self.diarize_path)
            del self.diarize_model

    def vad_detection(self,model = ''):
        
        if not os.path.exists(self.vad_path):
            self.load_vad_model(model)
            #print("Doing Voice Activity Detection")
            self.vad = self.vad_model(self.audiopath)
            self.writepkl(self.vad,self.vad_path)
            del self.vad_model
   
    def separatetracks(self,segment_input,max_pause = 0.5, max_speakers = 1,maxduration=60,make_new = False):
        
        if make_new:
            files = glob.glob(self.directory+ '/*.wav')
            for f in files:
                if not os.path.basename(f) == 'audio.wav':
                    os.remove(f)

        if self.speaker_dict == None and not make_new:

            self.speaker_dict={}
            self.tracks=list()
            data, fs = sf.read(self.audiopath)
            for speech_turn, track, speaker in segment_input.itertracks(yield_label=True):
                if speaker in self.speaker_dict:
                    self.speaker_dict[speaker]['duration'] = self.speaker_dict[speaker]['duration'] + speech_turn.end - speech_turn.start
                    if self.speaker_dict[speaker]['tracklist'][-1]['end'] + max_pause >= speech_turn.start:
                        if speech_turn.end - self.speaker_dict[speaker]['tracklist'][-1]['start']>maxduration:
                            self.speaker_dict[speaker]['tracklist'].append({'start':speech_turn.start,'end':speech_turn.end})
                            self.speaker_dict[speaker]['num_tracks'] += 1
                        else:
                            self.speaker_dict[speaker]['tracklist'][-1]['end']= speech_turn.end
                    else:
                        if speech_turn.end - speech_turn.start>maxduration:
                            self.speaker_dict[speaker]['tracklist'].append({'start':speech_turn.start,'end':speech_turn.start + maxduration})
                        else:
                            self.speaker_dict[speaker]['tracklist'].append({'start':speech_turn.start,'end':speech_turn.end})
                        #self.speaker_dict[speaker]['tracklist'].append({'start':speech_turn.start,'end':speech_turn.end})
                        self.speaker_dict[speaker]['num_tracks'] += 1
            
                else:
                    self.speaker_dict[speaker] = {}
                    
                    self.speaker_dict[speaker]['tracklist']=list()
                    if speech_turn.end - speech_turn.start>maxduration:
                        self.speaker_dict[speaker]['tracklist'].append({'start':speech_turn.start,'end':speech_turn.start + maxduration})
                        self.speaker_dict[speaker]['tracklist'].append({'start':speech_turn.start + maxduration,'end':speech_turn.end})
                    else:
                        self.speaker_dict[speaker]['tracklist'].append({'start':speech_turn.start,'end':speech_turn.end})
                    
                    
                    self.speaker_dict[speaker]['duration']= speech_turn.end - speech_turn.start
                    self.speaker_dict[speaker]['num_tracks']=1
                    
            
            
            res = dict(sorted(self.speaker_dict.items(), key=lambda item: item[1]['duration'], reverse = True)[:max_speakers])
            self.speaker_dict = res
            self.writepkl(self.speaker_dict,self.speaker_dict_path)
        
        
            for speaker in self.speaker_dict.items():
                for i in range(len(speaker[1]['tracklist'])):
                    fname = str(speaker[0]) + 'track_'+ str(i)+'.wav'
                    #print(fname)
                    fname = os.path.join(self.directory,fname)
                    self.tracks.append(fname)
                    startPoint = math.floor(speaker[1]['tracklist'][i]['start'])*fs
                    endPoint = math.ceil(speaker[1]['tracklist'][i]['end'])*fs
                    if not os.path.exists(fname):
                        sf.write(fname, data[startPoint:endPoint], fs)             
            
    def speechtotext(self,wav2vec2typemodel='',processorwithlmhead='',make_new=False):
        if not hasattr(self,'tracks'):
            self.tracks = glob.glob(self.directory+ '/*.wav')
            self.tracks.remove(os.path.join(self.directory,'audio.wav'))
            if self.videopath in self.tracks:
                self.tracks.remove(self.videopath)
      
        if self.transcribe_op==None or make_new:
            self.full_transcript = ''
            self.load_transcribe_model(wav2vec2typemodel)
            self.load_transcribe_decoder(processorwithlmhead)
            self.transcribe_op = {}
            self.transcribe_full=''
        
            #print("Model Loaded")
            
            #return '0'
        
            for track in self.tracks:
                trackdictname = os.path.basename(track)
                self.transcribe_op[trackdictname]={}
                #print("Loading Track:" + track)
                waveform, sample_rate = torchaudio.load(track)
                #waveform = waveform.to(device)
                if sample_rate != 16000:
                    waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
                    
            
                with torch.no_grad():
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    logits = self.transcribe_model(waveform.to(device)).logits[0].cpu().numpy()
                    ratio = waveform.size(1)/logits.shape[0]
                    del waveform
                    transcript = self.decoder.decode(logits,output_word_offsets = False)['text']
                    sentence_probs = self.decoder.decoder.decode_beams(logits)
                    nsentence_probs = []
                    for probs in sentence_probs:
                        nsentence_probs.append(dict(word_offsets=probs[2],logit = probs[3],lm_score = probs[4]))
                    #print(transcript)
                    
                #print(nsentence_probs)
                self.transcribe_op[trackdictname]['op_beams'] = nsentence_probs
                self.transcribe_op[trackdictname]['logits'] = logits
                self.transcribe_op[trackdictname]['ratio'] = ratio
                del logits
                self.transcribe_op[trackdictname]['transcript'] = transcript
                if self.transcribe_full=='':
                    self.transcribe_full = transcript
                else:
                    self.transcribe_full = self.transcribe_full + ' ' + transcript
            
                
                
                torch.cuda.empty_cache()
                #break
            
            del self.transcribe_model
            del self.decoder
            torch.cuda.empty_cache()
            self.writepkl(self.transcribe_op,self.transcribe_op_path)
            self.writepkl(self.transcribe_full,self.transcribe_full_path)
            self.create_words_op()



            
            
        return self.transcribe_full


    def create_words_op(self,query=''):

        if self.word_op==None:
            self.word_op = []
            if query == '':
                tr_op = self.transcribe_op
            else:
                tr_op = query
            for track in tr_op:
                words = []
                logit_scores=[]
                beams = tr_op[track]['op_beams']
                offsets = beams[0]['word_offsets']
                for i in beams:
                    if len(i['word_offsets'])==len(beams[0]['word_offsets']):
                        words.append([j[0] for j in i['word_offsets']])
                        logit_scores.append(i['logit'])
                l = list(zip_longest(*words))
                m = list(torch.nn.Softmax(dim=-1)(torch.Tensor(logit_scores)).numpy())
                #word_certain=[]
                #probable_words = []
                for mi,prob_words in enumerate(l):
                    targets=[]
                    certain = 0
                                
                    if len(set(prob_words))==1:
                        #word_certain.append(1)
                        #probable_words.append(prob_words[0])
                        self.word_op.append({'track':track,'word':offsets[mi][0],
                                        'start':offsets[mi][1][0],'end':offsets[mi][1][1],
                                        'certain':1,'prob_words':prob_words[0]})
                        #print(prob_words[0])
                        continue
                    targets.append(prob_words[0])
                    for i, wrd in enumerate(prob_words):
                        if wrd == prob_words[0]:
                            certain = certain + m[i]
                        else:
                            #print(prob_words[0],wrd)
                            if not wrd in targets:
                                targets.append(wrd)
                    self.word_op.append({'track':track,'word':offsets[mi][0],
                                    'start':offsets[mi][1][0],'end':offsets[mi][1][1],
                                    'certain':certain,'prob_words':targets})          

            self.writepkl(self.word_op,self.word_op_path)


    def return_punctuated(self,query,model_path =''):
        
        self.load_punctuate_model(model_path=model_path)
        with torch.no_grad():
            inference = self.punctuate_model.add_punctuation_capitalization(query,max_seq_length=128,step=8,margin=16,batch_size=32)

        return inference


    
    def full_punctuate(self,model_path= ''):

        if self.punctuate_full==None:

            self.load_punctuate_model(model_path)
            #print("loading form pretrained")
            query = [self.transcribe_full.lower()]

            self.punctuate_full = self.return_punctuated(query,model_path)[0]
            torch.cuda.empty_cache()    
            self.writepkl(self.punctuate_full,self.punctuate_full_path)
        return self.punctuate_full

    def spellcheck(self,query = '',spellcheck_model_path='',EACH_SENTENCE=True):
        

        if os.path.exists(self.spell_checked_path_1):
            if query == '':
                query = self.full_transcript_corrected
            

            full_corrected=''
            self.load_spell_checker_model(spellcheck_model_path)

            if EACH_SENTENCE:
                query = sent_tokenize(query)
            else:
                query = [query]
            
            self.spell_checked_1 = self.spell_checker_model.correct_strings(query)
            with open(self.spell_checked_path_1, 'wb') as pickle_file:
                pickle.dump(self.spell_checked_1, pickle_file, pickle.HIGHEST_PROTOCOL)

        return self.spell_checked_1[0]

    def punctuate(self,model_path= ''):
        if model_path=='' and type(model_path)==str:
            model_path=self.punctuate_model_path
        self.punctuate_path = os.path.join(self.directory,"punctuate.pkl")
        
        if os.path.exists(self.punctuate_path):
            print("PKL File Found")
            with open(self.punctuate_path, 'rb') as inp:
                self.transcript = pickle.load(inp)
        else:
            self.load_punctuate_model()

            
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.punctuate_model = self.punctuate_model.to(device)
            
            for track in self.transcript.items():
                #print(track)
                query = [track[1]['transcript'].lower()]
                inference = self.punctuate_model.add_punctuation_capitalization(query)
                #print(inference)
                #print(track[0])
                self.transcript[track[0]]['Punctuated']=inference
                #print(abtranscript[track[0]])
                torch.cuda.empty_cache()
                #break
            torch.cuda.empty_cache()    
            with open(self.punctuate_path, 'wb') as pickle_file:
                pickle.dump(self.transcript, pickle_file, pickle.HIGHEST_PROTOCOL)
                   
    def correct_by_masking(self,bert_model = "",min_threshold = 0.6,max_mask_per_sentence = 3,min_mask_distance = 2,punctuate_model_path =''):
        
        self.human_review={}
        self.full_transcript_corrected=""
        if self.transcribe_full=='':
            return ''
        if self.mlm_full==None or self.errors_mlm==None:
            self.load_mlm_pipeline(bert_model)
            fstr = ""
            tl = 0
            self.errors_mlm=[]

            for sent in sent_tokenize(self.punctuate_full):
                words = sent.split(" ")
                fwords = words.copy()
                for i,wrd in enumerate(words):
                    #print(wrd)
                    if self.word_op[tl+i]['certain']<min_threshold:
                        aa=11
                        tempwords = words.copy()
                        if '.' in tempwords[i] or '?' in tempwords[i] or "!" in tempwords[i]:
                            tempwords[i] = "[MASK]" + tempwords[i][-1]
                        else:
                            tempwords[i] = "[MASK]"
                            
                        trgt = self.word_op[tl+i]['prob_words']
                        trgt = [i.lower() for i in trgt]
                        tempquery = " ".join(tempwords)
                        #print(tempquery,trgt)
                        result = self.mlm_unmasker(tempquery,targets = trgt)
                        #print(result)
                        fsent = result[0]['sequence']
                        if '.' in tempwords[i] or '?' in tempwords[i] or "!" in tempwords[i]:
                            words[i] = result[0]['token_str'].replace(" ","") + words[i][-1]
                        else:
                            words[i] = result[0]['token_str'].replace(" ","")
                        #print(self.word_op[tl+len(words)-1]['word'],wrd)
                        self.errors_mlm.append({'topic':self.topic_path,'video_id':self.video_id,
                               'start_trackname':self.word_op[tl]['track'],
                               'start_track_ratio':self.transcribe_op[self.word_op[tl]['track']]['ratio'],
                               'start_sent':self.word_op[tl]['start'],
                               'end_trackname':self.word_op[tl+len(words)-1]['track'],
                               'end_sent':self.word_op[tl+len(words)-1]['end'],
                               'end_track_ratio':self.transcribe_op[self.word_op[tl+len(words)-1]['track']]['ratio'],
                               'orig_sentence':sent,
                               'orig_word':self.word_op[tl+i]['word'],
                               'mlm_sentence':fsent,
                               'masked_query':tempquery})
            
            
                tl = tl+len(words)
                full_sent = " ".join(words)
     
                if self.mlm_full=="" or self.mlm_full == None:
                    self.mlm_full = full_sent
                else:
                    self.mlm_full = self.mlm_full + full_sent
    
    
    
               
            
            
            self.mlm_full = self.return_punctuated([self.mlm_full],punctuate_model_path)[0]
            self.writepkl(self.mlm_full,self.mlm_full_path)
            self.writepkl(self.errors_mlm,self.errors_mlm_path)
        
        self.upload_to_firebasedb()
        return self.mlm_full  
        




    def upload_to_firebasedb(self,firebase_config_path='',cred_file_path=''):
        if not self.upload_to_db:
            return None
        import firebase_admin
        from firebase_admin import credentials
        from firebase_admin import firestore
        import pyrebase
        #print(self.videopath)
        if firebase_config_path=='':
            self.firebase_config_path = os.path.join(self.base_path,'firebase_config.json')
        else:
            self.firebase_config_path = firebase_config_path

        if cred_file_path=='':
            self.cred_path = os.path.join(self.base_path,'firebase_creds.json')
        else:
            self.cred_path = cred_file_path

        with open(self.firebase_config_path) as f:
            data = f.read()
        
        self.firebase_config = json.loads(data)
        
        firebase = pyrebase.initialize_app(self.firebase_config)
        storage=firebase.storage()
        cred = credentials.Certificate(self.cred_path)
        if not firebase_admin._apps:
            firebase_admin.initialize_app(cred)
        db=firestore.client()
        collection_obj = db.collection('audios')


        for error in self.errors_mlm:
            if error['start_trackname'] == error['end_trackname']:
                start = error['start_sent']
                end = error['end_sent']
                ratio = error['start_track_ratio']
                trackname = error['start_trackname']
                uid = str(self.video_id) + '_' + str(trackname).replace(".","") + '_' + str(start) + '_' + str(end)
                mlm_sentence = error['mlm_sentence']
                origword = error['orig_word']
                origsent = error['orig_sentence']
                mask_query = error['masked_query']
                
                
                if not collection_obj.document(uid).get().exists:
                    trackname = os.path.join(self.directory,trackname)
                    start = int(start * ratio)
                    end = int(end * ratio)
                    wv,sr = torchaudio.load(trackname)
                    if sr != 16000:
                        wv = torchaudio.functional.resample(wv, sr, 16000)
                    if os.path.exists('temp.wav'):
                        os.remove('temp.wav')
                    torchaudio.save('temp.wav', wv[:, start:end], 16000)
                    saveFile1 = storage.child(uid + '.wav').put('temp.wav')
                    audio1 = storage.child(saveFile1['name']).get_url(saveFile1['downloadTokens'])
                    collection_obj.document(uid).create({'video_id':self.video_id,
                                                        'track_name':error['start_trackname'],
                                                        "audio1": audio1,
                                                        "correctedTranscript": mlm_sentence, 
                                                        "feedback": origword, 
                                                        "transcript": origsent,
                                                        "mask_query":mask_query,
                                                        "fine_tuned":False,
                                                        "topic_path":self.topic_path,
                                                        "submitted_word":''})    




    #class end
        

