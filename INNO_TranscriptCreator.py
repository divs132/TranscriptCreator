import os
import shutil
from typing import Tuple
import moviepy.editor as mp
import pickle
from pyannote.audio import Pipeline
import soundfile as sf
import math
import torch
import torchaudio
from transformers import AutoProcessor, HubertForCTC
from nemo.utils.exp_manager import exp_manager
from nemo.collections import nlp as nemo_nlp
import regex as re
import glob
from nltk.tokenize import sent_tokenize
from neuspell import available_checkers, BertChecker
from itertools import zip_longest 
from transformers import BertTokenizer, BertForMaskedLM
from transformers import pipeline
import nltk
from itertools import repeat

class TranscriptCreator():
    def __init__(self,videopath,savedirectory='',video_topic='Computer Science',baserootpath = '/home/divyansh/Documents/Capstone'):
        self.base_path=baserootpath
        self.initialize_env()
        self.initialize_topic(video_topic)

        self.original_videopath = videopath
        vdname = os.path.basename(videopath).replace(".","")
        dirname = os.path.join(self.base_path, 'Videos')
        dirname = os.path.join(dirname, vdname)
        
        if(savedirectory==''):
            self.directory = dirname
        else:
            self.directory = savedirectory
            
        if not os.path.exists(self.directory):
            os.makedirs(self.directory) 
            
        self.videopath = os.path.join(self.directory,os.path.basename(videopath))  

        if not os.path.exists(self.videopath):
            shutil.copyfile(self.original_videopath, self.videopath)
            
        clip = mp.VideoFileClip(self.videopath)
        self.audiopath = os.path.join(self.directory,"audio.wav")
        
        if not os.path.exists(self.audiopath):
            clip.audio.write_audiofile(self.audiopath)


    def create_transcript(self,diarize=False):
        if diarize:
            self.diarization()
            torch.cuda.empty_cache()
            self.separatetracks()
            torch.cuda.empty_cache()
            self.speechtotext()
            torch.cuda.empty_cache()
            self.full_punctuate()
            torch.cuda.empty_cache()
            self.correct_by_masking()
            torch.cuda.empty_cache()
            self.spellcheck()
            torch.cuda.empty_cache()
        else:
            self.tracks = [os.path.join(self.directory,'audio.wav')]
            self.speechtotext()
            torch.cuda.empty_cache()
            self.full_punctuate()
            torch.cuda.empty_cache()
            self.correct_by_masking()
            torch.cuda.empty_cache()
            self.spellcheck()
            torch.cuda.empty_cache()

        return dict(Corrected_Transcript = self.full_transcript_corrected,Spell_Checked = self.spell_checked_1)


    def initialize_env(self):
        if not os.path.exists(self.base_path):
            raise NotImplementedError("Do not currently support completely new environment")
        
        if not os.path.exists(os.path.join(self.base_path,'Diarization/pipeline_checkpoint')):
            shutil.copytree(os.path.join(self.base_path,'Checkpoints/Diarization'), os.path.join(self.base_path,'Diarization'))
        
        self.diarize_model_path = os.path.join(self.base_path,'Diarization')
        self.diarize_model_path = os.path.join(self.diarize_model_path,'pipeline_checkpoint')

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
  
    def diarization(self,model = ''):
        
        if not model == '':
            self.diarize_model_path = model

        
        self.diarize_path = os.path.join(self.directory,'diarized.pkl')
        if os.path.exists(self.diarize_path):
            print("PKL File Found")
            self.New_Separation=False
            with open(self.diarize_path, 'rb') as inp:
                self.diarized = pickle.load(inp)   
        else:
            print("Doing Diarization")
            self.diarize_model = Pipeline.from_pretrained(self.diarize_model_path)
            self.diarized = self.diarize_model(self.audiopath)
            with open(self.diarize_path, 'wb') as pickle_file:
                pickle.dump(self.diarized, pickle_file, pickle.HIGHEST_PROTOCOL)
   
    def separatetracks(self,max_pause = 1.0, max_speakers = 1,maxduration=60,make_new = False):
        self.speaker_dict={}
        self.tracks=list()
        if make_new:
            files = glob.glob(self.directory+ '/*.wav')
            for f in files:
                if not os.path.basename(f) == 'audio.wav':
                    os.remove(f)



        data, fs = sf.read(self.audiopath)
        for speech_turn, track, speaker in self.diarized.itertracks(yield_label=True):
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
                else:
                    self.speaker_dict[speaker]['tracklist'].append({'start':speech_turn.start,'end':speech_turn.end})
                
                
                self.speaker_dict[speaker]['duration']= speech_turn.end - speech_turn.start
                self.speaker_dict[speaker]['num_tracks']=1
                
        
        
        res = dict(sorted(self.speaker_dict.items(), key=lambda item: item[1]['duration'], reverse = True)[:max_speakers])
        self.speaker_dict = res
        
        
        for speaker in self.speaker_dict.items():
            for i in range(len(speaker[1]['tracklist'])-1):
                fname = str(speaker[0]) + 'track_'+ str(i)+'.wav'
                #print(fname)
                fname = os.path.join(self.directory,fname)
                self.tracks.append(fname)
                startPoint = math.floor(speaker[1]['tracklist'][i]['start'])*fs
                endPoint = math.ceil(speaker[1]['tracklist'][i]['end'])*fs
                if not os.path.exists(fname):
                    sf.write(fname, data[startPoint:endPoint], fs)             
            
    def speechtotext(self,wav2vec2typemodel='',processorwithlmhead=''):
            self.full_transcript = ''
            if not wav2vec2typemodel=='':
                self.audio_to_text_model_path=wav2vec2typemodel
            if not processorwithlmhead=='':
                self.audio_to_text_tokenizer_path=processorwithlmhead
            
            self.transcript_path = os.path.join(self.directory,"transcript.pkl")
            if os.path.exists(self.transcript_path):
                print("PKL File Found")
                with open(self.transcript_path, 'rb') as inp:
                    self.transcript = pickle.load(inp)
                    self.full_transcript    = self.transcript['full_transcript']
            else:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                      
                self.transcribe_model = HubertForCTC.from_pretrained(self.audio_to_text_model_path).to(device)
                self.decoder = AutoProcessor.from_pretrained(self.audio_to_text_tokenizer_path)
            
                print("Model Loaded")
                self.transcript = {}
                
                #return '0'
            
                for track in self.tracks:
                    print("Loading Track:" + track)
                    waveform, sample_rate = torchaudio.load(track)
                    #waveform = waveform.to(device)

                    if sample_rate != 16000:
                        waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
                
                    with torch.no_grad():
                        logits = self.transcribe_model(waveform.to(device)).logits[0].cpu().numpy()
                        del waveform
                        transcript = self.decoder.decode(logits,output_word_offsets = False)['text']
                        sentence_probs = self.decoder.decoder.decode_beams(logits)
                        nsentence_probs = []
                        for probs in sentence_probs:
                            nsentence_probs.append(dict(word_offsets=probs[2],logit = probs[3],lm_score = probs[4]))
                        #print(transcript)
                        

                    
                    trackdictname = os.path.basename(track)
                    if not trackdictname in self.transcript:
                        self.transcript[trackdictname] = {}
                    self.transcript[trackdictname]['logits'] = logits
                    del logits
                    self.transcript[trackdictname]['transcript'] = transcript
                    self.full_transcript = self.full_transcript + ' ' + transcript
                
                    logit_scores=[]
                    word_offsets = nsentence_probs[0]['word_offsets']
                    
                    
                    words = []
                    for i in nsentence_probs:
                        if len(i['word_offsets'])==len(nsentence_probs[0]['word_offsets']):
                            words.append([j[0] for j in i['word_offsets']])
                            logit_scores.append(i['logit'])
                    l = list(zip_longest(*words))
                    m = list(torch.nn.Softmax()(torch.Tensor(logit_scores)).numpy())
                    word_certain=[]
                    for prob_words in l:
                        certain = 0
                        if len(set(prob_words))==1:
                            word_certain.append(1)
                            #print(prob_words[0])
                            continue
                        for i, wrd in enumerate(prob_words):
                            if wrd == prob_words[0]:
                                certain = certain + m[i]
                            else:
                                #print(prob_words[0],wrd)
                                aa=11
                            
                        word_certain.append(certain)
    
                    self.transcript[trackdictname]['word_probs'] = word_certain
                    self.transcript[trackdictname]['word_offsets'] = word_offsets 
                    
                    torch.cuda.empty_cache()
                    #break
                
                del self.transcribe_model
                torch.cuda.empty_cache()
                self.transcript['full_transcript'] = self.full_transcript
    
                with open(self.transcript_path, 'wb') as pickle_file:
                    pickle.dump(self.transcript, pickle_file, pickle.HIGHEST_PROTOCOL)
        

    def full_punctuate(self,model_path= ''):
        if model_path=='':
            model_path=self.punctuate_model_path
        self.full_punctuate_path = os.path.join(self.directory,"full_punctuate.pkl")
        
        if os.path.exists(self.full_punctuate_path):
            print("PKL File Found")
            with open(self.full_punctuate_path, 'rb') as inp:
                self.transcript = pickle.load(inp)
        else:
            if os.path.exists(model_path):
                print("loading form pretrained")
                self.punctuate_model = nemo_nlp.models.PunctuationCapitalizationModel.restore_from(model_path)
            
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.punctuate_model = self.punctuate_model.to(device)
            query = [self.full_transcript.lower()]
            with torch.no_grad():
                inference = self.punctuate_model.add_punctuation_capitalization(query,max_seq_length=128,step=8,margin=16,batch_size=32)

            self.transcript['Full_Punctuated'] = inference
            torch.cuda.empty_cache()    
            with open(self.full_punctuate_path, 'wb') as pickle_file:
                pickle.dump(self.transcript, pickle_file, pickle.HIGHEST_PROTOCOL)

    def spellcheck(self,query = '',spellcheck_model_path='',EACH_SENTENCE=True):
        
        self.spell_checked_path_1 = os.path.join(self.directory,'spell_checked_1.pkl')
        
        if os.path.exists(self.spell_checked_path_1):
            print("PKL File Found")
            with open(self.spell_checked_path_1, 'rb') as inp:
                self.spell_checked_1  = pickle.load(inp)
        else:
        
            if not spellcheck_model_path=='':
                self.spellcheck_model_path = spellcheck_model_path
            
            if query == '':
                query = self.full_transcript_corrected
            

            full_corrected=''
            self.spell_checker_model = BertChecker(device="cuda")
            self.spell_checker_model._from_pretrained(ckpt_path = self.spellcheck_model_path,vocab_path = os.path.join(self.spellcheck_model_path,'vocab.pkl'))

            if EACH_SENTENCE:
                query = sent_tokenize(query)
            else:
                query = [query]
            
            self.spell_checked_1 = self.spell_checker_model.correct_strings(query)
            with open(self.spell_checked_path_1, 'wb') as pickle_file:
                pickle.dump(self.spell_checked_1, pickle_file, pickle.HIGHEST_PROTOCOL)



        





    
    def punctuate(self,model_path= ''):
        if model_path=='':
            model_path=self.punctuate_model_path
        self.punctuate_path = os.path.join(self.directory,"punctuate.pkl")
        
        if os.path.exists(self.punctuate_path):
            print("PKL File Found")
            with open(self.punctuate_path, 'rb') as inp:
                self.transcript = pickle.load(inp)
        else:
            if os.path.exists(model_path):
                print("loading form pretrained")
                self.punctuate_model = nemo_nlp.models.PunctuationCapitalizationModel.restore_from(model_path)
            
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
                



                
                
    def correct_by_masking(self,bert_model = "",min_threshold = 0.6,max_mask_per_sentence = 3,min_mask_distance = 2):
        self.corrected_path = os.path.join(self.directory,"corrected.pkl")
        self.human_path = os.path.join(self.directory,"human.pkl")
        self.human_review={}
        self.full_transcript_corrected=""
        
        
        
        torch.cuda.empty_cache()
        if os.path.exists(self.corrected_path) and os.path.exists(self.human_path):
            print("PKL File Found")
            with open(self.corrected_path, 'rb') as inp:
                self.full_transcript_corrected = pickle.load(inp)
            with open(self.human_path, 'rb') as inp:
                self.human_review = pickle.load(inp)
        else:
            punct_marks = self.transcript['Full_Punctuated'][0].split(" ")

            wprb = []
            wofs = []
            audio_name = []

            fstr = ""
            tokenizer = BertTokenizer.from_pretrained(self.mlm_model_path)
            model = BertForMaskedLM.from_pretrained(self.mlm_model_path)
            unmasker = pipeline(task = 'fill-mask', model=model,tokenizer = tokenizer,device=-1,top_k=1)
            human_dict=dict.fromkeys(self.transcript.keys(),[])

            for i in self.transcript:
                if '.wav' in i:
                    wprb = wprb + self.transcript[i]['word_probs']
                    wofs = wofs + self.transcript[i]['word_offsets']
                    audio_name.extend(repeat(i,len(self.transcript[i]['word_probs'])))
                    


            #print(len(wprb),len(wofs))
            tl=0
            for sentence_text in sent_tokenize(self.transcript['Full_Punctuated'][0]):
                masks = 0
                errors=[]
                #print(test_str,1)
                res = re.sub(r'[^\w\s]', '', sentence_text).split()
                if res == []:
                    continue
                fres=[]
                cur_dist = 0
                for i in range(len(res)):
                    curword = res[i]
                    curwprb = wprb[tl+i]
                    curwoff = wofs[tl+i]
                    cur_punct = punct_marks[tl + i]
                    #print(curword,curwseg)
                    if curwprb<min_threshold and masks<=max_mask_per_sentence and cur_dist ==0 and len(res)>min_mask_distance:
                        if '.' in cur_punct or '?' in cur_punct or "!" in cur_punct:
                            fres.append("[MASK]" + cur_punct[-1])
                        else:
                            fres.append("[MASK]")
                        
                        
                        
                        errors.append([curwoff,curwprb])
                        masks = masks + 1
                        cur_dist = min_mask_distance
                        
                    else:
                        
                        if '.' in cur_punct or '?' in cur_punct or "!" in cur_punct:
                            fres.append(curword + cur_punct[-1])
                            #print(fres[-1])
                        else:
                            fres.append(curword )
                        if cur_dist > 0:
                            cur_dist = cur_dist - 1
                
                
                            
                tl = tl + len(res)
                torch.cuda.empty_cache()
                query = " ".join(fres)
                if "[MASK]" in query:
                    #print(query)
                    result = unmasker(query)
                    human_dict[audio_name[tl]].append([errors,result])
                    torch.cuda.empty_cache()
                    if type(result[0]) == type({}):
                        query = query.replace("[MASK]",result[0]['token_str'].replace(" ",""),1)
                        

                                    
                    else:
                        for i in range(len(result)):
                            #print(result[i][0]['token_str'].replace(" ",""))
                            query = query.replace("[MASK]",result[i][0]['token_str'].replace(" ",""),1)
                            #print(query)
                            
                if fstr=="":
                    fstr=query
                else:
                    fstr = fstr + " " + query


    
                
            if self.full_transcript_corrected=="":
                self.full_transcript_corrected = fstr
            else:
                self.full_transcript_corrected = self.full_transcript_corrected + ". " + fstr
    
    
    
               
            
            
            
            self.human_review=human_dict
            with open(self.corrected_path, 'wb') as pickle_file:
                pickle.dump(self.full_transcript_corrected, pickle_file, pickle.HIGHEST_PROTOCOL)
            with open(self.human_path, 'wb') as pickle_file:
                pickle.dump(self.human_review, pickle_file, pickle.HIGHEST_PROTOCOL)
            
        
    #class end
        

