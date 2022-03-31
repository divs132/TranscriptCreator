import os,json,torchaudio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from transformers import Wav2Vec2ProcessorWithLM
import torch
from transformers import AutoProcessor, HubertForCTC,PreTrainedTokenizer

import numpy as np
from itertools import groupby
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
import pyrebase
from transformers import BertTokenizer, BertForMaskedLM
import math

class hrecord(object):
    def __init__(self, audio,mlm_op,orig_word,fine_tuned,mlm_query,submitted_word,topic_path
                 ,track_name,transcribe_op,video_id):
        self.audio_link = audio
        self.mlm_op = mlm_op
        self.orig_word = orig_word
        self.fine_tuned = fine_tuned
        self.mlm_query = mlm_query
        self.submitted_word = submitted_word
        self.topic_path=topic_path
        self.track_name=track_name
        self.transcribe_full=transcribe_op
        self.video_id=video_id

def from_dict(source):
        hrc = hrecord(
        source['audio1'],
        source['correctedTranscript'],
        source['feedback'],
        source['fine_tuned'],
        source['mask_query'],
        source['submitted_word'],
        source['topic_path'],
        source['track_name'],
        source['transcript'],
        source['video_id'])
        return hrc

class FineTuner():
    def __init__(self,firebase_config_path='',cred_file_path='',baserootpath='/home/divyansh/Documents/Capstone',
    oldbaserootpath='/home/divyansh/Documents/Capstone',outdir = '/home/divyansh/Desktop/train1') -> None:
        self.base_path = baserootpath
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
        docs = collection_obj.where(u'fine_tuned', u'==', False).where('submitted_word','!=',"").stream()
        hrecords=[]
        for doc in docs:
            hrecords.append(from_dict(doc.to_dict()))

        self.all_datasets=[]
        self.all_tokens =[]
        self.all_hfine_tuner = []
        self.all_mlm_tuner = []
        self.all_gtruths=[]
        self.all_mlm_datasets = []
        
        groupedlist = [list(g[1]) for g in groupby(sorted(hrecords, key=lambda x:x.video_id), lambda x:x.video_id)]
        for topic in groupedlist:
            dataset = []
            tokens = []
            gtruths =[]
            mlm_dataset=[]
            for doc in topic:
                wv,sr = torchaudio.load(doc.audio_link)
                if sr != 16000:
                    arr = torchaudio.functional.resample(arr, sr, 16000)
                if wv.shape[0]>1:
                    wv = wv[1:,]
                gtruth = doc.mlm_query.replace('[MASK]',doc.submitted_word)
                dataset.append(dict(array=wv,text=gtruth,sampling_rate = 16000))
                tokens.append(doc.submitted_word)
                gtruths.append(gtruth)
                mlm_dataset.append(dict(mask_text = doc.mlm_query,ground_truth = gtruth))

            sel_topic = doc.topic_path
            hfine_tuner = Hubert_Model_FineTuner(topic_path=sel_topic,baserootpath=baserootpath,oldbaserootpath=oldbaserootpath,ground_truths = gtruths,out_dir=outdir)
            mlm_tuner = MLM_Model_FineTuner(topic_path=sel_topic,baserootpath=baserootpath,oldbaserootpath=oldbaserootpath,ground_truths = gtruths,out_dir=outdir)
            self.all_hfine_tuner.append(hfine_tuner)
            self.all_mlm_tuner.append(mlm_tuner)
            self.all_datasets.append(dataset)
            self.all_tokens.append(tokens)
            self.all_gtruths.append(gtruths)
            self.all_mlm_datasets.append(mlm_dataset)
    
    def start_finetune(self):

        for i,wav_tuner in enumerate(self.all_hfine_tuner):
            wav_tuner.finetune_model(dataset = self.all_datasets[i],tokens = self.all_tokens[i])
            del wav_tuner
            self.all_mlm_tuner[i].finetune_model(dataset = self.all_mlm_datasets[i],tokens = self.all_tokens[i])
            





@dataclass
class DataCollatorforMLM:
    tokenizer: PreTrainedTokenizer
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        #print(features)
        input_features = [{"input_ids": feature["input_ids"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        #print(input_features)
        batch = self.tokenizer.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        #print(batch)
        special_tokens_mask = batch.pop("special_tokens_mask", None)
        labels = batch["input_ids"].clone()
        if self.tokenizer.pad_token_id is not None:
            labels[labels == self.tokenizer.pad_token_id] = -100
        batch["labels"] = labels
        return batch           
            



@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2ProcessorWithLM
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        #print(input_features)
        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        #print(batch)
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels
        #print(batch)
        return batch



class MLM_Model_FineTuner():
    def __init__(self,topic_path = '',
            baserootpath = '/home/divyansh/Documents/Capstone',
            oldbaserootpath = '/home/divyansh/Documents/Capstone',out_dir='',ground_truths = []) -> None:
        
        self.topic_path = topic_path.replace(oldbaserootpath,baserootpath)
        self.output_dir = out_dir
        self.initialize_topic()
        self.ground_truths = ground_truths
        pass
    


    def checkfortokens(self,tokens):
        vocab_tokens = self.mlm_tokenizer.get_vocab()
        toadd=[]
        print(tokens)
        for token in tokens:
            if not token.upper() in vocab_tokens:
                toadd.append(token.upper())
        
        if len(toadd)>0:
            self.mlm_tokenizer.add_tokens(toadd)
            self.mlm_model.resize_token_embeddings(len(toadd))
        
    def finetune_model(self,dataset,tokens,
            mlm_model_path=''):
        self.load_mlm_pipeline(mlm_model_path)
        self.train_dataset=[self.prepare_dataset(data) for data in dataset]
        self.eval_dataset = self.train_dataset
        self.data_collator = dcl = DataCollatorforMLM(tokenizer = self.mlm_tokenizer,padding = True)
        #self.checkfortokens(tokens)
        self.create_trainer()
        self.bfr_train_eval = self.trainer.evaluate()
        self.trainer.train()
        self.aftr_train_eval = self.trainer.evaluate()
        self.trainer.save_model(self.mlm_model_path)
        print(math.exp(self.bfr_train_eval['eval_loss']) , math.exp(self.aftr_train_eval['eval_loss']))



    def initialize_topic(self):
        self.audio_to_text_model_path = os.path.join(self.topic_path,'HuBERT')
        self.audio_to_text_tokenizer_path = os.path.join(self.topic_path,'Beam_LM')
        self.mlm_model_path = os.path.join(self.topic_path,'BERT_MLM')



    def load_mlm_pipeline(self,model_path):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if not hasattr(self,'mlm_unmasker'):
            if type(model_path)==str and model_path!='':
                self.mlm_tokenizer = BertTokenizer.from_pretrained(model_path)
                self.mlm_model = BertForMaskedLM.from_pretrained(model_path).to(device)
            else:
                self.mlm_tokenizer = BertTokenizer.from_pretrained(self.mlm_model_path)
                self.mlm_model = BertForMaskedLM.from_pretrained(self.mlm_model_path)

    def prepare_dataset(self,batch):

        # batched output is "un-batched" to ensure mapping is correct
        #batch["input_values"] = np.squeeze(batch["array"].numpy()).transpose()
        print(batch)
        batch['text'] = batch['mask_text']
        batch['input_ids']  = np.squeeze(self.mlm_tokenizer(batch['text'],return_tensors = 'pt')['input_ids'].numpy())
        batch['labels'] = np.squeeze(self.mlm_tokenizer(batch['ground_truth'],return_tensors = 'pt')['input_ids'].numpy())
 
        return batch

    def create_trainer(self):
        from transformers import TrainingArguments

        training_args = TrainingArguments(
        output_dir=self.output_dir,
        group_by_length=True,
        per_device_train_batch_size=2,
        evaluation_strategy="steps",
        num_train_epochs=5,
        fp16=True,
        gradient_checkpointing=False, 
        save_steps=500,
        eval_steps=500,
        logging_steps=500,
        learning_rate=1e-4,
        weight_decay=0.005,
        warmup_steps=1000,
        save_total_limit=2,
            push_to_hub=False
        )

        from transformers import Trainer

        self.trainer = Trainer(
            model=self.mlm_model,
            data_collator=self.data_collator,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            tokenizer=self.mlm_tokenizer,
        )
    



class Hubert_Model_FineTuner():
    def __init__(self,topic_path = '',
            baserootpath = '/home/divyansh/Documents/Capstone',
            oldbaserootpath = '/home/divyansh/Documents/Capstone',out_dir='',ground_truths = []) -> None:
        
        self.topic_path = topic_path.replace(oldbaserootpath,baserootpath)
        self.output_dir = out_dir
        self.initialize_topic()
        self.ground_truths = ground_truths
        
        




        pass
    


    def checkfortokens(self,tokens):
        vocab_tokens = self.decoder.tokenizer.get_vocab()
        toadd=[]
        print(tokens)
        for token in tokens:
            if not token in vocab_tokens:
                toadd.append(token)
        
        if len(toadd)>0:
            self.decoder.tokenizer.add_tokens(toadd)
            #self.transcribe_model.resize_token_embeddings(len(toadd))
        







    def finetune_model(self,dataset,tokens,
            transcribe_model_path='',transcribe_decoder_path = ''):
        self.load_transcribe_decoder(transcribe_decoder_path)
        self.train_dataset=[self.prepare_dataset(data,self.decoder) for data in dataset]
        self.eval_dataset = self.train_dataset
        self.data_collator = DataCollatorCTCWithPadding(processor=self.decoder,padding=True)
        self.load_transcribe_model(transcribe_model_path)
        self.checkfortokens(tokens)
        self.create_trainer()
        self.bfr_train_eval = self.trainer.evaluate()
        self.trainer.train()
        self.aftr_train_eval = self.trainer.evaluate()
        self.trainer.save_model(self.audio_to_text_model_path)
        print(self.bfr_train_eval,self.aftr_train_eval)
        del self.transcribe_model
        del self.decoder



    def initialize_topic(self):
        self.audio_to_text_model_path = os.path.join(self.topic_path,'HuBERT')
        self.audio_to_text_tokenizer_path = os.path.join(self.topic_path,'Beam_LM')
        self.mlm_model_path = os.path.join(self.topic_path,'BERT_MLM')



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

    def prepare_dataset(self,batch,proc):

        # batched output is "un-batched" to ensure mapping is correct
        #batch["input_values"] = np.squeeze(batch["array"].numpy()).transpose()
        print(batch)
        batch["input_values"] = proc.feature_extractor(batch["array"], sampling_rate=batch["sampling_rate"]).input_values[0]
        #print(batch["array"].shape,batch["input_values"].shape)
        batch["input_values"] = np.squeeze(batch["input_values"])
        with proc.as_target_processor():
            batch["labels"] = proc.tokenizer.encode(" ".join(batch["text"].upper()))
        return batch

    def compute_metrics(self,pred):
        pred_logits = np.squeeze(pred.predictions)

        pred.label_ids[pred.label_ids == -100] = self.decoder.tokenizer.pad_token_id

        pred_str = self.decoder.decode(pred_logits).text
        #print(pred_str)
        # we do not want to group tokens when computing the metrics
        label_str = self.ground_truths
        #print(label_str)
        import jiwer
        transformation = jiwer.Compose([
                    jiwer.ToLowerCase(),
                    jiwer.RemovePunctuation(),
                    jiwer.RemoveWhiteSpace(replace_by_space=True),
                    jiwer.RemoveMultipleSpaces(),
                    jiwer.ReduceToListOfListOfWords(word_delimiter=" ")
                ])

        wer = jiwer.wer(label_str, pred_str,truth_transform=transformation,hypothesis_transform=transformation)

        return {"wer": wer}

    def create_trainer(self):
        from transformers import TrainingArguments

        training_args = TrainingArguments(
        output_dir=self.output_dir,
        group_by_length=True,
        per_device_train_batch_size=2,
        evaluation_strategy="steps",
        num_train_epochs=5,
        fp16=True,
        gradient_checkpointing=False, 
        save_steps=500,
        eval_steps=500,
        logging_steps=500,
        learning_rate=1e-4,
        weight_decay=0.005,
        warmup_steps=1000,
        save_total_limit=2,
            push_to_hub=False
        )
        from transformers import Trainer

        self.trainer = Trainer(
            model=self.transcribe_model,
            data_collator=self.data_collator,
            args=training_args,
            compute_metrics=self.compute_metrics,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            tokenizer=self.decoder.feature_extractor,
        )
    
