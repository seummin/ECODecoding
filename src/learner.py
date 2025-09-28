import sys 
import json
import torch
import torch.nn as nn
import math
from tqdm import tqdm
# pytorch_lightning
from lightning.pytorch import LightningModule
from torch.optim import AdamW
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
from torch.distributions import Categorical
 
# from models.modeling_bart import BartForConditionalGeneration
from util import *
from transformers import AutoConfig
from transformers import RobertaForSequenceClassification
from model.modeling_gpt2_director import GPT2LMHeadModel
from transformers import AutoTokenizer, AutoModelWithLMHead, AutoModelForSequenceClassification

# from transformers import GPT2LMHeadModel

import pdb
import gzip
import pickle
import evaluate
from distinct_n import *
from perplexity import *
import json
from omegaconf import OmegaConf


accuracy = evaluate.load("accuracy")
rouge = evaluate.load('rouge')
f1_metric = evaluate.load("f1")
bleu = evaluate.load("bleu")
perplexity = Perplexity()

def top_k_top_p_filtering(
    logits,
    top_k: int = 0,
    top_p: float = 1.0,
    filter_value: float = -float("Inf"),
    min_tokens_to_keep: int = 1,
):
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    Args:
        logits: logits distribution shape (batch size, vocabulary size)
        if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
        if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        Make sure we keep at least min_tokens_to_keep per batch example in the output
    From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits

def grammaticality(sentences, tokenizer, model, device='cuda'):
    with torch.no_grad():
        total_good = 0
        for sent in tqdm(sentences, total=len(sentences)):
            good_prob = F.softmax(model(tokenizer.encode(sent, return_tensors='pt').to(device))[0].flatten(), dim=0)[1]
            total_good += good_prob
        return (total_good / len(sentences)).item() # avg probability of grammaticality according to model

class Learner(LightningModule):
    def __init__(self, cfg, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.config = AutoConfig.from_pretrained("microsoft/DialoGPT-small")
        
        self.config.update(OmegaConf.to_container(self.cfg))
        
        
        self.lm_model = GPT2LMHeadModel.from_pretrained("microsoft/DialoGPT-small",config=self.config)
        self.validation_step_outputs = []
        self.test_step_outputs = []
        
        # Classifier
        self.classifier_tokenizer =  AutoTokenizer.from_pretrained("roberta-large")
        config = AutoConfig.from_pretrained('roberta-large')
        
        config.num_labels = 6
        emo_all_classifier_path = '/nlp_data/seungmin/seungmin_was_in_ssd/seungmin/eco_decoding/classifier/no_emo_all/emo/2048/0.0001/checkpoint-7500/pytorch_model.bin'
        # emo_all_classifier_path = '/nlp_data/seungmin/seungmin_was_in_ssd/seungmin/eco_decoding/classifier/roberta-meld-6cls/best_all'
        # # '/nlp_data/seungmin/seungmin_was_in_ssd/seungmin/eco_decoding/classifier/no_emo_all/emo/2048/0.0001/checkpoint-7500/pytorch_model.bin'
        self.emo_all_classifier = RobertaForSequenceClassification.from_pretrained(emo_all_classifier_path, config=config).to(self.device)
        # self.emo_classifier = [self.emo_min_classifier,self.emo_all_classifier]
        # self.emo_classifier = [self.emo_all_classifier]
        config.num_labels = 4
        act_classifier_path = '/nlp_data/seungmin/seungmin_was_in_ssd/seungmin/eco_decoding/classifier/act/2048/0.0001/checkpoint-10000/pytorch_model.bin'
        self.act_classifier = RobertaForSequenceClassification.from_pretrained(act_classifier_path, config=config).to(self.device)
        
        
    def training_step(self, batch, batch_idx):
        outputs = self.lm_model(
            input_ids = batch['input_ids'],
            attention_mask = batch['attention_mask'],
            labels = batch['label_ids'],
            class_labels = batch['class_labels'],
            controls = batch['labels']
        )
        loss = outputs['loss']
        self.log('train_loss', loss, prog_bar = True)

        return {
            'loss' : loss
        }

    def validation_step(self, batch, batch_idx):
        outputs = self.lm_model(
            input_ids = batch['input_ids'],
            attention_mask = batch['attention_mask'],
            labels = batch['label_ids'],
            class_labels = batch['class_labels'],
            controls = batch['labels']
        )
        loss = outputs['loss']
        self.validation_step_outputs.append(loss)
        return {
            'loss' : loss
        }

    def on_validation_epoch_end(self):
        val_loss = torch.stack([x for x in self.validation_step_outputs]).mean()
        self.validation_step_outputs.clear()
        self.log('val_loss', val_loss, prog_bar = True)


    
    def test_step(self, batch, batch_idx):
        
        self.lm_model.reset_timing()
        output = self.lm_model.generate(
            input_ids = batch['input_ids'],
            attention_mask = batch['attention_mask'],
            controls = batch['labels'],
            condition_lambda = self.config.condition_lambda,
            smoothing_factor=self.config.smoothing_factor,
            # num_beams=1, min_length=0, max_length=512,
            num_beams=1, min_length=0, max_length=128,
            # top_k=10, max_new_tokens=64, early_stopping=True,
            # no_repeat_ngram_size=3,
        )
        # ② 평균 시간 출력 (스텝/토큰 단위 모두)
        print(f"[CTRL ON ] step avg : {self.lm_model.get_avg_step_time_ms(True):.2f} ms   "
            f"token avg : {self.lm_model.get_avg_token_time_ms(True):.2f} ms")
        print(f"[CTRL OFF] step avg : {self.lm_model.get_avg_step_time_ms(False):.2f} ms   "
            f"token avg : {self.lm_model.get_avg_token_time_ms(False):.2f} ms")
        self.test_step_outputs.append({
            'inputs' : self.tokenizer.batch_decode(batch['input_ids'])[0],
            'generated' : self.tokenizer.decode(output[0]),\
            'labels' : batch['labels'],\
            'response' : batch['response'],\
            "ctrl_step_ms": self.lm_model.get_avg_step_time_ms(True),
            "ctrl_tok_ms" : self.lm_model.get_avg_token_time_ms(True),
            "base_step_ms": self.lm_model.get_avg_step_time_ms(False),
            "base_tok_ms" : self.lm_model.get_avg_token_time_ms(False),
            # 'golden' : batch['golden'],\
            })
        
        return {
            # 'golden' : batch['golden'],
            'generated response' :self.tokenizer.decode(output[0]),
        }



    def on_test_epoch_end(self):
        ngram_response_list, generated_response_list, perplexity_list, labels, label_response = [], [], [], [],[]
        score = {}
        output_file =[]
        inputs=[]
        entropy=0
        lent=0
        ctrl_tok_sum_ms  = 0.0   # controls ON 총 소요 ms
        ctrl_tok_cnt     = 0     # controls ON 카운트 (= test step 수)
        base_tok_sum_ms  = 0.0   # controls OFF 총 소요 ms
        base_tok_cnt     = 0

        for output in self.test_step_outputs:
            # ---- 기존 코드 ----------------------------------------
            output_file.append(output['generated'].replace(self.tokenizer.eos_token, "\n").lstrip())
            inputs.append(output['inputs'].replace(self.tokenizer.eos_token, "\n").lstrip())

            gen_part = output['generated'][len(output['inputs']):].split(self.tokenizer.eos_token)[0]
            generated_response_list.append(gen_part)
            ngram_response_list.append(gen_part.split())

            label_response.append(output['response'][0].split(self.tokenizer.eos_token)[0])
            labels.append(output['labels'])
            # -------------------------------------------------------

            # ───────────────────────────────────────────────────────
            # ➋ 토큰‑평균 소요 시간 누적
            # ───────────────────────────────────────────────────────
            if 'ctrl_tok_ms' in output and output['ctrl_tok_ms'] > 0:
                ctrl_tok_sum_ms += output['ctrl_tok_ms']
                ctrl_tok_cnt    += 1
            if 'base_tok_ms' in output and output['base_tok_ms'] > 0:
                base_tok_sum_ms += output['base_tok_ms']
                base_tok_cnt    += 1
        # ────────────────────────────────────────────────────────────

        # ➌ 최종 평균 계산
        avg_ctrl_tok_ms = ctrl_tok_sum_ms  / ctrl_tok_cnt  if ctrl_tok_cnt  > 0 else 0.0
        avg_base_tok_ms = base_tok_sum_ms  / base_tok_cnt  if base_tok_cnt  > 0 else 0.0

        print(f"\n토큰당 평균 생성 속도")
        print(f"  ▸ CONTROL  ON : {avg_ctrl_tok_ms :.3f} ms")
        print(f"  ▸ CONTROL OFF : {avg_base_tok_ms:.3f} ms")
        # import pdb;pdb.set_trace();
        # aspect accuracy
        if self.cfg.data_name=='emo':
            acc=[]
            f1_score=[]
            bleu_score=[]
            preds = []
            for i in range(len(generated_response_list)):
                model_input = self.classifier_tokenizer(
                    generated_response_list[i],
                    max_length = 512,
                    padding = 'max_length',
                    return_tensors = 'pt',
                    truncation = True,
                    add_special_tokens = True,
                    return_attention_mask = True
                    )
                # model_output = self.emo_min_classifier(input_ids = model_input['input_ids'].to(self.device), attention_mask = model_input['attention_mask'].to(self.device))
                model_output1 = self.emo_all_classifier(input_ids = model_input['input_ids'].to(self.device), attention_mask = model_input['attention_mask'].to(self.device))
                # preds.append(model_output['logits'].argmax(dim=1).item())
                preds.append(model_output1['logits'].argmax(dim=1).item())
            # acc.append(accuracy.compute(predictions=preds, references=labels)) 
            acc.append(accuracy.compute(predictions=preds, references=labels))
            
            f1_score.append(f1_metric.compute(predictions=preds, references=labels, average="macro"))
            bleu_score.append(bleu.compute(predictions=generated_response_list, references=label_response, max_order=1))
            bleu1=0
            for pre, ref in zip(generated_response_list, label_response):
                if pre=='':
                    continue
                bleu1 += bleu.compute(predictions=[pre], references=[ref], max_order=1)['bleu']
            bleu1/=len(generated_response_list)
        elif self.cfg.data_name=='act':
            acc=[]
            f1_score=[]
            bleu_score=[]
            preds = []
            for i in range(len(generated_response_list)):
                model_input = self.classifier_tokenizer(
                    generated_response_list[i],
                    max_length = 256,
                    padding = 'max_length',
                    return_tensors = 'pt',
                    truncation = True,
                    add_special_tokens = True,
                    return_attention_mask = True
                    )
                model_output = self.act_classifier(input_ids = model_input['input_ids'].to(self.device), attention_mask = model_input['attention_mask'].to(self.device))
                preds.append(model_output['logits'].argmax(dim=1).item())
            acc = accuracy.compute(predictions=preds, references=labels)
            f1_score.append(f1_metric.compute(predictions=preds, references=labels, average="macro"))
            bleu_score.append(bleu.compute(predictions=generated_response_list, references=label_response, max_order=1))
            bleu1=0
            for pre, ref in zip(generated_response_list, label_response):
                if pre=='':
                    continue
                bleu1 += bleu.compute(predictions=[pre], references=[ref], max_order=1)['bleu']
            bleu1/=len(generated_response_list)
        # ppl
        # ppl = perplexity.compute(predictions=perplexity_list, tokenizer=self.tokenizer, model=self.lm_model, max_length=512)
        # distinct 1,2,3
        dist_1,dist_2,dist_3,=0,0,0
        for sentence in ngram_response_list:
            tmp_dist1, tmp_dist2, tmp_dist3 = distinct_metric(sentence)
            dist_1+=tmp_dist1
            dist_2+=tmp_dist2
            dist_3+=tmp_dist3
        dist_1 /= len(ngram_response_list)
        dist_2 /= len(ngram_response_list)
        dist_3 /= len(ngram_response_list)
        
        # grammar
        grammar_tokenizer = AutoTokenizer.from_pretrained('textattack/roberta-base-CoLA')
        grammar_model = AutoModelForSequenceClassification.from_pretrained('textattack/roberta-base-CoLA').to(self.device)
        grammar_model.eval()
        # import pdb;pdb.set_trace();
        score['grammar'] =  grammaticality(generated_response_list, grammar_tokenizer, grammar_model, device=self.device)
        
        # rouge
        rouge_output = rouge.compute(predictions=generated_response_list, references=label_response)
        score['ROUGE-1'] = rouge_output['rouge1']
        score['ROUGE-2'] = rouge_output['rouge2']
        score['ROUGE-L'] = rouge_output['rougeL']
        
        if self.cfg.data_name=='act':
            score['ACCURACY'] = acc['accuracy']
            score['F1-score'] = f1_score[0]['f1']
            score['BLEU1'] = bleu1
            score['BLEU'] = bleu_score[0]['bleu']
        elif self.cfg.data_name=='emo':
            score['ACCURACY_ALL'] = acc[0]['accuracy']
            score['F1-score'] = f1_score[0]['f1']
            score['BLEU1'] = bleu1
            score['BLEU'] = bleu_score[0]['bleu']
            # score['ACCURACY_MIN'] = acc[0]['accuracy']
            # score['ACCURACY_ALL'] = acc[1]['accuracy']
        # score['PERPLEXITY'] = round(ppl['mean_perplexity'], 2)
        score['DISTINCT-1'] = dist_1
        score['DISTINCT-2'] = dist_2
        score['DISTINCT-3'] = dist_3

        for key, value in score.items():
            self.log(key, value, prog_bar = True)
        result=[]
        label2id_emo =['anger','disgust','fear','happiness','sadness','surprise']
        label2id_act =['inform','question','directive','commissive']
        for lab,label, data in zip(labels,preds, output_file):
            if self.cfg.data_name=='act':
                result.append(("answer : "+label2id_act[lab],"predict : "+label2id_act[label],data))
            elif self.cfg.data_name=='emo':
                result.append(("answer : "+label2id_emo[lab],"predict : "+label2id_emo[label],data))
        ckpt_path = self.cfg.ckpt_path + "/smoothing"
        result.append(score)
        save_path = "tmp_"+str(self.cfg.smoothing_factor)+"_entropy_"+str(self.config.condition_lambda)+".json"
        with open(os.path.join(ckpt_path, save_path), 'w') as f:
            f.write(json.dumps(result))

        self.test_step_outputs.clear()
        
    

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), betas=(0.9,0.98), eps=1e-9, lr = self.config.learning_rate)
        scheduler = InverseSqrtScheduler(optimizer, 4000)
        sch_config = {
            "scheduler" : scheduler,
            "interval" : "step"
        }
        
        return {
                "optimizer" : optimizer, 
                "lr_scheduler" : sch_config, 
            }