
from utils import stream

import os
from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
from datasets import Dataset, load_dataset
import peft
print("peft.__version__: ",peft.__version__)
import torch
print("torch.__version__:",torch.__version__)
import transformers
print("transformers.__version__: ",transformers.__version__)

from datasets import Dataset as HFDataset

from transformers import (AdamW,
                          AutoTokenizer,
                          HfArgumentParser,
                          TrainingArguments,
                          AutoModelForCausalLM,
                          TextDataset,
                          DataCollatorForLanguageModeling,
                          BitsAndBytesConfig,
                          TextStreamer,
                          get_scheduler,
                          get_linear_schedule_with_warmup)
import datasets
print("datasets.__version__: ",datasets.__version__)
from peft import LoraConfig, get_peft_config, get_peft_model, LoraConfig, TaskType
from datasets import load_dataset
import bitsandbytes as bnb
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler
import pandas as pd
from typing import Any, Dict, Sequence, Union


from determined import pytorch
from determined.pytorch import DataLoader, PyTorchTrial, PyTorchTrialContext, LRScheduler, PyTorchCallback

class MyCallbacks(PyTorchCallback):
    def __init__(self, cat_mapping=3,model=None,tokenizer=None,save_model_dir=None) -> None:
        self.cat_mapping = cat_mapping
        self.output_json_file = None
        self.model=model
        self.save_model_dir = save_model_dir
        self.tokenizer=tokenizer
        super().__init__()

    def on_checkpoint_load_start(self, checkpoint: Dict[str, Any]) -> None:
        print("loading checkpoint")
        print(checkpoint)

    def on_checkpoint_save_start(self,checkpoint: Dict[str, Any]):
        print("saving checkpoint")
        
    def on_checkpoint_write_end(self, checkpoint_dir: str):
        print(f"checkpoint dir: {checkpoint_dir}")
        
        # save pretrained model
        save_model_dir = self.save_model_dir
        # tokenizer_dir = '/mnt/efs/shared_fs/mistral_ckpt/mistral_tokenizer'
        print("Saving model at {}...".format(save_model_dir))
        # self.model.save_state_dict(model_dir)
        merged_model = self.model.merge_and_unload()
        # merged_model.to(torch.bfloat16)
        merged_model.save_pretrained(save_model_dir,safe_serialization=True)

        print("Done")
        print("Saving tokenizer at {}...".format(save_model_dir))
        self.tokenizer.save_pretrained(save_model_dir)
        print("Done")        

            
class MixtralFinetuneTrial(PyTorchTrial):
    
    def __init__(self, context: PyTorchTrialContext) -> None:
        '''
        '''
        self.context=context
        self.model_path = self.context.get_hparam('model_path')
        print("self.model_path: ",self.model_path)
        self.tokenizer_path = self.context.get_hparam('tokenizer_path')
        print("self.tokenizer_path: ",self.tokenizer_path)
        self.finetune_results_dir = self.context.get_hparam('finetune_results_dir')
        print("self.finetune_results_dir: ",self.finetune_results_dir)
        self.context=context
        peft_config = LoraConfig(
                                    r=1,
                                    lora_alpha=1,
                                    lora_dropout=0.05,
                                    bias="none",
                                    task_type="CAUSAL_LM",
                                    target_modules=["q_proj","k_proj","v_proj","o_proj","w1","w2","w3","lm_head"]
                                )
        bnb_config = BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold = 6.0)

        self.model = AutoModelForCausalLM.from_pretrained(self.model_path,
                                  load_in_8bit=False,
                                  torch_dtype=torch.bfloat16,
                                  device_map={"": 0},
                                  trust_remote_code=True,
                                  quantization_config=bnb_config,
                                  cache_dir=None,
                                  local_files_only=False)
        # bnb_config=BitsAndBytesConfig(  load_in_4bit= True,
        #                                 bnb_4bit_quant_type= "nf4",
        #                                 bnb_4bit_compute_dtype= torch.bfloat16,
        #                                 bnb_4bit_use_double_quant= False)
        # self.model = AutoModelForCausalLM.from_pretrained(self.model_path,
        #                                       load_in_8bit=False,
        #                                       torch_dtype=torch.bfloat16,
        #                                       device_map={"": 0},
        #                                       trust_remote_code=True,
        #                                       quantization_config=bnb_config,
        #                                       cache_dir=None,
        #                                       local_files_only=False)
        self.model = get_peft_model(self.model, peft_config)
        print(self.model.print_trainable_parameters())
        self.model = self.context.wrap_model(self.model)
        self.batch_size = self.context.get_per_slot_batch_size()
        self.test_batch_size = 1
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        self.tokenizer.padding_side = 'right'
        self.tokenizer.pad_token = self.tokenizer.eos_token

        print(self.tokenizer.add_bos_token, self.tokenizer.add_eos_token)
        dataset_name = self.context.get_hparam('dataset_name')
        self.mapped_dataset = self.get_dataset(dataset_name)

        BATCHES=4
        
        learning_rate = self.context.get_hparam('learning_rate')
        # per_device_train_batch_size= 2
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
                {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': self.context.get_hparam('weight_decay')},
                {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
                ]

        self.optimizer =  self.context.wrap_optimizer(bnb.optim.Adam8bit(optimizer_grouped_parameters, lr=learning_rate))
        
        # These are the only changes you need to make
        # The first part sets the optimizer to use 8-bits
        # The for loop sets embeddings to use 32-bits
        # Thank you @gregorlied https://www.kaggle.com/nbroad/8-bit-adam-optimization/comments#1661976
        for module in self.model.modules():
            if isinstance(module, torch.nn.Embedding):
                bnb.optim.GlobalOptimManager.get_instance().register_module_override(
                    module, 'weight', {'optim_bits': 32}
                )     




        self.data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)
        # get learn rate scheduler
        self.warmup_steps = self.context.get_hparam('warmup_steps')
        self.scheduler = self.context.wrap_lr_scheduler(
            get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=self.warmup_steps,
                                            num_training_steps=16),
            LRScheduler.StepMode.MANUAL_STEP
        )

    def build_callbacks(self) -> Dict[str, PyTorchCallback]:
        return {"my_callbacks": MyCallbacks(cat_mapping=3,
                                            model=self.model,
                                            tokenizer=self.tokenizer,
                                            save_model_dir=self.finetune_results_dir)}
    
    def get_dataset(self,dataset_name):
        '''
        '''
        if dataset_name=='english_to_latex':
            dataset=self.get_eng_to_latex_dataset()
        elif dataset_name=='gath_baize':
            dataset=self.get_gath_baize_dataset()
        elif dataset_name=='hpe_press_release':
            dataset=self.get_hpe_dataset()
            # /mnt/efs/shared_fs/determined/nb_fs/dev-llm-rag-app/data/HPE_qa_dataset.csv
            
        else:
            dataset_path = '/run/determined/workdir/shared_fs/workshop_data/{}.txt'.format(self.dataset_name)
            dataset = TextDataset(
                tokenizer=self.tokenizer,
                file_path=dataset_path,  # Principles of Data Science - Sinan Ozdemir
                block_size=32  # length of each chunk of text to use as a datapoint
            )
        return dataset

    def get_gath_baize_dataset(self):
        '''
        high-quality multi-turn chat corpus by leveraging ChatGPT to engage in a conversation with itself. 
        https://huggingface.co/datasets/gathnex/Gath_baize
        '''
        dataset = load_dataset("/mnt/efs/shared_fs/determined/Gath_baize/", split="train")
        BATCHES=4
        dataset2 = dataset.select(indices=list(range(8*BATCHES)))
        def preprocess(samples):
            samples = self.tokenizer(samples['chat_sample'],truncation=True)# (11.6.2023) should change
            return samples
        mapped_dataset = dataset2.map(preprocess,batched=True ,remove_columns=['chat_sample','dataset_origin'])
        return latmapped_datasetex_data    
    def get_hpe_dataset(self):
        '''
        '''
        # Add our singular prompt
        CONVERSION_PROMPT = 'LCT\n'  # LaTeX conversion task

        CONVERSION_TOKEN = 'LaTeX:'
        # data = pd.read_csv('/mnt/efs/shared_fs/determined/nb_fs/dev-llm-rag-app/data/HPE_qa_dataset.csv')
        #TODO (12/12/2023): update to download dataset 
        data = pd.read_csv('/nvmefs1/shared_nb/01 - Users/cyrill.hug/pdk-llm-rag-demo-houston/data/HPE_qa_dataset.csv')

        # combined_prompt_ans = f'''[INST]Given the document:\n{data['Content']}\n Answer the question: {data['Question']}\n[\INST]Answer: {data['Answer']}'''
        combined_prompt_ans = ('[INST]Given the document:\n'+data['Content']+'\n Answer the question: '+data['Question']+'\n[\INST]Answer: '+data['Answer']).astype(str)
        latex_data = HFDataset.from_pandas( pd.DataFrame({'text':combined_prompt_ans}))  # turn a pandas DataFrame into a Dataset
        def preprocess(examples):  # tokenize our text but don't pad because our collator will pad for us dynamically
            # combined_prompt_ans = f'''[INST]Given the document:\n{examples['Content']}\n Answer the question: {examples['Question']}\n[\INST]Answer: {examples['Answer']}'''

            return self.tokenizer(examples['text'], truncation=True)
        latex_data = latex_data.map(preprocess, batched=True,remove_columns=['text'])
        print("latex_data: ",latex_data)
        return latex_data
    def get_eng_to_latex_dataset(self):
        '''
        '''
        # Add our singular prompt
        CONVERSION_PROMPT = 'LCT\n'  # LaTeX conversion task

        CONVERSION_TOKEN = 'LaTeX:'
        data = pd.read_csv('/mnt/efs/shared_fs/determined/nb_fs/dev-llm-rag-app/notebooks/determined_files/english_to_latex.csv')
        training_examples = ('[INST]'+CONVERSION_PROMPT+'English: ' + data['English'] + '\n' + CONVERSION_TOKEN + ' ' + data['LaTeX']+'[/INST]').astype(str)
        task_df = pd.DataFrame({'text': training_examples})
        latex_data = HFDataset.from_pandas(task_df)  # turn a pandas DataFrame into a Dataset
        def preprocess(examples):  # tokenize our text but don't pad because our collator will pad for us dynamically
            return self.tokenizer(examples['text'], truncation=True)
        latex_data = latex_data.map(preprocess, batched=True,remove_columns='text')
        return latex_data
    def get_batch_length(self, batch):
        '''
        Count the number of records in a given batch.
        Override this method when you are using custom batch types, as produced
        when iterating over the `DataLoader`.
        '''
        return batch['input_ids'].shape[0]
    def format_batch(self,batch,device='cuda'):
        return (batch['input_ids'].to(device),batch['input_ids'].to(device))

    
    def build_training_data_loader(self) -> DataLoader:
        '''
        '''
        return DataLoader(self.mapped_dataset, collate_fn =self.data_collator ,shuffle=True, batch_size=self.batch_size)
    def build_validation_data_loader(self) -> DataLoader:
        '''
        '''
        return DataLoader(self.mapped_dataset, collate_fn =self.data_collator ,shuffle=False, batch_size=self.test_batch_size)
    
    def train_batch(self,batch,epoch_idx, batch_idx):
        '''
        '''
        inputs,labels = self.format_batch(batch)
        # print("inputs: ",inputs.shape)
        outputs = self.model(inputs, labels=labels)
        # print(outputs.keys())
        loss = outputs[0]
        train_result = {
            'loss': loss
        }
        self.context.backward(train_result["loss"])
        self.context.step_optimizer(self.optimizer)
        self.scheduler.step()
        print("lr: ",self.scheduler.get_lr()[0])
        return train_result
    
    def evaluate_batch(self,batch):
        '''
        '''
        inputs,labels = self.format_batch(batch)
        outputs = self.model(inputs, labels=labels)
        lm_loss = outputs[0]
        eval_loss = lm_loss.mean().item()
        perplexity = torch.exp(torch.tensor(eval_loss))

        results = {
            "eval_loss": eval_loss,
            "perplexity": perplexity
        }
        return results
        
        
    
