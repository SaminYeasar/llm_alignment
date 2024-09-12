import torch
from torch.utils.data import Dataset, DataLoader
from transformers import ChameleonForCausalLM, Trainer, TrainingArguments
from torch.nn.utils.rnn import pad_sequence
import jsonlines

from constants_facilitating_image_generation import (
    CHAMELEON_PATH_HF,
    ANOLE_PATH_HF,
    DATASET_TOKENIZED_PATH
)

# Define the dataset class
class TokenizedDataset(Dataset):
    def __init__(self, filepath):
        self.tokenized_data = []
        with jsonlines.open(filepath) as reader:
            for obj in reader:
                self.tokenized_data.append(torch.tensor(obj['image_tokens'], dtype=torch.long))
    
    def __len__(self):
        return len(self.tokenized_data)
    
    def __getitem__(self, idx):
        return self.tokenized_data[idx],

# Define custom collate function for DataLoader
def collate_fn(batch):
    batch_inputs = [item[0] for item in batch]
    batch_inputs_padded = pad_sequence(batch_inputs, batch_first=True, padding_value=-100)

    # Create attention masks
    attention_masks = torch.zeros_like(batch_inputs_padded, dtype=torch.long)
    attention_masks = attention_masks.masked_fill(batch_inputs_padded != -100, 1)
   
    return {'input_ids': batch_inputs_padded, 'attention_mask': attention_masks, 'labels': batch_inputs_padded.clone()}

# ===================== Samin: need to update the following =============
CHAMELEON_PATH_HF='facebook/chameleon-7b'
ANOLE_PATH_HF='/home/v-sarnob/code/llm_alignment/model/Anole_HF'
DATASET_TOKENIZED_PATH = 'anole/facilitating_image_generation/dataset_tokenized.jsonl'
deepspeed_config = 'anole/facilitating_image_generation/ds_config.json'
# =========================================================================

# Initialize the model
#model = ChameleonForCausalLM.from_pretrained(CHAMELEON_PATH_HF) # default
#model = ChameleonForCausalLM.from_pretrained(ANOLE_PATH_HF) # we want to use the finetuned Anole model


from cpo_trainer import PreferenceModel
model = PreferenceModel.from_pretrained(ANOLE_PATH_HF)

model = model.to('cuda')
#from dpo_model import DPO
#model = DPO.from_pretrained(ANOLE_PATH_HF)
print(model)

# Define the range of weights that should remain trainable
image_grad_range = (4, 8196)
train_image_head = True # default
# Define a hook to zero out the gradient for weights outside the trainable range during the backward pass
def zero_out_gradient_text(grad):
    grad[:image_grad_range[0], :] = 0
    grad[image_grad_range[1] + 1:, :] = 0
    return grad

def zero_out_gradient_image(grad):
    grad[image_grad_range[0]: image_grad_range[1], :] = 0
    return grad

# Freeze all layers except the output layer
for param in model.parameters():
    param.requires_grad = False
for param in model.lm_head.parameters():
    # Compute the standard deviation for He initialization
    std_dev = (2.0 / 4096) ** 0.5
    
    # Initialize the specific rows with He initialization
    #param.shape = [65536, 4096]

    # case 1 : default finetune image head
    if train_image_head: 
        param[4:8196] = torch.randn((8196 - 4, 4096)) * std_dev
    # case 2: default finetune text head
    else:
        param[:4] = torch.randn((4, 4096)) * std_dev
        param[8196:] = torch.randn((len(param[8196:]), 4096)) * std_dev


    param.requires_grad = True
    # Register the hook on the weight tensor
    if train_image_head:
        param.register_hook(zero_out_gradient_text) # sets the grad for text-head 0
    else:
        param.register_hook(zero_out_gradient_image) # sets the grad for image-head 0 


##################
# DATASET
##################

# Initialize the dataset
#dataset = TokenizedDataset(DATASET_TOKENIZED_PATH)

from transformers import LlamaTokenizerFast
import os
input_base_path='/home/v-sarnob/code/llm_alignment/model/Anole-7b-v0.1/'
tokenizer = LlamaTokenizerFast(
    tokenizer_file=os.path.join(input_base_path, "tokenizer/text_tokenizer_modified.json"), legacy=False
)
tokenizer.sep_token_id = 8710  # assign <reserved08706> to sep so that we can append it after input text
tokenizer.pad_token_id = 1  # assing <pad> to special pad_token

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# # source: https://github.com/huggingface/trl/blob/main/examples/scripts/cpo.py#L79
# from transformers import AutoTokenizer
# tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3.1-8B-Instruct',use_fast=True)
# if tokenizer.pad_token is None:
#     tokenizer.pad_token = tokenizer.eos_token
# if tokenizer.pad_token_id is None:
#      tokenizer.pad_token_id = tokenizer.eos_token_id

from trl import setup_chat_format
# # # If we are aligning a base model, we use ChatML as the default template
if tokenizer.chat_template is None:
    tokenizer.chat_template = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"


from datasets import load_dataset
from accelerate import PartialState
#ds = load_dataset("Anthropic/hh-rlhf")
ds = load_dataset("trl-internal-testing/hh-rlhf-trl-style")
print(ds.column_names)
def process(row):
    row["chosen"] = tokenizer.apply_chat_template(row["chosen"], tokenize=False)
    row["rejected"] = tokenizer.apply_chat_template(row["rejected"], tokenize=False)
    return row
# Compute that only on the main process for faster data processing.
# see: https://github.com/huggingface/trl/pull/1255
with PartialState().local_main_process_first():
    ds = ds.map(process, num_proc=4)
train_dataset = ds["train"]
eval_dataset = ds["test"]


output_dir='/home/v-sarnob/code/llm_alignment/test'
# Define training arguments

from trl import CPOConfig
training_args = CPOConfig(
    beta=0.1,
    #output_dir=ANOLE_PATH_HF,
    output_dir=output_dir,
    learning_rate=1e-3,
    num_train_epochs=3,
    per_device_train_batch_size=1,
    save_steps=3000,
    fp16=False,
    logging_strategy="steps",
    logging_steps=1,
    deepspeed=deepspeed_config, #"ds_config.json" # NOTE: turning off deepspeed
)

from accelerate.utils import DistributedType
training_args.distributed_state.distributed_type = DistributedType.DEEPSPEED

# Initialize the Trainer with custom collate_fn
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=dataset,
#     data_collator=collate_fn
# )


from cpo_trainer import PreferenceTrainer


# cpo_config = CPOConfig(
#     beta=0.1,
#     args=training_args)

trainer = PreferenceTrainer(
    model=model,
    args=training_args,
    #data_collator=collate_fn,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset, 
    tokenizer=tokenizer,
    callbacks=None, 
    peft_config=None
)


# Train the model
trainer.train()




# Save the model
torch.save(model.state_dict(), ANOLE_PATH_HF / 'pytorch_model.bin')
