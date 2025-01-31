# pip installs
# pip install -q datasets requests torch peft bitsandbytes transformers trl accelerate sentencepiece tiktoken matplotlib

# imports

import os
import re
import math
from tqdm import tqdm
#from google.colab import userdata
from huggingface_hub import login
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, set_seed
from peft import LoraConfig, PeftModel
from datasets import load_dataset, Dataset, DatasetDict
from datetime import datetime
import matplotlib.pyplot as plt
import pickle
from transformers import LlamaForCausalLM, LlamaTokenizer, LlamaConfig

# Tokenizers

LLAMA_3_1 = "meta-llama/Meta-Llama-3.1-8B"

# Constants

BASE_MODEL = LLAMA_3_1
HF_USER = "markobo"
DATASET_NAME = f"{HF_USER}/appliances"
MAX_SEQUENCE_LENGTH = 182
QUANT_4_BIT = True

# Used for writing to output in color

GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"
COLOR_MAP = {"red":RED, "orange": YELLOW, "green": GREEN}

#%matplotlib inline


# Log in to HuggingFace

#hf_token = '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'
#login(hf_token, add_to_git_credential=True)

def investigate_tokenizer(model_name):
  print("Investigating tokenizer for", model_name)
  tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
  for number in [0, 1, 10, 100, 999, 1000]:
    tokens = tokenizer.encode(str(number), add_special_tokens=False)
    print(f"The tokens for {number}: {tokens}")


    # Now we will try this with each model: LLAMA_3_1, QWEN_2_5, GEMMA_2, PHI_3

investigate_tokenizer(LLAMA_3_1)

dataset = load_dataset(DATASET_NAME)
train = dataset['train']
test = dataset['test']



import pickle
import os
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch

def load_model(model_path='..\\week8\\base_model_jan31.pkl'):
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    try:
        # Validate path
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found at {model_path}")

        # Load model with safety checks
        with open(model_path, 'rb') as f:
            # Basic security check
            if os.path.getsize(model_path) > 10 * 1024 * 1024 * 1024:  # 10GB limit
                raise ValueError("File size exceeds safety limit")
            
            base_model = pickle.load(f)
            
            # Validate loaded model
            if not hasattr(base_model, 'forward'):
                raise ValueError("Loaded object does not appear to be a valid model")
            
            return base_model, tokenizer

    except (pickle.UnpicklingError, ModuleNotFoundError) as e:
        raise RuntimeError(f"Error loading model: {str(e)}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error: {str(e)}")

# Initialize model and tokenizer
base_model, tokenizer = load_model()

print(base_model)

print(f"Memory footprint: {base_model.get_memory_footprint() / 1e9:.1f} GB")

def extract_price(s):
    if "Price is $" in s:
      contents = s.split("Price is $")[1]
      contents = contents.replace(',','').replace('$','')
      match = re.search(r"[-+]?\d*\.\d+|\d+", contents)
      return float(match.group()) if match else 0
    return 0


def model_predict(prompt):
    try:
        set_seed(42)
        inputs = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
        attention_mask = torch.ones(inputs.shape, device="cuda")
        outputs = base_model.generate(
            inputs, 
            max_new_tokens=4, 
            attention_mask=attention_mask, 
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id
        )
        response = tokenizer.decode(outputs[0])
        return extract_price(response)
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return 0


class Tester:

    def __init__(self, predictor, data, title=None, size=25): #original size=250
        self.predictor = predictor
        self.data = data
        self.title = title or predictor.__name__.replace("_", " ").title()
        self.size = size
        self.guesses = []
        self.truths = []
        self.errors = []
        self.sles = []
        self.colors = []

    def color_for(self, error, truth):
        if error<40 or error/truth < 0.2:
            return "green"
        elif error<80 or error/truth < 0.4:
            return "orange"
        else:
            return "red"

    def run_datapoint(self, i):
        datapoint = self.data[i]
        guess = self.predictor(datapoint["text"])
        truth = datapoint["price"]
        error = abs(guess - truth)
        log_error = math.log(truth+1) - math.log(guess+1)
        sle = log_error ** 2
        color = self.color_for(error, truth)
        title = datapoint["text"].split("\n\n")[1][:20] + "..."
        self.guesses.append(guess)
        self.truths.append(truth)
        self.errors.append(error)
        self.sles.append(sle)
        self.colors.append(color)
        print(f"{COLOR_MAP[color]}{i+1}: Guess: ${guess:,.2f} Truth: ${truth:,.2f} Error: ${error:,.2f} SLE: {sle:,.2f} Item: {title}{RESET}")

    def chart(self, title):
        max_error = max(self.errors)
        plt.figure(figsize=(12, 8))
        #max_val = max(max(self.truths), max(self.guesses))
        max_val = 200
        plt.plot([0, max_val], [0, max_val], color='deepskyblue', lw=2, alpha=0.6)
        plt.scatter(self.truths, self.guesses, s=3, c=self.colors)
        plt.xlabel('Ground Truth')
        plt.ylabel('Model Estimate')
        plt.xlim(0, max_val)
        plt.ylim(0, max_val)
        plt.title(title)
        plt.show()

    def report(self):
        average_error = sum(self.errors) / self.size
        rmsle = math.sqrt(sum(self.sles) / self.size)
        hits = sum(1 for color in self.colors if color=="green")
        title = f"{self.title} Error=${average_error:,.2f} RMSLE={rmsle:,.2f} Hits={hits/self.size*100:.1f}%"
        self.chart(title)

    def run(self):
        self.error = 0
        for i in range(self.size):
        #for i in range(5):
            self.run_datapoint(i)
        self.report()

    @classmethod
    def test(cls, function, data):
        cls(function, data).run()

Tester.test(model_predict, test)
