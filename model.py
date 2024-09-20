import warnings
warnings.filterwarnings('ignore')
import os, sys, itertools
import pandas as pd
import torch
import datasets
import transformers
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, VisionEncoderDecoderModel, TrOCRProcessor, default_data_collator
from sklearn.model_selection import train_test_split
from PIL import Image
from torch.utils.data import Dataset
from datasets import load_dataset, load_metric

MODEL_CKPT = "microsoft/trocr-base-printed"
MODEL_NAME =  MODEL_CKPT.split("/")[-1] + "_lp"
NUM_OF_EPOCHS = 300 # epochs
ROOT_DIR = r"data" # root directory
IMAGE_FILE_DIR = "license number image" # image file directory

# Correct CSV file name
csv_file_path = os.path.join(ROOT_DIR, "license number image.csv")  # Update with the correct file name

# Load the CSV file
df = pd.read_csv(csv_file_path)
df = df.drop("Unnamed: 0", axis=1)

# Convert all image values to strings and construct the full path for each image
df['images'] = df['images'].apply(lambda x: os.path.join(ROOT_DIR, IMAGE_FILE_DIR, str(x)))


labels = df.labels.values.tolist()
images = df.images.values.tolist()

MAX_LENGTH = max([len(label) for label in labels])

labels = [label.ljust(MAX_LENGTH) for label in labels]

characters = set(char for label in labels for char in label)
characters = sorted(list(characters))

train_dataset, test_dataset = train_test_split(df, train_size=0.80, random_state=42)

train_dataset.reset_index(drop=True, inplace=True)
test_dataset.reset_index(drop=True, inplace=True)

class LP_Dataset(Dataset):
    
    def __init__(self, df, processor, max_target_length=128):
        self.df = df
        self.processor = processor
        self.max_target_length = max_target_length
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # get file name + text
        file_name = self.df['images'][idx]
        text = self.df['labels'][idx]
        # prepare image (i.e. resize + normalize)
        image = Image.open(file_name).convert("RGB")
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        # add labels (input_ids) by encoding the text
        labels = self.processor.tokenizer(text, padding="max_length", max_length=self.max_target_length).input_ids
        # important: make sure that PAD tokens are ignored by the loss function
        labels = [label if label != self.processor.tokenizer.pad_token_id 
                  else -100 for label in labels]
        
        encoding = {"pixel_values" : pixel_values.squeeze(), "labels" : torch.tensor(labels)}
        return encoding

#processor = TrOCRProcessor.from_pretrained(MODEL_CKPT)
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")
root_dir = ROOT_DIR + IMAGE_FILE_DIR
train_ds = LP_Dataset(df=train_dataset,processor=processor)
test_ds = LP_Dataset(df=test_dataset, processor=processor)


print(f"The training dataset has {len(train_ds)} samples in it.")
print(f"The testing dataset has {len(test_ds)} samples in it.")

model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id

model.config.vocab_size = model.config.decoder.vocab_size

model.config.eos_token_id = processor.tokenizer.sep_token_id
model.config.max_length = 64
model.config.early_stopping = True
model.config.no_repeat_ngram_size = 3
model.config.length_penalty = 2.0
model.config.num_beams = 4

cer_metric = load_metric("cer", trust_remote_code=True)

def compute_metrics(pred):
    label_ids = pred.label_ids
    pred_ids = pred.predictions
    
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)
    
    cer = cer_metric.compute(predictions=pred_str, references=label_str)
    
    return {"cer" : cer}

training_args = Seq2SeqTrainingArguments(
    predict_with_generate=True,
    evaluation_strategy="steps",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    fp16=False, 
    output_dir="./",
    logging_steps=2,
    save_steps=1000,
    eval_steps=200,
    report_to="none"
)

from transformers import default_data_collator

# instantiate trainer
trainer = Seq2SeqTrainer(
    model=model,
    tokenizer=processor.feature_extractor,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    data_collator=default_data_collator,
)

train_results = trainer.train()

trainer.save_model()
trainer.log_metrics("train", train_results.metrics)
trainer.save_metrics("train", train_results.metrics)
trainer.save_state()