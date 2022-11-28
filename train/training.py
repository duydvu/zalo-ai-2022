

from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from preprocessor import preprocess_training_data, preprocess_validation_data
from datasets import load_dataset
from transformers import TrainingArguments, Trainer


ORI_DATASET_PATH = "original_datasets"
PRE_DATASET_PATH = "preprocessed_datasets"
MODEL_PATH = "model"

model_checkpoint = "nguyenvulebinh/vi-mrc-large"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)


all_data = load_dataset('json', data_files=f'{PRE_DATASET_PATH}/final_dataset_squadv2.jsonl')['train']

raw_datasets = all_data.train_test_split(test_size=0.1)

train_dataset = raw_datasets["train"].map(
    preprocess_training_data,
    batched=True,
    remove_columns=raw_datasets["train"].column_names,
    num_proc=3
)

validation_dataset = raw_datasets["test"].map(
    preprocess_validation_data,
    batched=True,
    remove_columns=raw_datasets["test"].column_names,
    num_proc=3
)

epoch = 10
batch_size = 16

training_args = TrainingArguments(
    output_dir=f"{MODEL_PATH}/bert-finetuned-squad-ver3",
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    learning_rate=1e-4,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=epoch,
    warmup_ratio=0.05,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    tokenizer=tokenizer
)

trainer.train()

