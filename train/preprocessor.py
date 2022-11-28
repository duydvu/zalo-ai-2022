

from transformers import AutoTokenizer

ORI_DATASET_PATH = "original_datasets"
PRE_DATASET_PATH = "preprocessed_datasets"
MODEL_PATH = "model"

MAX_LENGTH = 256
STRIDE = 128
model_checkpoint = "nguyenvulebinh/vi-mrc-large"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)


def preprocess_training_data(df_train):
    df_train["question"] = [q.strip() for q in df_train["question"]]
    df_train["context"] = [c.strip() for c in df_train["context"]]
    
    inputs = tokenizer(
        df_train["question"],
        df_train["context"],
        max_length=MAX_LENGTH,
        truncation="only_second",
        stride=STRIDE,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    offset_mapping = inputs.pop("offset_mapping")
    sample_mapping = inputs.pop("overflow_to_sample_mapping")
    
    
    
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):

        input_ids = inputs["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id) #0
        
        sequence_ids = inputs.sequence_ids(i)

        sample_index = sample_mapping[i]
        answers = df_train["answers"][sample_index]
        
        if len(answers["answer_start"]) == 0:
            start_positions.append(cls_index)
            end_positions.append(cls_index)
            
        else:
            start_char = answers["answer_start"][0]
            end_char = answers["answer_start"][0] + len(answers["text"][0])

            context_start = 0
            while sequence_ids[context_start] != 1:
                context_start += 1

            context_end = len(input_ids) - 1
            while sequence_ids[context_end] != 1:
                context_end -= 1

            if not (offset[context_start][0] <= start_char and offset[context_end][1] >= end_char):
                start_positions.append(cls_index)
                end_positions.append(cls_index)
                
            else:
                idx = context_start
                while idx <= context_end and offset[idx][0] <= start_char:
                    idx += 1
                start_positions.append(idx - 1)

                idx = context_end
                while idx >= context_start and offset[idx][1] >= end_char:
                    idx -= 1
                end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs

def preprocess_validation_data(df_val):
    df_val["question"] = [q.strip() for q in df_val["question"]]
    df_val["context"] = [c.strip() for c in df_val["context"]]

    inputs = tokenizer(
        df_val["question"],
        df_val["context"],
        max_length=MAX_LENGTH,
        truncation="only_second",
        stride=STRIDE,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_mapping = inputs.pop("overflow_to_sample_mapping")
    example_ids = []

    for i in range(len(inputs["input_ids"])):
        sample_index = sample_mapping[i]
        example_ids.append(df_val["id"][sample_index])

        sequence_ids = inputs.sequence_ids(i)
        offset = inputs["offset_mapping"][i]
        inputs["offset_mapping"][i] = [
            o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
        ]

    inputs["example_id"] = example_ids
    return inputs

    
    

