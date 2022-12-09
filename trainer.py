import torch
import librosa
import evaluate

from dataclasses import dataclass
from typing import Any, Dict, List, Union

from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor, WhisperForConditionalGeneration
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer

from datasets import load_dataset, DatasetDict

# DATA LOADING

asr_dataset = DatasetDict()

train_set = load_dataset("csv", data_files='/home/ubuntu/data/train.csv', cache_dir='./data/cache')['train']
test_set = load_dataset("csv", data_files='/home/ubuntu/data/test.csv',  cache_dir='./data/cache')['train']

print(train_set)
print('---')
print(test_set)

asr_dataset["train"] = train_set
asr_dataset["test"] = test_set

print('***')

print(asr_dataset)

# FINE TUNING

feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small", language="Ukrainian", task="transcribe")
processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="Ukrainian", task="transcribe")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")

model.config.forced_decoder_ids = None
model.config.suppress_tokens = []
model.config.max_length = 500

max_label_length = model.config.max_length

MAX_DURATION_IN_SECONDS = 30.0
max_input_length = MAX_DURATION_IN_SECONDS * 16000


def prepare_dataset(batch):
    # load a file
    audio, sr = librosa.load(batch['path'])

    # compute log-Mel input features from input audio array 
    batch["input_features"] = feature_extractor(audio, sampling_rate=16_000).input_features[0]

    # compute input length
    batch["input_length"] = len(batch["input_features"])

    # encode target text to label ids 
    batch["labels"] = tokenizer(batch["sentence"]).input_ids

    # compute labels length **with** special tokens! -> total label length
    batch["labels_length"] = len(batch["labels"])

    return batch


def filter_inputs(input_length):
    """Filter inputs with zero input length or longer than 30s"""
    return 0 < input_length < max_input_length


def filter_labels(labels_length):
    """Filter label sequences longer than max length (448)"""
    return labels_length < max_label_length


asr_dataset = asr_dataset.map(prepare_dataset, remove_columns=asr_dataset.column_names["train"], num_proc=4)

# asr_dataset = asr_dataset.filter(filter_inputs, input_columns=["input_length"])
# asr_dataset = asr_dataset.filter(filter_labels, input_columns=["labels_length"])


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-small-uk",
    per_device_train_batch_size=16,
    gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
    learning_rate=1e-5,
    warmup_steps=500,
    max_steps=4000,
    gradient_checkpointing=True,
    fp16=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=1000,
    eval_steps=1000,
    logging_steps=25,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=False,
)

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=asr_dataset["train"],
    eval_dataset=asr_dataset["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)

trainer.train()
