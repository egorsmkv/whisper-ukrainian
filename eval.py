import torch
import librosa

from evaluate import load

from transformers import WhisperForConditionalGeneration, WhisperProcessor
from datasets import load_dataset

model = WhisperForConditionalGeneration.from_pretrained(".....")
processor = WhisperProcessor.from_pretrained("......")

test_set = load_dataset("csv", data_files='/home/yehor/ext-disk/whisper-ukrainian/data/testset.csv',  cache_dir='./data/cache')['train']


def map_to_pred(batch):
    # load a file
    audio, sr = librosa.load(batch['path'])

    input_features = processor(audio, return_tensors="pt").input_features

    with torch.no_grad():
        logits = model(input_features.to("cuda")).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids, normalize = True)

    batch['text'] = processor.tokenizer._normalize(batch['text'])
    batch["transcription"] = transcription

    return batch

result = librispeech_eval.map(map_to_pred, batched=True, batch_size=1)

wer = load("wer")

print(wer.compute(predictions=ds["text"], references=ds["transcription"]))
