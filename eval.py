import torch
import librosa

from evaluate import load

from transformers import WhisperForConditionalGeneration, WhisperProcessor
from datasets import load_dataset

model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
processor = WhisperProcessor.from_pretrained("openai/whisper-small")

test_set = load_dataset("csv", data_files='/home/ubuntu/data/test.csv',  cache_dir='./data/cache')['train']


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
    
    print('Ground truth:', batch["transcription"])
    print('Predicted text:', batch["text"])

    return batch

result = test_set.map(map_to_pred, batched=True, batch_size=1)

wer = load("wer")

print(wer.compute(predictions=ds["text"], references=ds["transcription"]))
