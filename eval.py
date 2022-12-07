import torch
import librosa

from evaluate import load

from transformers import WhisperForConditionalGeneration, WhisperProcessor
from datasets import load_dataset

print('libs loaded')

model = WhisperForConditionalGeneration.from_pretrained("/home/ubuntu/whisper-ukrainian/whisper-small-uk/checkpoint-test")
model.to('cuda')
processor = WhisperProcessor.from_pretrained("/home/ubuntu/whisper-ukrainian/whisper-small-uk/checkpoint-test")

print('models loaded')

test_set = load_dataset("csv", data_files='/home/ubuntu/data/test.csv',  cache_dir='./data/cache')['train']

print('dataset loaded')

def map_to_pred(batch):
    # load files
    audios = []
    for path in batch['path']:
        audio, _ = librosa.load(path)
        audios.append(audio)

    input_features = processor(audios, return_tensors="pt", sampling_rate=16_000).input_features
    input_features = input_features.to('cuda')
    generated_ids = model.generate(inputs=input_features)

    # with torch.no_grad():
    #    generated_ids = model.generate(input_ids=input_features)
    #    #logits = model(input_features.to("cuda")).logits

    #predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(generated_ids, normalize=True, skip_special_tokens=True)

    batch['text'] = [processor.tokenizer._normalize(it) for it in batch['sentence']]
    batch["transcription"] = transcription

    print('Ground truth:', batch["transcription"])
    print('Predicted text:', batch["text"])

    return batch

result = test_set.map(map_to_pred, batched=True, batch_size=1)

wer = load("wer")

print(wer.compute(predictions=ds["text"], references=ds["transcription"]))
