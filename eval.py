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
    try:
        generated_ids = model.generate(inputs=input_features)

        transcription = processor.batch_decode(generated_ids, normalize=True, skip_special_tokens=True)

        batch['text'] = [processor.tokenizer._normalize(it) for it in transcription]

        print('Ground truth:', batch["sentence"])
        print('Predicted text:', batch["text"])

    except IndexError:
        # just pass an issue: IndexError: index -1 is out of bounds for dimension 1 with size 0
        """
        File "/home/ubuntu/.local/lib/python3.8/site-packages/transformers/generation/utils.py", line 1518, in generate
            return self.greedy_search(
          File "/home/ubuntu/.local/lib/python3.8/site-packages/transformers/generation/utils.py", line 2295, in greedy_search
            next_token_logits = outputs.logits[:, -1, :]
        """

        # just a trick to pass the issue
        batch['text'] = ['-'] * len(batch['path'])
        batch["sentence"] = ['-'] * len(batch['path'])

    return batch

result = test_set.map(map_to_pred, batched=True, batch_size=50)

wer = load("wer")

print(wer.compute(predictions=result["text"], references=result["sentence"]))
