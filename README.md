# Whisper Ukrainian

Trainer and Evaluation scripts for fine-tuning Whisper models for the Ukrainian language

## How to use?

1. Prepare your data in CSV format

Example `train.csv` and `test.csv`:

```
path,sentence
/path/to/1.wav,добрий вечір ми з україни
```

2. Edit `trainer.py` to select preferred model size and to fix paths to your CSV files

3. Run `python3 trainer.py`

## Results

- Fine-tuned small Whisper model: https://huggingface.co/Yehor/whisper-small-ukrainian
