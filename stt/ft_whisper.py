from datasets import load_dataset, DatasetDict, Audio
from transformers import WhisperProcessor

common_voice = DatasetDict()
common_voice["train"] = load_dataset("Shekharmeena/Hindi-Female-Single-Speaker-High-Quality-Speech", split="train")

split = common_voice["train"].train_test_split(test_size=0.2, seed=42)

common_voice["train"] = split["train"]
common_voice["test"] = split["test"]

print(common_voice["train"])
print(common_voice["test"])

common_voice = common_voice.select_columns(["audio", "transcript"])

processor = WhisperProcessor.from_pretrained(pretrained_model_name_or_path="openai/whisper-small", language="hi", task="transcribe")

print(common_voice["train"].features)

sampling_rate = processor.feature_extractor.sampling_rate
print(sampling_rate)

common_voice = common_voice.cast_column("audio", Audio(sampling_rate=sampling_rate, num_channels=1))
print(common_voice["train"].features)

#todo: actual tuning goes after this