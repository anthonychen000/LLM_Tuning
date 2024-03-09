from torch.utils.data import DataLoader, Dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration
import pandas as pd

class TrainDataset(Dataset):
    def __init__(self, csv_path, tokenizer, max_source_length, max_target_length, task_prefix="paraphrase: "):
        self.df = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.task_prefix = task_prefix

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        input_sequence = self.task_prefix + self.df['text'].iloc[idx]
        output_sequence = self.df['paraphrase'].iloc[idx]

        return input_sequence, output_sequence

    def collate_fn(self, batch):
        input_sequences, output_sequences = zip(*batch)

        encoding = self.tokenizer(
            [self.task_prefix + seq for seq in input_sequences],
            padding="max_length",
            max_length=self.max_source_length,
            truncation=True,
            return_tensors="pt",
        )

        input_ids = encoding.input_ids
        attention_mask = encoding.attention_mask

        target_encoding = self.tokenizer(
            output_sequences,
            padding="max_length",
            max_length=self.max_target_length,
            truncation=True,
            return_tensors="pt",
        )
        labels = target_encoding.input_ids

        # Replace padding token id's of the labels by -100 so it's ignored by the loss
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
        
class TestDataset(Dataset):
    def __init__(self, csv_path, tokenizer, max_source_length, max_target_length, model, task_prefix="generate a paraphrase:"):
        self.df = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.task_prefix = task_prefix
        self.model = model

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        input_sentence = self.df['text'].iloc[idx]

        # Tokenize and prepare inputs
        inputs = self.tokenizer([self.task_prefix + input_sentence], return_tensors="pt", padding=True, max_length=self.max_source_length, truncation=True)

        # Generate paraphrased sequence
        generated_paraphrase = self.model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            do_sample=False,
            max_length=self.max_target_length
        )

        # Decode and return the generated paraphrased sequence
        return generated_paraphrase
    
    def collate_fn(self, batch):
         return {
            "generated_paraphrases": batch
        }
    
