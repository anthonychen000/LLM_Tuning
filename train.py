from engine import TrainStep, Evaluate
from torch.utils.data import DataLoader
from dataload import TrainDataset, TestDataset
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import time
from utils import epoch_time
import pandas as pd

epochs = 10
clip = 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = T5ForConditionalGeneration.from_pretrained('t5-small')
for name, param in model.named_parameters():
  parts = name.split('.')
  if len(parts) > 4:
      try:
          layer_index = int(parts[4])
          if layer_index < 2:
              # Freeze the layer
              param.requires_grad = False
      except ValueError:
          continue
      
print('Model Configuration:')
print(model.config)
      
tokenizer = T5Tokenizer.from_pretrained('t5-small')
         
train_path = 'data/train_data.csv'
test_path = 'data/test_data.csv'
df = pd.read_csv(test_path) 
target_sequences = df['paraphrase'].tolist()
max_source_length = 512
max_target_length = 128
batch_size = 32

train_dataset = TrainDataset(train_path, tokenizer, max_source_length, max_target_length)
train_iterator = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=train_dataset.collate_fn)

test_dataset = TestDataset(test_path, tokenizer, max_source_length, max_target_length, model)
test_iterator = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=test_dataset.collate_fn)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)


for epoch in range(epochs):
    start_time = time.time()
    
    train_loss = TrainStep(model, train_iterator, optimizer, clip)
    bleu_score = Evaluate(model, test_iterator, target_sequences)
    
    end_time = time.time()
    
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f}')
    print(f'Bleu Score: {bleu_score}')
     