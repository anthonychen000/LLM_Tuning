from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch.nn as nn

class CustomModel(nn.Module):  # Add inheritance from nn.Module
    def __init__(self):
        super(CustomModel, self).__init__()
        self.model = T5ForConditionalGeneration.from_pretrained('t5-small')
        self.tokenizer = T5Tokenizer.from_pretrained('t5-small')

        for name, param in self.model.named_parameters():
            parts = name.split('.')
            if len(parts) > 4:
                try:
                    block_number = int(parts[2])
                    layer_index = int(parts[4])
                    if block_number < 5 or layer_index < 2:
                        # Freeze the layer or take any other desired action
                        param.requires_grad = False
                except ValueError:
                    continue
                
    def forward(self, input_ids, decoder_input_ids=None):
        return self.model(
            input_ids=input_ids,
            decoder_input_ids=decoder_input_ids
        )
    
'''
model = CustomModel()

# Print trainable/frozen parameters
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f'Trainable: {name}')
    else:
        print(f'Frozen: {name}')

# Count trainable parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

num_trainable_params = count_parameters(model)
print(f"Number of trainable parameters: {num_trainable_params}")
'''

