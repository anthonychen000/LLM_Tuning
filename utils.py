import torch

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

def freeze_model(model):
    for name, param in model.named_parameters():
     parts = name.split('.')
     if len(parts) > 4:
         try:
             layer_index = int(parts[4])
             if layer_index < 2:
                 # Freeze the layer or take any other desired action
                 param.requires_grad = False
         except ValueError:
             continue
