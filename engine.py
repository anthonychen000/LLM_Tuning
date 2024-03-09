import torch
from nltk.translate.bleu_score import corpus_bleu

def TrainStep(model: torch.nn.Module,
               iterator,
               optimizer,
               clip,
            ):
    
        model.train()
        epoch_loss = 0
        for batch in iterator:
            print('kek')
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["labels"]
            print('zero grad')
            optimizer.zero_grad()
            print('loss')
            loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
            print('backward')
            loss.backward()
            print('clip')
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            print('step')
            optimizer.step()
            epoch_loss += loss.item()

        return epoch_loss / len(iterator)
    
def Evaluate(model, 
              iterator,
              targets
            ):
    model.eval()
    generated_paraphrases = []
    with torch.no_grad():
        for batch in iterator:
            print('kek1')
            generated_paraphrase = batch["generated_paraphrases"]
            # Forward pass: generate outputs
            print('generate')
            generated_paraphrase = model.tokenizer.batch_decode(generated_paraphrase, skip_special_tokens=True)
            print('append')
            # Append the generated paraphrases to the list
            generated_paraphrases.extend(generated_paraphrase)
        
    corpus_bleu_score = corpus_bleu(targets, generated_paraphrases)
    return corpus_bleu_score
            



