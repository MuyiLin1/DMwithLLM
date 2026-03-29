import torch
import argparse
from transformers import T5Tokenizer, T5ForConditionalGeneration
from sentence_transformers import SentenceTransformer
import gzip
import pickle
import torch.nn as nn
from tqdm import tqdm

# ==========================================
# 1. Data Structures & Helpers
# ==========================================
# Note: MyDataset definition must remain here so pickle can load it
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, threshold):
        pass 
        
    def __len__(self):
        return self.Xs.shape[0]

    def __getitem__(self, index):
        return self.Xs[index], self.ys[index], self.pre_texts[index], self.post_texts[index], self.text_entities[index]

def loadMyDataset(threshold):
    with gzip.open(f'mydataset.{threshold}.pickle.gz', 'rb') as handle:
        return pickle.load(handle)

def load_model_and_tokenizer(model_name, device):
    tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)
    model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
    return model, tokenizer

def generate_mask_fillings(sentences, model, tokenizer, device):
    input_texts = [f"question: {sent.replace('[MASK]', '<extra_id_0>')}" for sent in sentences]
    inputs = tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)
    output_ids = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=20)
    answers = [tokenizer.decode(ids, skip_special_tokens=True).split('<extra_id_0>')[-1].strip() for ids in output_ids]
    return answers

def language_model_outputs(pre_texts, post_texts, sentence_model, t5_tokenizer, t5_model, entity_model, entity_embd, num_entities, cos, device):
    sentences = [(pre_tokens + ' [MASK]. ' + post_tokens) for pre_tokens, post_texts in zip(pre_texts, post_texts)]
    predicted_entities = generate_mask_fillings(sentences, t5_model, t5_tokenizer, device)
    predicted_entities_embd = torch.FloatTensor(entity_model.encode(predicted_entities)).to(device)
    predicted_entities_embd = torch.unsqueeze(predicted_entities_embd, 1).repeat(1, num_entities, 1)
    entity_embd_expanded = torch.unsqueeze(entity_embd, 0).repeat(len(pre_texts), 1, 1).to(device)
    cos_similarities = cos(predicted_entities_embd, entity_embd_expanded)
    predicted_labels = torch.argmax(cos_similarities, dim=-1)
    return predicted_labels.detach()

def get_embd(dataset):
    temp_d = {}
    for k in dataset.labelfeats.keys():
        id, embd = dataset.labelfeats[k]
        temp_d[id] = torch.FloatTensor(embd)
    return torch.stack([temp_d[k] for k in sorted(temp_d.keys())])

# ==========================================
# 2. The Evaluation Loop
# ==========================================
def evaluate_baseline(dataset, batch_size, cuda, llm_type):
    entity_embd = get_embd(dataset)
    num_entities = entity_embd.shape[0]
    
    if cuda and torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        
    print(f"Running Baseline Test on device: {device}")
    
    t5_model, t5_tokenizer = load_model_and_tokenizer("google/flan-t5-" + llm_type, device)
    entity_model = SentenceTransformer('bert-base-nli-mean-tokens', device=device)
    cos = nn.CosineSimilarity(dim=2, eps=1e-6)
    
    generator = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    
    print(f"Testing Pure LLM Accuracy: {llm_type}...")
    print(f"Total batches to process: {len(generator)}")
    
    total_correct = 0
    total_samples = 0

    for bno, (Xs, ys, pre_texts, post_texts, text_entities) in enumerate(generator):
        ys = ys.to(device)
        
        # Get LLM Predictions
        with torch.no_grad():
            lm_predicted_labels = language_model_outputs(pre_texts, post_texts, entity_model, t5_tokenizer, t5_model, entity_model, entity_embd, num_entities, cos, device)
            
        # Tally correct answers
        correct_in_batch = (lm_predicted_labels == ys).sum().item()
        total_correct += correct_in_batch
        total_samples += len(ys)
        
        current_accuracy = total_correct / total_samples
        
        if bno % 5 == 0:
            print(f"Batch {bno} | Pure LLM Accuracy: {current_accuracy:.4f} ({total_correct}/{total_samples})")
            
    print("\n==========================================")
    print("Baseline Evaluation Complete!")
    print(f"Final LLM Accuracy: {(total_correct / total_samples):.4f}")
    print("==========================================")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--llm_type', type=str, default='large') 
    args = parser.parse_args()
    
    # Using your new 500 threshold dataset
    mydata = loadMyDataset(500)
    evaluate_baseline(mydata, batch_size=128, cuda=True, llm_type=args.llm_type)