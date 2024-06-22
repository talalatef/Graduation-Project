from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

class Embedder:
    def __init__(self, model_name='dmis-lab/biobert-base-cased-v1.1'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()

    def generate_embeddings(self, texts, batch_size=16):
        embeddings = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                inputs = self.tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
                outputs = self.model(**inputs)
                batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
                embeddings.extend(batch_embeddings)
        return np.array(embeddings)
