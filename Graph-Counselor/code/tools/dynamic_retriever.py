import json
import faiss
import numpy as np
import os
from sentence_transformers import SentenceTransformer

class DynamicFewShotRetriever:
    def __init__(self, bank_path, model_name='sentence-transformers/all-mpnet-base-v2', device='cpu'):
        """
        初始化检索器：加载JSON，计算Embedding，建立索引
        """
        print(f"Loading Few-Shot Bank from: {bank_path}")
        if not os.path.exists(bank_path):
            raise FileNotFoundError(f"Bank file not found: {bank_path}")
            
        with open(bank_path, 'r', encoding='utf-8') as f:
            self.examples = json.load(f)
        
        # 提取所有问题用于构建索引
        self.questions = [item['question'] for item in self.examples]
        
        print(f"Loading Embedding Model: {model_name}...")
        # 如果你有GPU，可以将device设为 'cuda'
        self.encoder = SentenceTransformer(model_name, device=device)
        
        print("Building Vector Index...")
        embeddings = self.encoder.encode(self.questions, show_progress_bar=True)
        embeddings = np.array(embeddings).astype('float32')
        
        # 归一化，以便计算余弦相似度
        faiss.normalize_L2(embeddings)
        
        self.dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(self.dimension) # Inner Product (Cosine Sim)
        self.index.add(embeddings)
        print(f"Index built with {self.index.ntotal} examples.")

    def retrieve(self, query, k=3):
        """
        检索与 query 最相似的 k 个示例，并拼接成字符串返回
        """
        # 确保 k 不超过库里示例的总数
        k = min(k, len(self.examples))
        
        query_vec = self.encoder.encode([query]).astype('float32')
        faiss.normalize_L2(query_vec)
        
        distances, indices = self.index.search(query_vec, k)
        
        retrieved_texts = []
        for idx in indices[0]:
            if idx != -1 and idx < len(self.examples):
                # 这里的 'trace' 必须和你 JSON 里的 key 保持一致
                trace = self.examples[idx].get('trace', '')
                if trace:
                    retrieved_texts.append(trace)
        
        # 用换行符拼接所有找到的示例
        return "\n\n".join(retrieved_texts)