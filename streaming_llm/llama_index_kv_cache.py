import torch
import os
from typing import List, Tuple, Optional, Dict
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores import SimpleVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from transformers import AutoTokenizer, AutoModel


def slice2d(x, start, end):
    return x[:, :, start:end, ...]

def slice3d(x, start, end):
    return x[:, :, :, start:end, ...]

def slice1d(x, start, end):
    return x[:, start:end, ...]

DIM_TO_SLICE = {
    1: slice1d,
    2: slice2d,
    3: slice3d,
}
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

class LlamaIndexKVCache:
    def __init__(
        self,
        start_size=4,
        recent_size=512,
        k_seq_dim=2,
        v_seq_dim=2,
        embedding_model=None,
    ):
        print(f"LlamaIndexKVCache: {start_size}, {recent_size}")
        self.start_size = start_size
        self.recent_size = recent_size
        self.cache_size = start_size + recent_size
        self.k_seq_dim = k_seq_dim
        self.v_seq_dim = v_seq_dim
        self.k_slice = DIM_TO_SLICE[k_seq_dim]
        self.v_slice = DIM_TO_SLICE[v_seq_dim]
        
        # LlamaIndex components
        self.embedding_model = embedding_model or OpenAIEmbedding(model="text-embedding-3-small",)
        self.vector_store = SimpleVectorStore()
        self.token_map: Dict[int, str] = {}  # Maps position to token text
        self.evicted_chunks: List[TextNode] = []

        # self.embedding_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/distilbert-base-nli-stsb-mean-tokens")
        # self.embedding_model = AutoModel.from_pretrained("sentence-transformers/distilbert-base-nli-stsb-mean-tokens")

        
    def store_tokens(self, tokens: List[str]):
        """Store token mapping for future reference"""
        
        print(f"Stored tokens from {len(self.token_map)} to {len(self.token_map) + len(tokens)}, {len(tokens)}")
        for i, token in enumerate(tokens):
            self.token_map[len(self.token_map)] = token
            
    def create_chunk_from_range(self, start: int, end: int) -> TextNode:
        """Create a TextNode from a range of tokens"""
        print(f"Creating chunk from {start} to {end}, length of map is {len(self.token_map)}")
        # text = " ".join(self.token_map[i] for i in range(start, end))
        # create text from token map, we only want to include the numbers in the range that are 
        # present in the token map
        text = ""
        for i in range(start, end):
            if i in self.token_map:
                if text:  # if text is not empty
                    text += " "
                text += self.token_map[i]
        node = TextNode(text=text)
        # node = self.embedding_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        return node
        
    def index_evicted_tokens(self, start: int, end: int):
        """Index evicted tokens in LlamaIndex"""
        chunk = self.create_chunk_from_range(start, end)
        # print(f"Indexing chunk: {chunk.text}")

        # ran into openai.RateLimitError - exceeded current quota
        # print(f"Indexing chunk: {chunk}")
        # with torch.no_grad():
        #     model_output = self.embedding_model(**chunk)
        # embedding = mean_pooling(model_output, chunk["attention_mask"])
        # self.vector_store.add(embedding)

        embedding = self.embedding_model.get_text_embedding(chunk.text)
        self.vector_store.add(embedding, chunk)
        self.evicted_chunks.append(chunk)
        
    def retrieve_relevant_context(self, query_text: str, top_k: int = 3) -> List[str]:
        """Retrieve relevant context from evicted tokens"""
        query_embedding = self.embedding_model.get_text_embedding(query_text)
        results = self.vector_store.query(query_embedding, top_k=top_k)
        return [self.evicted_chunks[idx].text for idx, _ in results]

    def __call__(self, past_key_values):
        if past_key_values is None:
            return None
        seq_len = past_key_values[0][0].size(self.k_seq_dim)
        if seq_len <= self.cache_size:
            return past_key_values
            
        # Index tokens that will be evicted
        evict_start = self.start_size
        evict_end = seq_len - self.recent_size
        self.index_evicted_tokens(evict_start, evict_end)
        
        return [
            [
                torch.cat(
                    [
                        self.k_slice(k, 0, self.start_size),
                        self.k_slice(k, seq_len - self.recent_size, seq_len),
                    ],
                    dim=self.k_seq_dim,
                ),
                torch.cat(
                    [
                        self.v_slice(v, 0, self.start_size),
                        self.v_slice(v, seq_len - self.recent_size, seq_len),
                    ],
                    dim=self.v_seq_dim,
                ),
            ]
            for k, v in past_key_values
        ]

    def evict_for_space(self, past_key_values, num_coming, mode="default"):
        if past_key_values is None:
            return None
        seq_len = past_key_values[0][0].size(self.k_seq_dim)
        print(f"Evicting for space: {seq_len}, {num_coming}")
        if seq_len + num_coming <= self.cache_size:
            return past_key_values
        

        evict_start = self.start_size
        evict_end = seq_len - self.recent_size + num_coming

        if mode == "evict_start": 
            evict_start = 0
            evict_end = seq_len - self.recent_size + num_coming

        ########## default ######################
        # Index tokens that will be evicted

        self.index_evicted_tokens(evict_start, evict_end)

        # update token map according to the eviction
        # first remove the evicted tokens
        for i in range(evict_start, evict_end):
            if i in self.token_map:
                del self.token_map[i]
        
        # then update the remaining tokens
        new_tokens = {}
        for i, entry in enumerate(self.token_map.keys()):
            new_tokens[i] = self.token_map[entry]
        self.token_map = new_tokens
        
        return [
            [
                torch.cat(
                    [
                        self.k_slice(k, 0, self.start_size),
                        self.k_slice(
                            k, seq_len - self.recent_size + num_coming, seq_len
                        ),
                    ],
                    dim=self.k_seq_dim,
                ),
                torch.cat(
                    [
                        self.v_slice(v, 0, self.start_size),
                        self.v_slice(
                            v, seq_len - self.recent_size + num_coming, seq_len
                        ),
                    ],
                    dim=self.v_seq_dim,
                ),
            ]
            for k, v in past_key_values
        ]

    def evict_range(self, past_key_values, start, end):
        if past_key_values is None:
            return None
        seq_len = past_key_values[0][0].size(self.k_seq_dim)
        assert start <= end and end <= seq_len
        
        # Index evicted tokens
        self.index_evicted_tokens(start, end)
        
        return [
            [
                torch.cat(
                    [
                        self.k_slice(k, 0, start),
                        self.k_slice(k, end, seq_len),
                    ],
                    dim=self.k_seq_dim,
                ),
                torch.cat(
                    [
                        self.v_slice(v, 0, start),
                        self.v_slice(v, end, seq_len),
                    ],
                    dim=self.v_seq_dim,
                ),
            ]
            for k, v in past_key_values
        ]
