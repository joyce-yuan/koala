import torch
from typing import List, Tuple, Optional, Dict
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores import SimpleVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding


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
        self.embedding_model = embedding_model or OpenAIEmbedding()
        self.vector_store = SimpleVectorStore()
        self.token_map: Dict[int, str] = {}  # Maps position to token text
        self.evicted_chunks: List[TextNode] = []
        
    def store_tokens(self, tokens: List[str], start_pos: int):
        """Store token mapping for future reference"""
        for i, token in enumerate(tokens):
            self.token_map[start_pos + i] = token
            
    def create_chunk_from_range(self, start: int, end: int) -> TextNode:
        """Create a TextNode from a range of tokens"""
        text = " ".join(self.token_map[i] for i in range(start, end))
        node = TextNode(text=text)
        return node
        
    def index_evicted_tokens(self, start: int, end: int):
        """Index evicted tokens in LlamaIndex"""
        chunk = self.create_chunk_from_range(start, end)
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

    def evict_for_space(self, past_key_values, num_coming):
        if past_key_values is None:
            return None
        seq_len = past_key_values[0][0].size(self.k_seq_dim)
        if seq_len + num_coming <= self.cache_size:
            return past_key_values
            
        # Index tokens that will be evicted
        evict_start = self.start_size
        evict_end = seq_len - self.recent_size + num_coming
        self.index_evicted_tokens(evict_start, evict_end)
        
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
