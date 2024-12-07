import torch
import os
from typing import List, Tuple, Optional, Dict
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores import SimpleVectorStore
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding

os.environ["OPENAI_API_KEY"] = "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"

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
        print(f"LlamaIndexKVCache: Initialized with start_size={start_size}, recent_size={recent_size}")
        self.start_size = start_size
        self.recent_size = recent_size
        self.cache_size = start_size + recent_size
        self.k_seq_dim = k_seq_dim
        self.v_seq_dim = v_seq_dim
        self.k_slice = DIM_TO_SLICE[k_seq_dim]
        self.v_slice = DIM_TO_SLICE[v_seq_dim]
        
        # LlamaIndex components
        self.embedding_model = embedding_model or OpenAIEmbedding(model="text-embedding-3-small",)
        self.vector_index = VectorStoreIndex([])

    def store_text(self, text: str):
        """Store token mapping for future reference"""
        node = TextNode(text=text)
        self.vector_index.insert_nodes([node])
            
    def retrieve_relevant_context(self, query_string, top_k=2):
        """Retrieve relevant context from evicted tokens"""
        
        # Retrieve the top k similar nodes from vector index
        retriever = self.vector_index.as_retriever()
        retriever.similarity_top_k = top_k
    
        results = retriever.retrieve(query_string)
        print(f"Retrieved {len(results)} relevant contexts:")
        for i, result in enumerate(results, 1):
            print(f"Context {i}: '{result.text}'")
        
        return [result.text for result in results]

    def __call__(self, past_key_values):
        if past_key_values is None:
            return None
        
        seq_len = past_key_values[0][0].size(self.k_seq_dim)
        # print(f"__call__: Sequence length = {seq_len}, Cache size = {self.cache_size}")
        
        if seq_len <= self.cache_size:
            # print("  Sequence length within cache size, no eviction needed")
            return past_key_values
        
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

    def evict_for_space(self, past_key_values, num_coming, mode="default", past_context=None):
        if past_key_values is None:
            return None
        
        seq_len = past_key_values[0][0].size(self.k_seq_dim)
        # print(f"evict_for_space: Sequence length = {seq_len}, Num coming = {num_coming}, Mode = {mode}")
        
        if seq_len + num_coming <= self.cache_size:
            # print("  Sequence length + incoming tokens within cache size, no eviction needed")
            return past_key_values
        
        # By default, we evict from the middle of the cache
        evict_start = self.start_size
        evict_end = seq_len - self.recent_size + num_coming

        # Evict from the start of the cache instead
        if mode == "evict_start": 
            evict_start = 0
            evict_end = seq_len - self.recent_size + num_coming - self.start_size
                
        # Otherwise return the shortened cache
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