import torch
import os
from typing import List, Tuple, Optional, Dict
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores import SimpleVectorStore
from llama_index.core import VectorStoreIndex
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
        self.token_map: Dict[int, str] = {}  # Maps position to token text
        self.vector_index = VectorStoreIndex([])

    # def store_tokens(self, tokens: List[str]):
    #     """Store token mapping for future reference"""        
    #     print(f"Storing {len(tokens)} tokens")
    #     for token in tokens:
    #         current_index = len(self.token_map)
    #         self.token_map[current_index] = token
    #         print(f"  Stored token #{current_index}: '{token}'")

    def store_text(self, text: str):
        """Store token mapping for future reference"""
        # print("store text called on: ", text)
        words = text.split(" ")
        for word in words:
            current_index = len(self.token_map)
            self.token_map[current_index] = word
            # print(f"  Stored token #{current_index}: '{word}'")
            
    def create_chunk_from_range(self, start: int, end: int) -> TextNode:
        """Create a TextNode from a range of tokens"""
        print(f"Creating chunk from token range {start} to {end}")

        # Create a chunk of text from token map from start to end
        text = ""
        for i in range(start, end):
            # Check that the index exists in the token map
            if i in self.token_map:
                if text:  
                    # Add space between tokens
                    text += " "
                text += self.token_map[i]
        
        node = TextNode(text=text)
        print(f"  Created chunk: \n\n'{text}'\n\n")
        return node
        
    def index_evicted_tokens(self, start: int, end: int):
        """Index evicted tokens in LlamaIndex"""
        print(f"Indexing evicted tokens from {start} to {end}")
        
        # Create a node from the provided range
        chunk = self.create_chunk_from_range(start, end)

        # Store this node in the vector store
        self.vector_index.insert_nodes([chunk])
        print(f"  Indexed chunk in vector store")
        
    def retrieve_relevant_context(self, query_string, top_k=3):
        """Retrieve relevant context from evicted tokens"""
        # Create a string from the prompt tokens
        # query_string = " ".join(query_tokens)
        print(f"Retrieving relevant context for query: '{query_string}'")
        
        # Retrieve the top k similar nodes from vector index
        retriever = self.vector_index.as_retriever()
        retriever.similarity_top_k = top_k
    
        results = retriever.retrieve(query_string)
        print(f"Retrieved {len(results)} relevant contexts:")
        for i, result in enumerate(results, 1):
            print(f"  Context {i}: '{result.text}'")
        
        return [result.text for result in results]
    
    def update_token_map(self, start, end):
        """Update the token map when evicting tokens from the cache"""
        print(f"Updating token map: removing tokens from {start} to {end}")
        
        # First remove evicted tokens
        for i in range(start, end):
            if i in self.token_map:
                print(f"  Removing token #{i}: '{self.token_map[i]}'")
                del self.token_map[i]

        new_tokens = {}
        # Then update the remaining tokens
        for i, entry in enumerate(self.token_map.keys()):
            new_tokens[i] = self.token_map[entry]
        
        self.token_map = new_tokens
        print("  Updated token map:")
        for idx, token in self.token_map.items():
            print(f"    #{idx}: '{token}'")

    def __call__(self, past_key_values):
        if past_key_values is None:
            return None
        
        seq_len = past_key_values[0][0].size(self.k_seq_dim)
        print(f"__call__: Sequence length = {seq_len}, Cache size = {self.cache_size}")
        
        if seq_len <= self.cache_size:
            print("  Sequence length within cache size, no eviction needed")
            return past_key_values
        
            
        evict_start = self.start_size
        evict_end = seq_len - self.recent_size

        print(f"  Evicting tokens from {evict_start} to {evict_end}")
        # print(f" Past key values: {past_key_values}")
        self.index_evicted_tokens(evict_start, evict_end)
        self.update_token_map(evict_start, evict_end)

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
        print(f"evict_for_space: Sequence length = {seq_len}, Num coming = {num_coming}, Mode = {mode}")
        
        if seq_len + num_coming <= self.cache_size:
            print("  Sequence length + incoming tokens within cache size, no eviction needed")
            return past_key_values
        
        # By default, we evict from the middle of the cache
        evict_start = self.start_size
        evict_end = seq_len - self.recent_size + num_coming

        # Evict from the start of the cache instead
        if mode == "evict_start": 
            evict_start = 0
            evict_end = seq_len - self.recent_size + num_coming - self.start_size
        
        print(f"  Evicting tokens from {evict_start} to {evict_end}")
        # print(f" Past key values: {past_key_values}")
        
        # Index tokens that will be evicted in LlamaIndex and update token map
        self.index_evicted_tokens(evict_start, evict_end)
        self.update_token_map(evict_start, evict_end)

        # If we have retrieved context to put back into the cache
        if past_context:
            print(f"  Adding past context: {past_context}")
            embedded_past_context = self.embedding_model.get_text_embedding(past_context)
            print("  Embedded past context for insertion")
            
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
        print(f"evict_range: Sequence length = {seq_len}, Start = {start}, End = {end}")
        
        assert start <= end and end <= seq_len
        
        # Index evicted tokens
        self.index_evicted_tokens(start, end)
        self.update_token_map(start, end)
        
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