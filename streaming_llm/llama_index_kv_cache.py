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

        self.vector_index = VectorStoreIndex([])
        # self.test_retreive = False
        self.counter = 0


        
    def store_tokens(self, tokens: List[str]):
        """Store token mapping for future reference"""
        
        # print(f"Stored tokens from {len(self.token_map)} to {len(self.token_map) + len(tokens)}, len: {len(tokens)}")
        for i, token in enumerate(tokens):
            self.token_map[len(self.token_map)] = token
            
    def create_chunk_from_range(self, start: int, end: int) -> TextNode:
        """Create a TextNode from a range of tokens"""
        # print(f"Creating chunk from {start} to {end}, length of map is {len(self.token_map)}")
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
        return node
        
    def index_evicted_tokens(self, start: int, end: int):
        """Index evicted tokens in LlamaIndex"""
        chunk = self.create_chunk_from_range(start, end)
        
        self.vector_index.insert_nodes([chunk])
        
        # embedding = self.embedding_model.get_text_embedding(chunk.text)
        # self.vector_store.add(embedding, chunk)
        # self.evicted_chunks.append(chunk)
        
    def retrieve_relevant_context(self, query_text, top_k=3):
        """Retrieve relevant context from evicted tokens"""
        retriever = self.vector_index.as_retriever()
        retriever.similarity_top_k = top_k
        # query_embedding = self.embedding_model.get_text_embedding(query_text)
        token_string = " ".join(query_text)

        # results = retriever.retrieve(query_text)
        results = retriever.retrieve(token_string)
        # print(results)
        return [result.text for result in results]

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

    def evict_for_space(self, past_key_values, num_coming, mode="default", past_context=None):
        if past_key_values is None:
            return None
        seq_len = past_key_values[0][0].size(self.k_seq_dim)
        print(f"Evicting for space: {seq_len}, {num_coming}")
        if seq_len + num_coming <= self.cache_size:
            return past_key_values
        
        # default is evict from the middle
        evict_start = self.start_size
        evict_end = seq_len - self.recent_size + num_coming
        self.counter += 1

        if mode == "evict_start": 
            evict_start = 0
            evict_end = seq_len - self.recent_size + num_coming

        
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

        if past_context:
            print("Past key values: ", past_key_values)
            # print("Past context: ", past_context)
            embedded_past_context = self.embedding_model.get_text_embedding(past_context)
            print("Embedded past context: ", embedded_past_context)
            #return embedded_past_context
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
            ] + embedded_past_context
        
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
