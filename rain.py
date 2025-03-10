import json
import torch.nn as nn
import torch
import torch.nn.functional as F
from transformers import RobertaTokenizer, RobertaModel

import sys
import os

from .inp import INP

class CustomQueryEncoder(nn.Module):
    """
    A simple custom query encoder.
    Depending on config.query_encoder, you might switch between using
    the INP x_encoder or a custom one.
    """
    def __init__(self, config):
        super().__init__()
        # Example: a single linear layer.
        self.linear = nn.Linear(config.input_dim, config.hidden_dim)

    def forward(self, x):
        # x: [batch_size, num_tokens, input_dim]
        x_mean = x.mean(dim=1)
        return self.linear(x_mean)
    
class RAIN(nn.Module):
    def __init__(self, config):
        super().__init__()  # Add this line
        self.config = config
        self.inp = INP(config)
        self.documents = []
        self.titles = []
        if config.rag:
            # Load documents from the specified JSON file.
            with open(config.documents_path, "r") as f:
                data = json.load(f)
                for elem in data:
                    self.documents.append(elem["text"])
                    self.titles.append(elem["name"])
            
            # Set up the query encoder based on the config.
            if config.query_encoder == "inp":
                self.query_encoder = self.inp.x_encoder
            else:
                self.query_encoder = CustomQueryEncoder(config)
            
            # Use the INP's knowledge encoder as the document encoder.
            self.doc_encoder = self.inp.get_knowledge_embedding
            self.doc_projection = nn.Linear(768, config.hidden_dim)
            
            with torch.no_grad():
                processed_docs = []
                for doc in self.documents:
                    doc_embedding = self.prepare_document(doc)
                    # print(f"Processed doc type: {type(doc_embedding)}")  # Debugging statement
                    processed_docs.append(doc_embedding.squeeze(0))
                
                self.document_embeddings = torch.stack(processed_docs, dim=0)

            # SKIP FOR NOW:  Normalize embeddings for cosine similarity. - we don't do this for  MIPS
            if self.config.similarity_metric == "cosine":
                self.document_embeddings = F.normalize(self.document_embeddings, p=2, dim=-1)
        else:
            self.documents = None

    
    def prepare_document(self, doc):
        """
        Tokenize and encode a document using RoBERTa before passing it to the document encoder.
        """
        # Initialize the tokenizer and model (you might want to cache these if performance is a concern)
        # tokenizer = RobertaTokenizer.from_pretrained("roberta-base", truncation=True, do_lower_case=True)  # TODO: use the knowledge encoder
        # model = RobertaModel.from_pretrained("roberta-base")
        tokenizer = self.inp.latent_encoder.knowledge_encoder.text_encoder.tokenizer
        model = self.inp.latent_encoder.knowledge_encoder.text_encoder.llm

        # Tokenize the document. Setting return_tensors="pt" gives PyTorch tensors.
        encoded_input = tokenizer(doc, return_tensors="pt", truncation=True, padding=True)
        # Move inputs to the correct device (if you have config.device set, e.g., "cpu" or "cuda")
        encoded_input = {k: v.to(self.config.device) for k, v in encoded_input.items()}
        # Get the model's output (here, we take the hidden states)
        with torch.no_grad():
            outputs = model(**encoded_input)[0]  # shape: [batch, seq_len, hidden_size]
        # For simplicity, use the first token's embedding (CLS token) as the document representation.
        doc_embedding = outputs[:, 0, :]  # shape: [1, hidden_size]
        doc_embedding = self.doc_projection(doc_embedding)
        # print(f"prepare_document output type: {type(doc_embedding)}")  # Debugging statement
        return doc_embedding
      

    
    def retrieve_knowledge(self, x_target):
        """
        Compute query embeddings using the query encoder, then calculate similarity scores
        against pre-computed document embeddings using PyTorch. Finally, select the top-k documents,
        concatenate them, and return the resulting knowledge string.
        """
        # Encode x_target using the query encoder.
        if self.config.x_encoder == "cnn":
            # print(f"{x_target.shape=}")
            x_target = x_target.permute(2, 0, 1).unsqueeze(0)
        query_embeddings = self.query_encoder(x_target)
       
        # If query_embeddings has extra dimensions (e.g. [batch, tokens, hidden_dim]),
        # average over tokens to get one vector per instance.
        if query_embeddings.dim() > 2:
            query_embeddings = query_embeddings.mean(dim=1)
        
        # Depending on the similarity metric, normalize embeddings if using cosine similarity.
        if self.config.similarity_metric == "cosine":
            query_embeddings = F.normalize(query_embeddings, p=2, dim=-1)
            doc_embeddings = F.normalize(self.document_embeddings, p=2, dim=-1)
        else:
            doc_embeddings = self.document_embeddings

        # Compute similarity scores using matrix multiplication.
        # scores shape: [batch_size, num_docs]
        scores = torch.matmul(query_embeddings, doc_embeddings.T)

        # Get the top-k document indices for each query.
        topk_values, topk_indices = torch.topk(scores, k=self.config.top_k, dim=-1)
        if self.config.top_k == 1:
            self._topk_titles = self.titles[topk_indices.squeeze()]
        else:
            self._topk_titles = [self.titles[i] for i in topk_indices.squeeze()]
        self._topk_values = topk_values
    
        batch_size = x_target.size(0)
        retrieved_knowledge = []
        for i in range(batch_size):
            # Retrieve the corresponding document texts.
            docs = [self.documents[idx] for idx in topk_indices[i].tolist()]
            concatenated_docs = " ".join(docs)
            retrieved_knowledge.append(concatenated_docs)
        return retrieved_knowledge
    
    def forward(self, x_context, y_context, x_target, y_target):
        if self.config.rag:
            retrieved_knowledge = self.retrieve_knowledge(x_target)
            knowledge = retrieved_knowledge
            # if self.config.verbose:
                # print("Knowledge: ", knowledge)
        if self.config.x_encoder == "cnn":
            x_context = x_context.squeeze(0).permute(0, 3, 1, 2)
            x_target = x_target.permute(2, 0, 1).unsqueeze(0)
        return self.inp(x_context, y_context, x_target, y_target, knowledge)
