import torch
from torch import nn
from dataclasses import dataclass
from typing import Union, List
import yaml
import json
import os
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class TrendAttentionConfig:
    def __init__(
            self,
            input_dim,
            input_activation_name,
            hidden_feature_dim,
            hidden_activation_function_name,
            embedding_dim,
            seq_len,
            num_heads,
            masked_attention,
            use_bias,
            attention_dropout,
            feature_engineering_dropout,
            n_layers,
            task
        ):

        self.activation_map = {
            "Identity": nn.Identity,
            "ELU": nn.ELU,
            "LeakyReLU": nn.LeakyReLU,
            "PReLU": nn.PReLU,
            "ReLU": nn.ReLU,
            "SELU": nn.SELU
        }

        self.input_dim = input_dim
        assert input_activation_name in self.activation_map.keys(), f"The following activation functions are available: {list(self.activation_map.keys())}"
        self.input_activation_function = self.activation_map[input_activation_name]
        self.hidden_feature_dim = hidden_feature_dim
        assert hidden_activation_function_name in self.activation_map.keys(), f"The following activation functions are available: {list(self.activation_map.keys())}"
        self.hidden_activation_function = self.activation_map[hidden_activation_function_name]
        self.embedding_dim = embedding_dim
        self.seq_len = seq_len
        self.num_heads = num_heads
        self.masked_attention = masked_attention
        self.use_bias = use_bias
        self.attention_dropout = attention_dropout
        self.feature_engineering_dropout = feature_engineering_dropout
        self.n_layers = n_layers
        assert task in ("regression", "binary_classification"), "Currently only regression or binary_classification tasks are implemented."
        self.task = task

        self.config_dictionary = {
            "input_dim": input_dim,
            "input_activation_name": input_activation_name,
            "hidden_feature_dim": hidden_feature_dim,
            "hidden_activation_function_name": hidden_activation_function_name,
            "embedding_dim": embedding_dim,
            "seq_len": seq_len,
            "num_heads": num_heads,
            "masked_attention": masked_attention,
            "use_bias": use_bias,
            "attention_dropout": attention_dropout,
            "feature_engineering_dropout": feature_engineering_dropout,
            "n_layers": n_layers,
            "task": task
        }

    @classmethod
    def from_file(cls, file_path):
        """
        Loads configuration from a YAML or JSON file and returns a TrendAttentionConfig instance.
        """
        with open(file_path, "r") as f:
            if file_path.endswith(".yaml") or file_path.endswith(".yml"):
                config_dict = yaml.safe_load(f)
            elif file_path.endswith(".json"):
                config_dict = json.load(f)
            else:
                raise ValueError("Unsupported file format. Please use .yaml, .yml, or .json.")

        return cls(**config_dict)


class FeatureEmbeddings(nn.Module):
    def __init__(self, config: TrendAttentionConfig, device=None):
        super().__init__()
        self.dense = nn.Linear(
            in_features=config.input_dim, 
            out_features=config.embedding_dim, 
            bias=config.use_bias, 
            device=device
        )
        self.activation = config.input_activation_function()
        self.dropout = nn.Dropout(config.feature_engineering_dropout)

    def forward(self, x):
        x = self.dense(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x
    

class PositionalEncoding(nn.Module):
    def __init__(self, config, device=None):
        super().__init__()
        self.embedding_dim = config.embedding_dim
        self.seq_len = config.seq_len

        # Learnable positional embeddings
        self.pe = nn.Parameter(torch.randn(1, config.seq_len, config.embedding_dim, device=device))

    def forward(self, x):
        return x + self.pe[:, :x.shape[1], :]
    

class FeatureMultiHeadAttention(nn.Module):
    def __init__(self, config: TrendAttentionConfig, device=None):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.embedding_dim // config.num_heads
        self.embedding_dim = config.embedding_dim
        self.seq_len = config.seq_len
        self.masked_attention = config.masked_attention
        self.device = device
        
        self.query_layer = nn.Linear(in_features = config.embedding_dim, out_features = config.embedding_dim, bias = config.use_bias, device=device)
        self.key_layer = nn.Linear(in_features = config.embedding_dim, out_features = config.embedding_dim, bias = config.use_bias, device=device)
        self.value_layer = nn.Linear(in_features = config.embedding_dim, out_features = config.embedding_dim, bias = config.use_bias, device=device)
        #self.output_layer = nn.Linear(in_features = config.embedding_dim, out_features = config.embedding_dim, bias = config.use_bias, device=device)

    def forward(self, feature_embeddings):
        batch_size = feature_embeddings.shape[0]
        
        queries = self.query_layer(feature_embeddings).view(batch_size, self.seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        keys = self.key_layer(feature_embeddings).view(batch_size, self.seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        values = self.value_layer(feature_embeddings).view(batch_size, self.seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) / (self.head_dim ** 0.5)

        if self.masked_attention:
            mask_value = torch.finfo(attention_scores.dtype).min  # Dynamically get the lowest float value
            mask = torch.triu(torch.ones(self.seq_len, self.seq_len, device=self.device), diagonal=1)
            mask = (mask * mask_value).unsqueeze(0).unsqueeze(0)  # Shape (1, 1, seq_len, seq_len)
            attention_scores = attention_scores + mask
        
        attention_weights = torch.softmax(attention_scores, dim=-1)
        embeddings = torch.matmul(attention_weights, values)
        
        embeddings = embeddings.transpose(1, 2).contiguous().view(batch_size, self.seq_len, self.embedding_dim)
        #output = self.output_layer(embeddings)
        
        return attention_weights, embeddings
    

class FeatureAttentionOutput(nn.Module):
    def __init__(self, config: TrendAttentionConfig, device = None):
        super().__init__()
        self.dense = nn.Linear(in_features = config.embedding_dim, out_features = config.embedding_dim, bias = config.use_bias, device = device)
        self.LayerNorm = nn.LayerNorm(config.embedding_dim, device = device)
        self.dropout = nn.Dropout(config.attention_dropout)

    def forward(self, attention_embeddings, before_attention_embeddings):
        embedding_states = self.dense(attention_embeddings)
        embedding_states = self.dropout(embedding_states)
        embedding_states = self.LayerNorm(embedding_states + before_attention_embeddings)
        return embedding_states


class FeatureEngineering(nn.Module):
    def __init__(self, config: TrendAttentionConfig, device = None):
        super().__init__()
        self.fc1 = nn.Linear(in_features = config.embedding_dim, out_features = config.hidden_feature_dim, bias = config.use_bias, device = device)
        self.activation = config.hidden_activation_function()
        self.dropout = nn.Dropout(config.feature_engineering_dropout) 
        self.fc2 = nn.Linear(in_features = config.hidden_feature_dim, out_features = config.embedding_dim, bias = config.use_bias, device = device)

    def forward(self, x):
        #residual = x  # residual connection
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        #return x + residual
        return x


class FeatureEngineeringAttentionLayer(nn.Module):   
    def __init__(self, config: TrendAttentionConfig, device = None):
        super().__init__()
        self.attention = FeatureMultiHeadAttention(config, device)
        self.output = FeatureAttentionOutput(config, device)
        self.engineering = FeatureEngineering(config, device)

    def forward(self, feature_embeddings):
        attention_weights, embedding_states = self.attention(feature_embeddings)
        feature_embeddings = self.output(embedding_states, feature_embeddings)
        feature_embeddings = self.engineering(feature_embeddings)

        return attention_weights, feature_embeddings
    

class TrendAttentionDecoder(nn.Module):
    def __init__(self, config, device = None, return_hidden_states = False):
        super().__init__()
        self.config = config
        self.input_embedding_layer = FeatureEmbeddings(config, device)
        self.positional_embedding_layer = PositionalEncoding(config, device)
        self.attention_layers = nn.ModuleList([FeatureEngineeringAttentionLayer(config, device) for _ in range(config.n_layers)])
        self.return_hidden_states = return_hidden_states

    def forward(self, features):
        feature_embeddings = self.input_embedding_layer(features)
        feature_embeddings = self.positional_embedding_layer(feature_embeddings)
        
        attention_states, feature_states = [], []
        for layer in self.attention_layers:
            attention_weights, feature_embeddings = layer(feature_embeddings)
            attention_states.append(attention_weights)
            feature_states.append(feature_embeddings)

        if self.return_hidden_states:
            return attention_states, feature_states
        else:
            return attention_states[-1], feature_states[-1]


class TrendAttentionHead(nn.Module):
    def __init__(self, config: TrendAttentionConfig, device = None):
        super().__init__()
        self.dense = nn.Linear(in_features = config.embedding_dim, out_features = 1, bias = True, device = device)

    def forward(self, feature_embeddings):
        logits = self.dense(feature_embeddings[:, -1, :])
        return logits
    

@dataclass
class TrendAttentionClassifierOutput:
    attention_weights: Union[torch.Tensor, List[torch.Tensor]]
    feature_embeddings: Union[torch.Tensor, List[torch.Tensor]]
    logits: torch.Tensor
    loss: torch.Tensor


class TrendAttentionClassifier(nn.Module):
    def __init__(self, config, device = None, return_hidden_states = False):
        super().__init__()
        self.config = config
        self.decoder = TrendAttentionDecoder(config, device, return_hidden_states = return_hidden_states)
        self.head = TrendAttentionHead(config, device)
        self.return_hidden_states = return_hidden_states
        self.task = config.task

    def forward(self, features, labels):
        attention_weights, feature_embeddings = self.decoder(features)
        if self.return_hidden_states:
            feature_predictions = feature_embeddings[-1]
            logits = self.head(feature_predictions)
        else:
            logits = self.head(feature_embeddings)
        
        if self.task == "binary_classification":
            loss_fun = nn.BCEWithLogitsLoss()
            loss = loss_fun(logits.view(-1), labels.float())
        else:
            loss_fun = nn.MSELoss()
            loss = loss_fun(logits.view(-1), labels.float()) 

        output = TrendAttentionClassifierOutput(
            attention_weights,
            feature_embeddings,
            logits,
            loss
        )

        return output
    
    def save_model(self, save_path: str):
        """
        Saves model state_dict and configuration file in the specified directory.
        """
        os.makedirs(save_path, exist_ok=True)

        # Save config as YAML
        config_file = os.path.join(save_path, "config.yaml")
        with open(config_file, "w") as f:
            yaml.dump(self.config.config_dictionary, f)

        # Save model state_dict
        model_file = os.path.join(save_path, "model.pth")
        torch.save(self.state_dict(), model_file)

        print(f"Model and config saved in: {save_path}")

    @classmethod
    def load_model(cls, save_path, device="cpu"):
        """
        Loads a saved model and its configuration.
        """
        # Load config
        config_file = os.path.join(save_path, "config.yaml")
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Config file not found in {save_path}")

        with open(config_file, "r") as f:
            config_dict = yaml.safe_load(f)

        # Initialize model with loaded config
        config = TrendAttentionConfig(**config_dict)
        model = cls(config, device=device)

        # Load model state_dict
        model_file = os.path.join(save_path, "model.pth")
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"Model file not found in {save_path}")

        model.load_state_dict(torch.load(model_file, map_location=device))
        model.to(device)
        model.eval()

        print(f"Model successfully loaded from {save_path}")
        return model


    

