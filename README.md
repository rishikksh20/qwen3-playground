# Readable Implementation of Qwen3 0.6B Model
## Qwen3 0.6B Model Config :
```python
QWEN3_CONFIG = {
            "vocab_size": 151_936, 
            "context_length": 40_960, 
            "emb_dim": 1024, 
            "n_heads": 16, 
            "n_layers": 28, 
            "hidden_dim": 3072, 
            "head_dim": 128,  
            "qk_norm": True,  
            "n_kv_groups": 8,  
            "rope_base": 1_000_000.0,
            "dtype": torch.bfloat16
        }
```

## Load Huggingface Pretrained Weight and Testing :
```python

import torch

from llm import Qwen3Model
from pathlib import Path
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download
from modules.mapping import load_weights_into_qwen
from modules.sampling import advance_decoding
from modules.tokenizer import Qwen3Tokenizer



config = {
    "vocab_size": 151_936,  # Vocabulary size
    "context_length": 40_960,  # Context length that was used to train the model
    "emb_dim": 1024,  # Embedding dimension
    "n_heads": 16,  # Number of attention heads
    "n_layers": 28,  # Number of layers
    "hidden_dim": 3072,  # Size of the intermediate dimension in FeedForward
    "head_dim": 128,  # Size of the heads in GQA
    "qk_norm": True,  # Whether to normalize queries and keys in GQA
    "n_kv_groups": 8,  # Key-Value groups for grouped-query attention
    "rope_base": 1_000_000.0,  # The base in RoPE's "theta"
    "dtype": torch.bfloat16
}
model = Qwen3Model(dim=config["emb_dim"], depth=config["n_layers"], n_heads=config["n_heads"],
                   num_groups=config["n_kv_groups"], head_dim=config["head_dim"], mlp_dim=config["hidden_dim"],
                   vocab_size=config["vocab_size"], context_length=config["context_length"], dtype=config["dtype"])
device = torch.device("cuda")


# Huggingface Weight loading 
repo_id = "Qwen/Qwen3-0.6B"

local_dir = Path(repo_id).parts[-1]
print("Download the model ...")
weights_file = hf_hub_download(
        repo_id=repo_id,
        filename="model.safetensors",
        local_dir=local_dir,
    )
weights_dict = load_file(weights_file)


load_weights_into_qwen(model, config, weights_dict)
model.to(device)
del weights_dict

hf_hub_download(
    repo_id=repo_id,
    filename="tokenizer.json",
    local_dir=local_dir,
)
tokenizer_file_path = f"Qwen3-0.6B/tokenizer.json"
tokenizer = Qwen3Tokenizer(
    tokenizer_file_path=tokenizer_file_path,
    repo_id=repo_id,
    add_generation_prompt=True,
    add_thinking=True
)


# Prompt 
prompt = "Please explain the climate change and how it impacts our future."
print(f"Prompt : {prompt}")
input_token_ids = tokenizer.encode(prompt)
text = tokenizer.decode(input_token_ids)
print(f"Decoded Text: {text}")

input_token_ids_tensor = torch.tensor(input_token_ids, device=device).unsqueeze(0)

for token in advance_decoding(
        model=model,
        token_ids=input_token_ids_tensor,
        max_new_tokens=8192,
        eos_token_id=tokenizer.eos_token_id,
        temperature=0.9,
        top_k=50,
        top_p=0.95,
        repetition_penalty=1.2,
        window_size=50
):
    token_id = token.squeeze(0).tolist()
    print(
        tokenizer.decode(token_id),
        end="",
        flush=True
    )

```


