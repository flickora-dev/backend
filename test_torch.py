from transformers import BertConfig, BertModel
import torch

config = BertConfig()

with torch.device("meta"):
    model = BertModel(config)

pretrained_model = BertModel.from_pretrained("bert-base-uncased")
model.load_state_dict(pretrained_model.state_dict(), assign=True)

device = "cuda:0"
model.to(device)