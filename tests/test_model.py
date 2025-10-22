# import torch
# from transformers import AutoModel

# print("Step 1: Check CUDA")
# print(torch.cuda.is_available())

# print("Step 2: Load model")
# model = AutoModel.from_pretrained("bert-base-uncased")

# print("Step 3: Model loaded successfully!")
import torch
print("CUDA available:", torch.cuda.is_available())
print("Device:", torch.device("cuda" if torch.cuda.is_available() else "cpu"))