# smoke_test.py
import os, json
from dotenv import load_dotenv
load_dotenv()
print("DATA_DIR:", os.getenv("DATA_DIR", "data"))
import torch
print("torch version:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("Sample data lines:", sum(1 for _ in open("data/denom_dataset.jsonl")))
