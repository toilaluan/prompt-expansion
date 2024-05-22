**Download model**
```
wget "https://huggingface.co/lllyasviel/misc/resolve/main/fooocus_expansion.bin?download=true" -P checkpoints --content-disposition
mv checkpoints/fooocus_expansion.bin checkpoints/pytorch_model.bin
```

**Install pytorch and requirements**
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip3 install -r requirements.txt
```

**Run**
```
uvicorn expansion:expansion.app --host 0.0.0.0 --port 10001 --workers 8
```
