**Download model**
```
wget "https://huggingface.co/lllyasviel/misc/resolve/main/fooocus_expansion.bin?download=true" -P checkpoints --content-disposition
mv checkpoints/fooocus_expansion.bin checkpoints/pytorch_model.bin
```


**Run**
```
uvicorn expansion:expansion.app --host 0.0.0.0 --port 10001 --workers 8
```
