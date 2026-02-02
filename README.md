1.
```bash
conda create -n ivs python=3.12
conda activate ivs
pip install -r requirements.txt
```

2. Install Ollama from https://ollama.com/download
```bash
ollama pull embeddinggemma
```

3. 
```bash
cd data
python embedding.py
```

3.
```bash
cd ../src
python -m workflow.graph
```