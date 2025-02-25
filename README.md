```bash
git clone https://github.com/darinchau/project-remucs.git
cd project-remucs
python -m venv .venv
source .venv/bin/activate
git submodule update --init --recursive
pip install -r requirements.txt
pip install torchvision
pip install google-cloud-storage
```

Train VQVAE
```bash
sudo apt-get install tmux
tmux new -s session_name
python -m scripts.train_vqvae

# Go to another session
# This deletes old models. Not strictly necessary
python -m scripts.monitor

# To attach back to session
tmux attach -t session_name
```
