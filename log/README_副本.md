# setup
1. conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
2. pip install -r requirements.txt

# run
python main.py -g 0 -d gowalla -s 352 -m ssl -c ./config/ssl.json