# setup
- conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
- pip install -r requirements.txt

# run
- mkdir log
- python main.py -g 0 -d gowalla -s 352 -m ssl -c ./config/ssl.json
