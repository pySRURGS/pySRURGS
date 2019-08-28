sudo apt-get update
sudo apt-get install -y python3-pip
sudo cp /usr/bin/python3 /usr/bin/python
sudo apt-get install -y git vim 
cd ~
git clone https://github.com/pySRURGS/pySRURGS.git
cd pySRURGS
pip3 install -r requirements.txt --user
cd experiments
pip3 install -r requirements.txt --user
vim secret.py