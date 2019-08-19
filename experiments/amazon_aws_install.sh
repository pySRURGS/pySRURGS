sudo apt-get update
sudo apt-get install -y python3-pip
sudo cp /usr/bin/python3 /usr/bin/python

cd ~
git clone https://github.com/pySRURGS/pySRURGS.git
cd pySRURGS
pip3 install -r requirements.txt --user
cd experiments
vim secret.py