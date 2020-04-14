mkdir -p ./pdoc_template
pdoc --html ./pySRURGS.py --template-dir ./pdoc_template/  --force
cp ./html/pySRURGS.html ./index.html
