mkdir -p data
cd data

url="https://www.dropbox.com/s/lx7qms82s4bsabj/test.zip?dl=0"
name="test"

wget $url -O ${name}.zip
unzip ${name}.zip
rm ${name}.zip

cd ..

