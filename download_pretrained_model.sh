mkdir -p photometric
cd photometric

url="https://drive.google.com/open?id=1FxDD9Ee1fruq8sKOwvH6LMFANC8t5pQN"
name="models"

wget $url -O ${name}.zip
unzip ${name}.zip
rm ${name}.zip

cd ..
