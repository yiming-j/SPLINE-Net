mkdir -p data
cd data

url="https://drive.google.com/open?id=103sJ8yQ1SF3H7D8Agmc8wyG7bZMTs1Qo"
name="test"

wget $url -O ${name}.zip
unzip ${name}.zip
rm ${name}.zip

cd ..

