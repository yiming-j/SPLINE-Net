mkdir -p photometric
cd photometric

url="https://www.dropbox.com/s/i2cznkuc0jg8rhn/models.zip?dl=0"
name="models"

wget $url -O ${name}.zip
unzip ${name}.zip
rm ${name}.zip

cd ..
