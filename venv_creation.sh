# create python venv
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt

# activate jupyter extensions
jupyter contrib nbextension install --user
