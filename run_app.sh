# Run as in bash shell - first installs associated unity avent repo
# (not located) in this repo - install v0.4.0 from: Unity's ml-agents and place
# the python file in the same directory as this app.

python -m pip install ./python
python -m pip install -r requirements.txt

python main.py
