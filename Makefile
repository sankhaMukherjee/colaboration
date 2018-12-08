
env:
	python3 -m venv env
	source env/bin/activate.fish
	pip3 install --upgrade pip
	pip3 install -e ./python
	pip3 install tqdm