install:
	pip install -r requirements.txt

clean:
	python3 src/clean_data.py

features:
	python3 src/features.py

train:
	python3 src/train_model.py

run:
	python3 main.py

test:
	python3 -m pytest