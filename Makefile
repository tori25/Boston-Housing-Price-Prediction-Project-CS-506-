.PHONY: all install collect clean features visualize train run test

# Run the full pipeline
all: install collect clean features visualize train

install:
	pip install -r requirements.txt

collect:
	python3 src/collect_data.py

clean:
	python3 src/clean_data.py

features:
	python3 src/features.py

visualize:
	python3 src/visualize.py

train:
	python3 src/train_model.py

run:
	python3 main.py

test:
	python3 -m pytest src/tests/test_project.py -v
