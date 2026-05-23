run:
	python -m src.main

listen:
	uvicorn src.main:src --reload

freeze:
	pip freeze > requirements.txt

install:
	pip install -r requirements.txt

clean:
	find . -type d -name "__pycache__" -exec rm -r {} + && find . -type f -name "*.pyc" -delete