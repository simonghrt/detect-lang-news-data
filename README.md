# Language detection

This project aims to detect the language of a text with simple ML techniques

For this project, I will use 6 different languages for the first implementation : en, de, fr, ru, es, nl

I can get around 100 articles for each of these languages and I've chosen getting articles about bitcoin on the website newsapi.org

### Requirements

Python version > 3
Scikitlearn installed
Pandas installed

### Usage

You simply need to run this :

``` bash
python3 train_test_ml.py
```

### Todos

* Implement an argument parser in order to know which embedding method and classifier to
  use
* Implement other embedding method
* Implement other classifier method
* Implement a solution with deep learning but will need more data
