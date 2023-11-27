import pandas as pd
from sklearn.model_selection import train_test_split

def build_t3_data():
    url = 'https://raw.githubusercontent.com/deckerkrogh/nlp243_data/main/MELD_train_efr.json'
    raw_data = pd.read_json(url)

    t3_train, t3_test = train_test_split(raw_data, test_size=0.2, random_state=1)
    t3_test, t3_dev = train_test_split(t3_test, test_size=0.5, random_state=1)

    t3_train.to_csv('datasets/task3_train.csv')
    t3_test.to_csv('datasets/task3_test.csv')
    t3_dev.to_csv('datasets/task3_dev.csv')


def build_t1_data():
    url = 'https://raw.githubusercontent.com/deckerkrogh/nlp243_data/main/MaSaC_train_erc.json'
    raw_data = pd.read_json(url)
    train, test = train_test_split(raw_data, test_size=0.2, random_state=1)
    test, dev = train_test_split(test, test_size=0.5, random_state=1)

    train.to_csv('datasets/task1_train.csv')
    test.to_csv('datasets/task1_test.csv')
    dev.to_csv('datasets/task1_dev.csv')


def build_t2_data():
    url = 'https://raw.githubusercontent.com/deckerkrogh/nlp243_data/main/MaSaC_train_efr.json'
    raw_data = pd.read_json(url)
    train, test = train_test_split(raw_data, test_size=0.2, random_state=1)
    test, dev = train_test_split(test, test_size=0.5, random_state=1)

    train.to_csv('datasets/task2_train.csv')
    test.to_csv('datasets/task2_test.csv')
    dev.to_csv('datasets/task2_dev.csv')


build_t1_data()
build_t2_data()
build_t3_data()
