import chardet as chardet
import pandas as pd
import string
import re
from nltk.corpus import stopwords

with open('data.csv', 'rb') as f:
    result = chardet.detect(f.read())
encoding = result['encoding']

with open("data.csv", 'r', encoding=encoding) as f:
    data = pd.read_csv(f)


def select_data_we_need(_data):
    _data = data[['text', 'ctext', 'headlines']]
    return _data


def check_data(_data):
    print(_data.head())
    print(_data.info())
    print(_data.describe())
    print("Null Data: " + str(_data.isnull().sum()))


def remove_duplicate(_data):
    _data.drop_duplicates(subset=['text'], inplace=True)
    _data.drop_duplicates(subset=['ctext'], inplace=True)
    _data.drop_duplicates(subset=['headlines'], inplace=True)

    return _data


def transform_to_lowercase(data_selected):
    # Try using .loc[row_indexer,col_indexer] = value instead
    data_selected.loc[:, 'text'] = data_selected['text'].apply(lambda x: x.lower())
    data_selected.loc[:, 'ctext'] = data_selected['ctext'].apply(lambda x: x.lower())
    data_selected.loc[:, 'headlines'] = data_selected['headlines'].apply(lambda x: x.lower())

    return data_selected


def remove_punctuation(_data):
    _data.loc[:, 'text'] = _data['text'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))
    _data.loc[:, 'ctext'] = _data['ctext'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))
    _data.loc[:, 'headlines'] = _data['headlines'].apply(
        lambda x: x.translate(str.maketrans('', '', string.punctuation)))
    return _data


def remove_stop_words(_data):
    stop_words = set(stopwords.words('english'))
    _data['text'] = _data['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))
    _data['ctext'] = _data['ctext'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))
    _data['headlines'] = _data['headlines'].apply(
        lambda x: ' '.join([word for word in x.split() if word not in stop_words]))
    return _data


def remove_non_string(_data):
    _data = _data[_data['text'].apply(lambda x: isinstance(x, str))]
    _data = _data[_data['ctext'].apply(lambda x: isinstance(x, str))]
    _data = _data[_data['headlines'].apply(lambda x: isinstance(x, str))]
    return _data


def remove_null_values(_data):
    _data.dropna(inplace=True)
    return _data


def non_english_values(_data):
    _data = _data[_data['text'].apply(lambda x: x.isascii())]
    _data = _data[_data['ctext'].apply(lambda x: x.isascii())]
    _data = _data[_data['headlines'].apply(lambda x: x.isascii())]
    return _data


def remove_single_char(_data):
    _data['text'] = _data['text'].apply(lambda x: re.sub(r'\s+[a-zA-Z]\s+', ' ', x))
    _data['ctext'] = _data['ctext'].apply(lambda x: re.sub(r'\s+[a-zA-Z]\s+', ' ', x))
    _data['headlines'] = _data['headlines'].apply(lambda x: re.sub(r'\s+[a-zA-Z]\s+', ' ', x))

    return _data


def remove_extra_spaces(_data):
    _data['text'] = _data['text'].apply(lambda x: re.sub(' +', ' ', x))
    _data['ctext'] = _data['ctext'].apply(lambda x: re.sub(' +', ' ', x))
    _data['headlines'] = _data['headlines'].apply(lambda x: re.sub(' +', ' ', x))
    return _data


def rename_columns(_data):
    data.rename(columns={'text': 'summary', 'ctext': 'text', 'headlines': 'query'}, inplace=True)
    print(_data.head())
    print(_data.info())
    print(_data.describe())
    return _data


def save_data(_data, path):
    _data.to_csv(path, index=False)
    print(_data.head())
    print(_data.info())
    print(_data.describe())


def main(_data):
    print("Selecting Data...")
    _data_selected = select_data_we_need(_data)
    print("Checking Data...")
    check_data(_data_selected)
    print("Cleaning Data...")
    print("remove non string")
    _data_remove_non_string = remove_non_string(_data_selected)
    _data_transformed = transform_to_lowercase(_data_remove_non_string)
    print("Transforming to Lowercase...")
    _data_removed_duplicate = remove_duplicate(_data_transformed)
    print("Removing Duplicates...")
    _data_remove_stop_words = remove_stop_words(_data_removed_duplicate)
    print("Removing Stop Words...")
    _data_remove_punctuation = remove_punctuation(_data_remove_stop_words)
    print("Removing Punctuation...")
    _data_remove_null_values = remove_null_values(_data_remove_punctuation)
    print("Removing Null Values...")
    _data_non_english_values = non_english_values(_data_remove_null_values)
    print("Removing Non English Values...")
    _data_remove_single_char = remove_single_char(_data_non_english_values)
    print("Removing Single Characters...")
    _data_remove_extra_spaces = remove_extra_spaces(_data_remove_single_char)
    print("Removing Extra Spaces...")
    data_rename_columns = rename_columns(_data_remove_extra_spaces)
    print("Renaming Columns...")
    save_data(data_rename_columns, 'data_cleaned.csv')
    print("Done")


main(data)
