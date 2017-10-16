import sys

import pandas as pd


# removing '?' as the null values
def cleaning(df):
    df = df.dropna()
    column_names = df.columns.values
    for i in column_names:
        df = df[~df[i].isin(['?'])]
    return df


def str_to_int(df):
    column_names = df.columns.values
    for column in column_names:
        col_data = df[column];
        unique = set(col_data)
        metadata = dict()
        for i, value in enumerate(unique):
            metadata[value] = i
        df = df.applymap(lambda s: metadata.get(s) if s in metadata else s)
    return df


if __name__ == "__main__":
    args = sys.argv
    raw_data = pd.read_csv(args[1], skipinitialspace=True, index_col=False, na_values=['?'],
                           na_filter='?')
    '''col_no = []
    for index, rows in raw_data.iterrows():
        col_no.append(index)
    raw_data.insert(loc=0, column='no', value=col_no)'''
    # print(raw_data)
    data = cleaning(raw_data)
    data = str_to_int(data)
    # print(data)
    data.to_csv(args[2], index=False, header=False)
