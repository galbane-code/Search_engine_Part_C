import pandas as pd
from functools import reduce

DF = pd.DataFrame(columns=['Avg Precision', 'Precision@5', 'Precision@10', 'Precision@50', 'Recall'])


# precision(df, True, 1) == 0.5
# precision(df, False, None) == 0.5
def precision(df, single=False, query_number=None, head=None):
    """
        This function will calculate the precision of a given query or of the entire DataFrame
        :param df: DataFrame: Contains tweet ids, their scores, ranks and relevance
        :param single: Boolean: True/False that tell if the function will run on a single query or the entire df
        :param query_number: Integer/None that tell on what query_number to evaluate precision or None for the entire DataFrame
        :return: Double - The precision
    """
    if single:
        df2 = df[df['query'] == query_number]
        return df2['y_true'].mean()
    else:
        x = df.groupby('query')['y_true'].mean()
        if head == 5:
            DF['Precision@5'] = x
        if head == 10:
            DF['Precision@10'] = x
        if head == 50:
            DF['Precision@50'] = x
        else:
            DF['Avg Precision'] = x
        return df.groupby('query')['y_true'].mean().mean()


def recall_single(df, num_of_relevant, query_number):
    """
        This function will calculate the recall of a specific query or of the entire DataFrame
        :param df: DataFrame: Contains tweet ids, their scores, ranks and relevance
        :param num_of_relevant: Integer: number of relevant tweets
        :param query_number: Integer/None that tell on what query_number to evaluate precision or None for the entire DataFrame
        :return: Double - The recall
    """
    df2 = df[df['query'] == query_number]
    return df2['y_true'].sum() / num_of_relevant


# recall(df, {1:2}, True) == 0.5
# recall(df, {1:2, 2:3, 3:1}, False) == 0.388
def recall(df, num_of_relevant):
    """
        This function will calculate the recall of a specific query or of the entire DataFrame
        :param df: DataFrame: Contains tweet ids, their scores, ranks and relevance
        :param num_of_relevant: Dictionary: number of relevant tweets for each query number. keys are the query number and values are the number of relevant.
        :return: Double - The recall
    """
    rec = 0
    lst = []
    for query_number in num_of_relevant.keys():
        relevant = num_of_relevant.get(query_number)
        recall = recall_single(df, relevant, query_number)
        rec += recall
        lst.append(recall)
    series = pd.Series(lst)
    DF['Recall'] = series
    return rec / len(num_of_relevant)


# precision_at_n(df, 1, 2) == 0.5
# precision_at_n(df, 3, 1) == 0
def precision_at_n(df, query_number=1, n=5):
    """
        This function will calculate the precision of the first n files in a given query.
        :param df: DataFrame: Contains tweet ids, their scores, ranks and relevance
        :param query_number: Integer that tell on what query_number to evaluate precision
        :param n: Total document to splice from the df
        :return: Double: The precision of those n documents
    """

    return precision(df[df['query'] == query_number][:n], True, query_number)


# map(df) == 2/3
def map(df):
    """
        This function will calculate the mean precision of all the df.
        :param df: DataFrame: Contains tweet ids, their scores, ranks and relevance
        :return: Double: the average precision of the df
    """
    acc = 0
    split_df = [pd.DataFrame(y).reset_index() for x, y in df.groupby('query', as_index=True)]
    indices = [sdf.index[sdf['y_true'] == 1].tolist() for sdf in split_df]
    for i, indexes in enumerate(indices):
        pres = [precision_at_n(split_df[i], i + 1, index + 1) for index in indexes]
        acc += reduce(lambda a, b: a + b, pres) / len(indexes) if len(pres) > 0 else 0
    return acc / len(split_df)
