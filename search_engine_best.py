import csv
import os
import pandas as pd
from reader import ReadFile
from configuration import ConfigClass
from parser_module import Parse
from indexer import Indexer
from searcher import Searcher, Spell_Searcher


# DO NOT CHANGE THE CLASS NAME
class SearchEngine:

    # DO NOT MODIFY THIS SIGNATURE
    # You can change the internal implementation, but you must have a parser and an indexer.
    def __init__(self, config=None):
        self._config = config
        try:
            self._reader = ReadFile(corpus_path=config.get__corpusPath())
        except:
            self._reader = ReadFile("")
        self._parser = Parse()
        self._indexer = Indexer(config)
        self._model = Spell_Searcher(self._indexer)
        self.last_parquet = False

    # DO NOT MODIFY THIS SIGNATURE
    # You can change the internal implementation as you see fit.
    def build_index_from_parquet(self, fn):
        """
        Reads parquet file and passes it to the parser, then indexer.
        Input:
            fn - path to parquet file
        Output:
            No output, just modifies the internal _indexer object.
        """
        documents_list = self._reader.read_file(fn)

        # Iterate over every document in the file
        number_of_documents = 0
        for idx, document in enumerate(documents_list):
            # parse the document
            parsed_document = self._parser.parse_doc(document)
            number_of_documents += 1
            # index the document data
            if self.last_parquet and idx == len(documents_list) - 1:
                self._indexer.last_doc = True
            self._indexer.add_new_doc(parsed_document)
        print('Finished parsing and indexing.')

    # DO NOT MODIFY THIS SIGNATURE
    # You can change the internal implementation as you see fit.
    def load_index(self, fn):
        """
        Loads a pre-computed index (or indices) so we can answer queries.
        Input:
            fn - file name of pickled index.
        """
        self._indexer.load_index(fn)

    # DO NOT MODIFY THIS SIGNATURE
    # You can change the internal implementation as you see fit.
    def load_precomputed_model(self, model_dir=None):
        """
        Loads a pre-computed model (or models) so we can answer queries.
        This is where you would load models like word2vec, LSI, LDA, etc. and 
        assign to self._model, which is passed on to the searcher at query time.
        """
        pass

    # DO NOT MODIFY THIS SIGNATURE
    # You can change the internal implementation as you see fit.
    def search(self, query):
        """ 
        Executes a query over an existing index and returns the number of 
        relevant docs and an ordered list of search results.
        Input:
            query - string.
        Output:
            A tuple containing the number of relevant search results, and 
            a list of tweet_ids where the first element is the most relavant 
            and the last is the least relevant result.
        """
        searcher = Searcher(self._parser, self._indexer, model=self._model)
        return searcher.search(query)

    def read_queries(self, queries_path, k=None):
        queries_df = pd.read_csv(queries_path, sep='\t')
        queries_only = queries_df["keywords"]
        queries_id = queries_df["query_id"]

        csv_list = [['query', 'tweet', 'y_true']]
        for i, query in enumerate(queries_only):
            length_of_results, tweets = self.search(query)
            query_num = queries_id[i]
            for tweet in tweets:
                csv_line = [query_num, tweet, -1]
                csv_list.append(csv_line)
            break

        with open('spell_engine_results.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(csv_list)

        self.calculate_metrics()

    def calculate_metrics(self):
        our_results = pd.read_csv("spell_engine_results.csv")
        bench_mark = pd.read_csv(os.path.join('data', 'benchmark_lbls_train.csv'))

        for idx, tweet_id in enumerate(our_results["tweet"]):
            row_bench = bench_mark.loc[bench_mark["tweet"] == tweet_id]
            row_results = our_results.loc[our_results["tweet"] == tweet_id]

            print(row_bench["query"].iloc[0])
            print(row_results["query"].iloc[0])
            if row_bench["query"].iloc[0] == row_results["query"].iloc[0]:
                rank = row_bench["y_true"].iloc[0]
                our_results.at[idx, "y_true"] = rank

        our_results.to_csv("spell_engine_results.csv", index=False)


def main():
    bench_data_path = os.path.join('data', 'benchmark_data_train.snappy.parquet')
    bench_lbls_path = os.path.join('data', 'benchmark_lbls_train.csv')
    queries_path = os.path.join('data', 'queries_train.tsv')


    config = ConfigClass()
    reader = ReadFile(config.get__corpusPath())
    search_engine = SearchEngine(config)
    corpus_list = reader.read_corpus()

    for idx, parquet in enumerate(corpus_list):
        if idx == len(corpus_list) - 1:
            search_engine.last_parquet = True
        search_engine.build_index_from_parquet(parquet)
    search_engine.load_index("idx_bench")
    search_engine.read_queries(queries_path)

