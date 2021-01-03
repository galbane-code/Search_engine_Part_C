import os

import pandas as pd
from reader import ReadFile
from configuration import ConfigClass
from parser_module import Parse
from indexer import Indexer
from searcher import Searcher
import utils


# DO NOT CHANGE THE CLASS NAME
class SearchEngine:

    # DO NOT MODIFY THIS SIGNATURE
    # You can change the internal implementation, but you must have a parser and an indexer.
    def __init__(self, config=None):
        self._config = config
        self._reader = ReadFile(corpus_path=config.get__corpusPath())
        self._parser = Parse()
        self._indexer = Indexer(config)
        self._model = None
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

    def read_queries(self, queries_path):
        queries_df = pd.read_csv(os.path.join('data', 'queries_train.tsv'), sep='\t')
        queries_only = queries_df["information_need"]

        for query in queries_only:
            self.search(query)


def main():
    bench_data_path = os.path.join('data', 'benchmark_data_train.snappy.parquet')
    bench_lbls_path = os.path.join('data', 'benchmark_lbls_train.csv')
    queries_path = os.path.join('data', 'queries_train.tsv')

    config = ConfigClass()
    reader = ReadFile(config.get__corpusPath())
    search_engine = SearchEngine(config)
    corpus_list = reader.read_corpus()

    # for idx, parquet in enumerate(corpus_list):
    #     if idx == len(corpus_list) - 1:
    #         search_engine.last_parquet = True
    #     search_engine.build_index_from_parquet(parquet)
    search_engine.load_index("inverted_idx")
    search_engine.read_queries(queries_path)

