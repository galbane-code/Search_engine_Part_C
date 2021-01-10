import copy
import math
import nltk
from spellchecker import SpellChecker
from nltk import pos_tag
from ranker import Ranker
from nltk.corpus import lin_thesaurus as thesaurus
from nltk.corpus import wordnet


# DO NOT MODIFY CLASS NAME
class Searcher:
    # DO NOT MODIFY THIS SIGNATURE
    # You can change the internal implementation as you see fit. The model
    # parameter allows you to pass in a precomputed model that is already in
    # memory for the searcher to use such as LSI, LDA, Word2vec models.
    # MAKE SURE YOU DON'T LOAD A MODEL INTO MEMORY HERE AS THIS IS RUN AT QUERY TIME.
    def __init__(self, parser, indexer, model=None):
        self._parser = parser
        self._indexer = indexer
        self._ranker = Ranker()
        self._model = model
        self._docs_dict = {}
        self.number_of_documents = len(indexer.docs_dict)

    # DO NOT MODIFY THIS SIGNATURE
    # You can change the internal implementation as you see fit.
    def search(self, query, k=None):
        """
        Executes a query over an existing index and returns the number of
        relevant docs and an ordered list of search results (tweet ids).
        Input:
            query - string.
            k - number of top results to return, default to everything.
        Output:
            2 parameters are returned - number of relevant search results, and
            a list of tweet_ids where the first element is the most relavant
            and the last is the least relevant result.
        """
        query_object = self._parser.parse_query(query)

        relevant_docs = self._relevant_docs_from_posting(query_object)
        normalized_query = self.normalized_query(query_object)
        n_relevant = len(relevant_docs)
        ranked_doc_ids = Ranker.rank_relevant_docs(relevant_docs, normalized_query, self._indexer.docs_dict, k)
        return n_relevant, ranked_doc_ids

    # feel free to change the signature and/or implementation of this function
    # or drop altogether.
    def _relevant_docs_from_posting(self, query_object):
        """
        This function loads the posting list and counts the amount of relevant documents per term.
        :param query_object: contains, tokens-frequency dict, query text, length etc.
        :return: dictionary of relevant documents mapping doc_id to document frequency.
        """
        try:
            self._model.query_expansion(query_object)
        except:
            pass

        query_dict = query_object.query_dict
        for term in query_dict:
            if term in self._indexer.inverted_idx:
                continue

            elif term.isupper() and term not in self._indexer.inverted_idx:
                if term.lower() in self._indexer.inverted_idx:
                    query_dict[term.lower()] = query_dict.pop(term)

            elif term.islower() and term not in self._indexer.inverted_idx:
                if term.upper() in self._indexer.inverted_idx:
                    query_dict[term.upper()] = query_dict.pop(term)

        relevant_posting_lists = {}
        for term in query_dict:
            if term in self._indexer.postingDict:
                relevant_posting_lists[term] = self._indexer.postingDict[term]

        self.document_dict_init(relevant_posting_lists, query_object.query_length)

        query_object.query_dict = query_dict

        return self._docs_dict

    def document_dict_init(self, postingDict, query_length):
        """
        calculates tf-idf to every single document relevant for the query
        :param postingDict:
        :param query_length:
        :return:
        """
        tf_idf_list = [0] * query_length

        for idx, (term, doc_list) in enumerate(postingDict.items()):
            for doc_tuple in doc_list:
                if doc_tuple[0] not in self._docs_dict:
                    self._docs_dict[doc_tuple[0]] = tf_idf_list

                try:
                    dfi = self._indexer.inverted_idx[term]
                except:
                    dfi = self._indexer.inverted_idx[term.lower()]

                idf = math.log(self.number_of_documents / dfi, 10)
                tf_idf = idf * doc_tuple[2]

                self._docs_dict[doc_tuple[0]][idx] = tf_idf
                tf_idf_list = [0] * query_length

    def normalized_query(self, query):
        """
       This function normalizes each term in the auery by the max term freq in the query dict.
       :param query: a query object
       :return: normalized query values
       """

        normalized = []
        max_freq_term = query.max_freq_term

        for key in query.query_dict:
            tf = query.query_dict[key]
            normalized.append(tf / max_freq_term)

        return normalized


class Spell_Searcher:
    def __init__(self, indexer):
        self._indexer = indexer
        self.spell = None

    def query_expansion(self, query):
        """
        This function finds a misspelled word and finds its closest similarity.
        first by tracking all of its candidates. the candidate with the most appearances in the inverted index
        will be the "replaced"
        :param query: query dictionary
        :return: query dictionary with replaced correct words.
        """
        try:
            self.spell = SpellChecker(local_dictionary='spell_dict.json', distance=1)
        except:
            pass

        query_dict = query.query_dict
        for term in query_dict:

            if term.lower() not in self._indexer.inverted_idx and term.upper() not in self._indexer.inverted_idx:

                misspelled_checker = self.spell.unknown([term])

                if len(misspelled_checker) != 0:
                    candidates = list(self.spell.edit_distance_1(term))

                    super_candidates = list(self.spell.candidates(term))
                    candidates.extend(super_candidates)

                    max_freq_in_corpus = 0
                    max_freq_name = ''

                    for i, candidate in enumerate(candidates):
                        if candidate in self._indexer.inverted_idx:
                            curr_freq = self._indexer.inverted_idx[candidate]
                            if curr_freq > max_freq_in_corpus:
                                max_freq_in_corpus = curr_freq
                                max_freq_name = candidate

                        elif candidate.upper() in self._indexer.inverted_idx:
                            curr_freq = self._indexer.inverted_idx[candidate.upper()]
                            if curr_freq > max_freq_in_corpus:
                                max_freq_in_corpus = curr_freq
                                max_freq_name = candidate

                    if max_freq_name != '':
                        print(max_freq_name)
                        query_dict[max_freq_name] = query_dict.pop(term)
                    else:
                        continue


class Thesaurus_Searcher:

    def __init__(self, indexer):
        self._indexer = indexer
        w = thesaurus.synonyms("")

    def query_expansion(self, query):
        """
        for each word in query.query_text apply Part Of Speach tagging.
        then, apply thesaurus for finding synonyms of each word in the query.
        expand the query with these synonyms/
        :param query:
        :return:
        """
        query_dict = query.query_dict
        query_length = query.query_length

        thes_dict = {}

        for word in query_dict.keys():
            thes_dict[word] = query_dict[word]
            text = [word]
            word_pos = nltk.pos_tag(text)
            word_pos = self.tag(word_pos[0][1])

            word_list_thesaurus = thesaurus.synonyms(word)

            if word_list_thesaurus:

                word_to_switch_list = []
                max_counter = 10
                chosen_words = []

                if word_pos == "ADJ":
                    word_to_switch_list = word_list_thesaurus[0][1]
                elif word_pos == "NOUN" or word_pos == "PROPN":
                    word_to_switch_list = word_list_thesaurus[1][1]
                elif word_pos == "VERB":
                    word_to_switch_list = word_list_thesaurus[2][1]

                for token in word_to_switch_list:
                    if len(chosen_words) == max_counter:
                        break
                    split_token = token.split(" ")
                    if len(split_token) > 1:
                        continue

                    if token in self._indexer.inverted_idx and token not in query_dict.keys():
                        chosen_words.append(token)

                for words in chosen_words:
                    thes_dict[str(words)] = query_dict[word]

        query.query_length = len(thes_dict)
        query.query_dict = thes_dict

    def tag(self, tag):
        if tag.startswith('J'):
            return "ADJ"
        elif tag.startswith('V'):
            return "VERB"
        else:
            return "NOUN"


class WordNet_Searcher:

    def __init__(self, indexer):
        self._indexer = indexer

    def query_expansion(self, query):
        """
        for each word in query.query_text apply Part Of Speech tagging.
        then, apply Wordnet for finding synonyms, antonyms, hyponyms of each word in the query.
        expand the query with these words.
        :param query:
        :return:
        """

        query_dict = query.query_dict
        query_length = query.query_length
        syn = set()
        ant = set()
        combined = []

        pos_list = nltk.pos_tag(query.tokenized_text)

        for i, word in enumerate(query_dict.keys()):
            wordnet_tag = self.get_wordnet_pos(pos_list[i][1])

            try:
                wordnet_list = wordnet.synsets(word, pos=wordnet_tag)
                synset = wordnet_list[0]
                syn.add(synset.hyponyms()[0].lemmas()[0].name())
                syn.add(synset.hypernyms()[0].lemmas()[0].name())
                syn.add(synset.lemmas()[0].name())

                if synset.lemmas()[0].antonyms():
                    ant.add(synset.lemmas()[0].antonyms()[0].name())
            except:
                combined.append([word, syn | ant])
                continue

            try:
                syn.add(synset.hyponyms()[1].lemmas()[0].name())
                syn.add(synset.lemmas()[1].name())
            except:
                combined.append([word, syn | ant])
                continue

            combined.append([word, syn | ant])
            syn = set()
            ant = set()

        for word_set in combined:
            word = word_set[0]  # word itself
            for term in word_set[1]:  # syns and ants for the word
                if term in self._indexer.inverted_idx and term not in query_dict:
                    query_dict[term] = query_dict[word]
                    query_length += 1

        query.query_length = query_length

    def get_wordnet_pos(self, tag):
        """
        gets Part Of Speech tag and returns corresponding Wordnet tag
        :param tag:
        :return: Wordnet Tag
        """
        if tag.startswith('J'):
            return wordnet.ADJ
        elif tag.startswith('V'):
            return wordnet.VERB
        elif tag.startswith('N'):
            return wordnet.NOUN
        elif tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN


class Mix_Searcher:

    def __init__(self, indexer):
        self._indexer = indexer
        self._model_2 = WordNet_Searcher(indexer)
        self._model_1 = Spell_Searcher(indexer)

    def query_expansion(self, query):
        """
        for each word in query.query_text apply Spell_Searcher and then Wordnet_Searcher.
        check these 2 classes above
        :param query:
        :return:
        """
        self._model_1.query_expansion(query)  # spell checker and then wordnet
        self._model_2.query_expansion(query)