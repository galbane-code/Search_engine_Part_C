# DO NOT MODIFY CLASS NAME
from datetime import datetime
import utils
from parser_module import Parse


class Indexer:
    # DO NOT MODIFY THIS SIGNATURE
    # You can change the internal implementation as you see fit.
    def __init__(self, config):
        self.docs_dict = {}
        self.inverted_idx = {}
        self.postingDict = {}
        self.spell_dict = {}
        self.not_finished_capital = {}
        self.config = config
        self.index_path = config.savedFileMainFolder
        self.last_doc = False

    # DO NOT MODIFY THIS SIGNATURE
    # You can change the internal implementation as you see fit.
    def add_new_doc(self, document):
        """
        This function perform indexing process for a document object.
        Saved information is captured via two dictionaries ('inverted index' and 'posting')
        in addition, Spell Correction dictionary is saved.
        :param document: a document need to be indexed.
        :return: -
        """

        document_dictionary = document.term_doc_dictionary
        max_freq_term = document.max_freq_term
        # Go over each term in the doc
        for term in document_dictionary.keys():
            try:
                # Update inverted index and posting
                if term not in self.inverted_idx.keys():
                    self.inverted_idx[term] = 1
                    self.postingDict[term] = []
                    self.spell_dict[term] = document_dictionary[term]
                else:
                    self.inverted_idx[term] += 1
                    self.spell_dict[term] += document_dictionary[term]

                term_freq = document_dictionary[term]
                self.postingDict[term].append((document.tweet_id, document_dictionary[term], term_freq / max_freq_term))

                if document.tweet_id not in self.docs_dict:
                    tweet_date = document.tweet_date
                    date_in_hours = self.date_diff(tweet_date)
                    self.docs_dict[document.tweet_id] = [document.doc_length, date_in_hours, max_freq_term]

            except:
                print('problem with the following key {}'.format(term[0]))

        if self.last_doc:
            self.remove_capital_entity()
            self.save_index("inverted_idx.pkl")
            self.save_spell("spell_dict")

    def date_diff(self, tweet_date):
        """
        calculates tweet date in minutes
        :param tweet_date:
        :return:
        """
        current_time = datetime.now()

        tweet_date_as_a_DATE = datetime.strptime(tweet_date, '%a %b %d %H:%M:%S +0000 %Y')
        date_sub = current_time - tweet_date_as_a_DATE

        date_in_minutes = int((date_sub.days * 60 * 24) + (date_sub.seconds // 3600))
        return date_in_minutes

    def remove_capital_entity(self):
        """
        when indexing is finished, we remove entities that appear in the corpus
        less then 2 times, 5 most frequent words in the corpus.
        in addition, words with upper_case appearance that have lower-case appearance, are removed
        and their values are added to the lower-case terms.
        :return:
        """
        keys_to_remove = {'covid', '19', 'mask', 'wear', 'coronavirus', 'virus'}
        for key in self.inverted_idx:
            if key in Parse.ENTITY_DICT and Parse.ENTITY_DICT[key] < 2:
                if key in self.inverted_idx:
                    keys_to_remove.add(key)

            elif key in Parse.CAPITAL_LETTER_DICT and Parse.CAPITAL_LETTER_DICT[key] is False:
                if key in self.inverted_idx:
                    count_docs = self.inverted_idx[key]
                    posting_file = self.postingDict[key]
                    keys_to_remove.add(key)

                    if key.lower() in self.postingDict:
                        self.inverted_idx[key.lower()] += count_docs
                        self.postingDict[key.lower()].extend(posting_file)
                    else:
                        self.not_finished_capital[key.lower] = [count_docs, posting_file]

            if key in self.not_finished_capital:
                count_docs_to_delete = self.not_finished_capital[key][0]
                posting_dict_to_delete = self.not_finished_capital[key][1]
                self.inverted_idx[key.lower()] += count_docs_to_delete
                self.postingDict[key.lower()].extend(posting_dict_to_delete)

            if self.inverted_idx[key] == 1:
                keys_to_remove.add(key)

        for key in keys_to_remove:
            del self.inverted_idx[key]
            del self.postingDict[key]

    # DO NOT MODIFY THIS SIGNATURE
    # You can change the internal implementation as you see fit.
    def load_index(self, fn):
        """
        Loads a pre-computed index (or indices) so we can answer queries.
        Input:
            fn - file name of pickled index.
        """
        objects = utils.load_obj(self.index_path+fn)
        self.postingDict = objects[0]
        self.inverted_idx = objects[1]
        self.docs_dict = objects[2]

    # DO NOT MODIFY THIS SIGNATURE
    # You can change the internal implementation as you see fit.
    def save_index(self, fn):
        """
        Saves a pre-computed index (or indices) so we can save our work.
        Input:
              fn - file name of pickled index.
        """
        utils.save_obj((self.postingDict, self.inverted_idx, self.docs_dict), fn)

    def save_spell(self, fn):
        utils.save_json_file(self.spell_dict, fn)

    # feel free to change the signature and/or implementation of this function
    # or drop altogether.
    def _is_term_exist(self, term):
        """
        Checks if a term exist in the dictionary.
        """
        return term in self.postingDict

    # feel free to change the signature and/or implementation of this function 
    # or drop altogether.
    def get_term_posting_list(self, term):
        """
        Return the posting list from the index for a term.
        """
        return self.postingDict[term] if self._is_term_exist(term) else []