""" TODO: - Noun - verb agreement
          - Rewrite bigram checker
"""

import nltk
from nltk.util import ngrams
from collections import Counter
import os
import pickle
import spacy
import re
from spacy.matcher import Matcher, DependencyMatcher
from spacy.tokens import Doc
from utils import arpabet_consonants, uncountables, uncountable_change
from prep_synt_ngram_creator import (load_tag_ngram_file, prep_pattern,
                                    load_notag_ngram_filename)


class Detector:
    """ A class to represent a grammatical error detection system.
    """

    def __init__(self):
        self.data_folder = r'/home/cyril/Desktop/Python/projects/error_detection/data'
        self.bigrams = self.load_bigrams()
        self.nlp = spacy.load('en_core_web_sm')
        self.arpabet = nltk.corpus.cmudict.dict()
        self.prep_tag_trigrams_file = 'trigrams_synt.csv'
        self.prep_notag_trigrams_file = 'trigrams_notag.csv'

    def load_bigrams(self):
        """ Loads the pickled dict with preprocesed bigrams
        """
        with open(os.path.join(self.data_folder, 'twitter.pickle'), 'rb') as f:
            return pickle.load(f)

    def __call__(self, text: str, vanilla_bigram_checker=False, 
                 noun_phrase_checker=True, preposition_checker=True, save_file=None):
        """ Find errors in text sentence-wise
        """
        
        if vanilla_bigram_checker:
            self.vanilla_bigram_check(text)

        doc = self.nlp(text)

        for sent in doc.sents:
            sent = sent.as_doc()                     # Convert each sentence into a Doc object
            sent_pack = {'original': sent.text, 'corrected': sent, 'corrections': []}           

            if noun_phrase_checker:
                sent_pack = self.noun_phrase_check(sent_pack)

            if preposition_checker:
                sent_pack = self.preposition_check(sent_pack)


            if sent_pack['corrections'] == []:
                sent_pack['corrections'] = ['A -1 -1|||noop|||-NONE-|||REQUIRED|||-NONE-|||0']

            if save_file:
                with open(save_file, 'a') as f:
                    f.write('S ' + sent_pack['original'] + '\n')
                    f.write('\n'.join(sent_pack['corrections']))
            else:
                print('S', sent.text)
                print(*sent_pack['corrections'], sep='\n')
                print("C", sent_pack['corrected'].text)
    
    def uncountable_noun_check(self, chunk_pack, sent_pack):
        """ Corrects all mistakes connected with uncountable nouns in an NP (chunk)
        """
        chunk_listed = [token.text.lower() for token in chunk_pack['chunk']]

        if chunk_pack['chunk'].root.lemma_ in uncountables:
            if chunk_pack['chunk'].root.tag_ == "NNS":
                correction = (f"A {chunk_pack['chunk'].root.i} {chunk_pack['chunk'].root.i+1}"
                              + "|||UncNoun|||{chunk_pack['chunk'].root.lemma_}|||REQUIRED|||-NONE-|||0")
                sent_pack['corrected'] = self.token_replace(sent_pack['corrected'], chunk_pack['chunk'].root.lemma_, chunk_pack['chunk'].root.i)
                sent_pack['corrections'].append(correction)
                # chunk.root.i points to the index of the token in the whole Doc, not in the chunk (Span)
                noun_idx_in_chunk = chunk_pack['chunk'].root.i - chunk_pack['start']
                # Replacing the token in the chunk_pack['chunk'] and in its listed version
                chunk_listed[noun_idx_in_chunk] = chunk_pack['chunk'].root.lemma_
                new_chunk = Doc(self.nlp.vocab, words=chunk_listed)[:]  # Make it a Span
                chunk_pack['chunk'] = new_chunk

            if 'a' in chunk_listed or 'an' in chunk_listed:
                art_idx_in_chunk = (chunk_listed.index('a' if 'a' in chunk_listed
                                                            else 'the' if 'the' in chunk_listed else 'an'))

                article_idx = chunk_pack['start'] + art_idx_in_chunk
                # the indefinite article is allowed to be in a sentence in the 
                # construction 'a little' (or 'a few', which will be corrected)
                if sent_pack['corrected'][article_idx + 1].text not in ['little', 'few']:          

                    # FIXME: deleting a token leads to a change of indices.
                    # Therefore, after each change we have to reparse the sentence
                    correction = f"A {article_idx} {article_idx+1}|||IndArt:UncNoun|||-NONE-|||REQUIRED|||-NONE-|||0"
                    # sent_pack['corrected'] = self.token_replace(sent_pack['corrected'], None, article_idx)
                    sent_pack['corrections'].append(correction)
                    # del chunk_listed[art_idx_in_chunk]
                    # chunk = Doc(self.nlp.vocab, words=chunk_listed)

            # Checks if there are some words in the noun phrase that are not
            # used with uncountable nouns (e.g. many, few) and replaces them
            # with other words (e.g. much, little)
            for key in uncountable_change.keys():
                if key in chunk_listed:
                    error_idx_in_chunk = chunk_listed.index(key)
                    correct_token = uncountable_change[key]
                    error_idx = error_idx_in_chunk + chunk_pack['start']
                    sent_pack['corrected'] = self.token_replace(sent_pack['corrected'],
                                                                correct_token, error_idx)
                    correction = f"A {error_idx} {error_idx + 1}|||Prn:UncNoun|||{correct_token}|||REQUIRED|||-NONE-|||0"
                    chunk_listed[error_idx_in_chunk] = correct_token
                    new_chunk = self.nlp(' '.join(chunk_listed))[:]
                    chunk_pack['chunk'] = new_chunk
                    
            
        return chunk_pack, sent_pack
            
            
    def vanilla_bigram_check(self, text):
        """ Print all collocations that haven't been found in preprocessed bigram corpus
        """
        #m2_corrections = []
        text = text.lower()
        tokens = nltk.word_tokenize(text)
        for i in range(len(tokens) - 1):
            if tuple(tokens[i:i+2]) not in self.bigrams:
                print(f"Unlikely collocation: {tokens[i:i+2]}")

    def noun_phrase_check(self, sent_pack):
        """ All methods that have to deal with problems contained only in noun phrase itself.
            These are;
            - Indefinite article - root noun agreement
            - Choice of a/an (calls another function)
            - Both a personal pronoun and an article in a single NP
        """
        for chunk in sent_pack['corrected'].noun_chunks:

            chunk_pack = {"chunk": chunk, "start": chunk.start}

            chunk_pack, sent_pack = self.uncountable_noun_check(chunk_pack, sent_pack)

            chunk_pack, sent_pack = self.indef_art_choice_corr(chunk_pack, sent_pack)

            # Create a list of tags of corresponding tokens with articles left in their initial form
            tag_chunk_listed = [token.tag_ if (token.text != 'a' and token.text != 'an'
                            and token.text != 'the') else token.text for token in chunk_pack['chunk']] 
            
            sent_pack = self.ind_art_plural_noun_corr(chunk_pack, tag_chunk_listed, sent_pack)

            sent_pack = self.ind_art_pers_pronoun_corr(chunk_pack, tag_chunk_listed, sent_pack)
            

        return sent_pack

    def ind_art_pers_pronoun_corr(self, chunk_pack, chunk_listed, sent_pack):
        """Checks there is both an article and a personal pronoun in the NP"""
        if ('a' in chunk_listed or 'an' in chunk_listed or 'the' in chunk_listed) and "PRP$" in chunk_listed:   
            article_idx = chunk_pack['start'] + (chunk_listed.index('a' if 'a' in chunk_listed 
                                                            else 'the' if 'the' in chunk_listed else 'an'))
            correction = f"A {article_idx} {article_idx+1}|||Art:PRP$|||-NONE-|||REQUIRED|||-NONE-|||0"
            sent_pack['corrections'].append(correction)
        return sent_pack

    def ind_art_plural_noun_corr(self, chunk_pack, chunk_listed, sent_pack):
        """Checks there is an indefinite article & a plural root noun in the NP"""
        if ('a' in chunk_listed or 'an' in chunk_listed) and 'NNS' in chunk_listed:
            # Find the index of the article in the s-ce
            article_idx = chunk_pack['start'] + (chunk_listed.index('a' if 'a' in chunk_listed else 'an'))   
            correction = f"A {article_idx} {article_idx+1}|||Art:NNS|||-NONE-|||REQUIRED|||-NONE-|||0"
            sent_pack['corrections'].append(correction)
        return sent_pack

    def indef_art_choice_corr(self, chunk_pack, sent_pack):
        """ Returns a sent_pack with added corrections concerning the choice
            between a/an articles
        """
        chunk_listed = [token.text.lower() for token in chunk_pack['chunk']]

        if ('a' in chunk_listed or 'an' in chunk_listed):
            article = ('a' if 'a' in chunk_listed else 'an')
            art_idx_in_chunk = chunk_listed.index(article)
            # The word that comes after the article in the chunk
            next_word = chunk_pack['chunk'][art_idx_in_chunk+1].text
            # returns a list of possible pronunciations (lists of strings)              
            phonetic = self.arpabet.get(next_word, None)

            if phonetic:
                new_article = None
                if article == 'a' and phonetic[0][0] not in arpabet_consonants:    # i.e. 'a' + a vowel -> 'an'
                    new_article = 'an'
                if article == 'an' and phonetic[0][0] in arpabet_consonants:       # i.e. 'an' + a consonant -> 'a'
                    new_article = 'a'
                if new_article:
                    article_idx_sent = chunk_pack['start'] + art_idx_in_chunk
                    correction = (f"A {article_idx_sent} {article_idx_sent + 1}"
                                    + f"|||IndArt:Sound|||{new_article}|||REQUIRED|||-NONE-|||0")
                    sent_pack['corrections'].append(correction)

                    sent_pack['corrected'] = self.token_replace(sent_pack['corrected'], 
                                                                new_article, article_idx_sent)
                    chunk_listed[art_idx_in_chunk] = new_article
                    
                    chunk_pack['chunk'] = self.nlp(' '.join(chunk_listed))[:]
                    
        
        return chunk_pack, sent_pack

    def preposition_check(self, sent_pack):
        """ Return a list of m2 corrections.
            Detects and corrects errors connected with the choice of preposition.
            It is done with the help of two datasets:
            1) trigram counts with tags in them. These are in the following form:
            'word/entity + preposition + word.tag'
            2) trigrams without tags in them ('notag'). These are as follows:
            'word/entity + preposition + word/entity'
        """
        tag_matches = load_tag_ngram_file(os.path.join(self.data_folder, self.prep_tag_trigrams_file))

        notag_matches = load_notag_ngram_filename(os.path.join(self.data_folder, self.prep_notag_trigrams_file))

        matcher = DependencyMatcher(self.nlp.vocab)

        matcher.add("prep", [prep_pattern])
        
        sent = sent_pack['corrected']
        
        for match in matcher(sent):
            if sent[match[1][0]].ent_type_:
                head = sent[match[1][0]].ent_type_
            else:
                head = sent[match[1][0]].text.lower()

            preposition = sent[match[1][1]]
            dependent = sent[match[1][2]]

            tag_key = ' '.join([head, preposition.text.lower(), dependent.tag_])

            # Get the entity tag of a token if possible (to generalise)
            if dependent.ent_type_:    
                dependent_text = dependent.ent_type_
            else:
                dependent_text = dependent.text.lower()

            notag_key = ' '.join((head, preposition.text.lower(), dependent_text))
            
            count = tag_matches.get(tag_key, 0)

            # If the given preposition with its context is found neither in
            # tagged nor non-tagged data
            if count == 0 and notag_key not in notag_matches:
                tag_keys = tag_matches.keys()

                regex_notag_prep_finder = re.compile(f"{head} .* {dependent_text}")
                relevant_notag_keys = list(filter(regex_notag_prep_finder.match, notag_matches))
                if not relevant_notag_keys:
                    regex_tag_prep_finder = re.compile(f'{head} .* {dependent.tag_}')
                    relevant_tag_keys = list(filter(regex_tag_prep_finder.match, tag_keys))
                    relevant_tag_keys.sort(key=lambda x: tag_matches[x])
                    if not relevant_tag_keys:
                        continue
                    # FOR FUTURE: can suggest a few prepositions, actually (TODO)
                    most_relevant = relevant_tag_keys[-1]
                else:
                    # Taking a random first trigram. Possible to change by choosing 
                    # the one with the largest count (TODO)
                    most_relevant = relevant_notag_keys[0]                              

                correct_prep = most_relevant.split(' ')[1]
                correction = f'A {preposition.i} {preposition.i+1}|||Prep|||{correct_prep}|||REQUIRED|||-NONE-|||0'
                sent_pack['corrections'].append(correction)

                sent_pack['corrected'] = self.token_replace(sent_pack['corrected'], correct_prep, preposition.i)
        
        return sent_pack

    def token_replace(self, doc, token:str, offset:int):
        """ Returns a new Doc with token at index 'offset' changed to 'token'.
        """
        listed_sentence = [token.text for token in doc]
        if token:
            listed_sentence[offset] = token
        else:
            del listed_sentence[offset]
        corrected = self.nlp(' '.join(listed_sentence))

        return corrected
            
def create_bigrams(text):
    """ Return a dictionaty of bigram counts"""
    twit_tokens = nltk.word_tokenize(text)
    bigrams = ngrams(twit_tokens, 2)
    bigrams = dict(Counter(bigrams))
    return bigrams


if __name__ == "__main__":
    detector = Detector()

    text = ('Since each among us was a faggot... I think he fought by war \
and winked by him and die by vain. He was unoccupied by the situation. \
There were an kids, but I didn\'t like them. There were pounds from cocaine.\
I have many evidences for wrong advices of my neighbours. Their kindnesses lacks morality. I have a advices to you')
    
    # text = "There were an kids, but I didn't like them."
    detector(text)