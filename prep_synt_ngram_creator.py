"""Texts processed:
   - Fight Club
"""

import nltk
import spacy
from spacy.matcher import DependencyMatcher
from spacy import displacy
import csv
import os
import datetime
import time

CSV_TAG_FILENAME = "/home/cyril/Desktop/Python/projects/error_detection/data/trigrams_synt.csv"
CSV_NOTAG_FILENAME = "/home/cyril/Desktop/Python/projects/error_detection/data/trigrams_notag.csv"
TRAIN_TEXT_FILENAME = "/home/cyril/Desktop/Python/projects/error_detection/data/fight_club.txt"     #CHANGE FILENAME

prep_pattern = [
        {"RIGHT_ID": "to_head", "RIGHT_ATTRS": {"IS_ALPHA": True}}, 
        {"RIGHT_ID": "prep", "RIGHT_ATTRS": {"TAG":"IN", "DEP": 'prep'}, "LEFT_ID": "to_head", "REL_OP": ">"},   # Matches any word > preposition > any word in a dependency relation
        {"LEFT_ID": "prep", "REL_OP": ">", "RIGHT_ID": "dependent", "RIGHT_ATTRS": {"DEP": "pobj", "IS_ALPHA": True}}
        ]

def main(filename_to_train:str, tag_trigrams_filename:str, notag_ngrams_filename:str):

    with open(filename_to_train, encoding='utf8') as f:
        text = f.read()

    tag_matches = load_tag_ngram_file(tag_trigrams_filename)

    notag_matches = load_notag_ngram_filename(notag_ngrams_filename)

    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)

    tag_matches, notag_matches = match_prep_pattern(doc, nlp, tag_matches, notag_matches)

       
    save_tag_ngrams(tag_matches, tag_trigrams_filename)
    save_notag_ngrams(notag_matches, notag_ngrams_filename)


def match_prep_pattern(doc, nlp, tag_matches, notag_matches):
    """ Returns a a dictionary of ('word preposition wordtag'): count.
        The dictionary is created from dependency parse tree syntactic relations.
    """

    matcher = DependencyMatcher(nlp.vocab)

    matcher.add("prep", [prep_pattern])
    count = 0
    for match in matcher(doc):               # Iterate through all of the matches and add 1 to its count in the dictionary
        if count % 1000 == 0:
            print(f"Processing match # {count}")
        indices = match[1]

        # Writing the named entity of the head token insted of the whole word in case a named entity exists
        if doc[indices[0]].ent_type_:                
             head_word = doc[indices[0]].ent_type_
        else:
            head_word = doc[indices[0]].text.lower()

        tag_trigram = ' '.join((head_word, doc[indices[1]].text.lower(), doc[indices[2]].tag_))
        if doc[indices[2]].ent_type_:
            dependent_notag = doc[indices[2]].ent_type_
        else:
            dependent_notag = doc[indices[2]].text.lower()
        notag_trigram = ' '.join((head_word, doc[indices[1]].text.lower(), dependent_notag))
        notag_matches.add(notag_trigram)
        tag_matches[tag_trigram] = tag_matches.get(tag_trigram, 0) + 1
        count += 1

    return tag_matches, notag_matches


def save_tag_ngrams(ngrams:dict, filename:str):
    """ Save a dictionary of 3-grams as a csv file. All entries should be in the format:
        'word word tag': count.
        NOTE: IT REWRITES ALL THE DATA IN THE FILE
    """
    with open(filename, 'w') as f:
        writer = csv.writer(f)
        for key, value in ngrams.items():
            row = [*key.split(' '), str(value)]
            writer.writerow(row)

def save_notag_ngrams(ngrams:set, filename):
    """ Save the given in a set to filename as a csv file"""
    with open(filename, 'w') as f:
        writer = csv.writer(f)
        for entry in ngrams:
            row = entry.split(" ")
            writer.writerow(row)
     

def load_tag_ngram_file(ngrams_filename):
    """ Returns a dictionary of ('word preposition wordtag'): count taken from a csv file.
    """
    matches = {}
    with open(ngrams_filename, 'r') as f: 
        reader = csv.reader(f)
        for line in reader:
            key = ' '.join(line[:3])
            matches[key] = int(line[3])
    
    return matches

def load_notag_ngram_filename(ngrams_filename):
    """Load ngrams from a csv into a set"""
    matches = set()
    with open(ngrams_filename, 'r') as f:
        reader = csv.reader(f)
        for line in reader:
            entry = ' '.join(line)
            matches.add(entry)

    return matches



if __name__ == "__main__":
    data_path = '/home/cyril/Desktop/Python/projects'
    logs_path = '/home/cyril/Desktop/Python/projects/error_detection/data/trigrams_logs'
    os.chdir(r'/home/cyril/Desktop/Python/projects/words_check')
    textfiles = [textfile for textfile in os.listdir() if textfile[-4:] == '.txt']
    # textfiles = ['fiesta.txt', 'paradise_lost.txt', 'the_two_towers.txt']
    processed = ['fiesta.txt', 'paradise_lost.txt', 'the_two_towers.txt']

    tag_matches = load_tag_ngram_file(CSV_TAG_FILENAME)
    notag_matches = load_notag_ngram_filename(CSV_NOTAG_FILENAME)
    
    nlp = spacy.load("en_core_web_sm")
    starttime = time.time()
    for textfile in textfiles:
        if textfile in processed:
            continue
        with open(logs_path, 'a') as logs:
            timestring = time.strftime('%H:%M:%S')
            logs.write(f"Started processing {textfile}. Time: {timestring}\n")
        with open(textfile, 'r', encoding='utf8') as f:
            text = f.read()
        text_len = len(text)
        text_chunks = list(range(0, text_len, 800000))
        text_chunks.append(text_len)
        for i in range(len(text_chunks) - 1):
            doc = nlp(text[text_chunks[i]:text_chunks[i+1]])
            tag_matches, notag_matches = match_prep_pattern(doc, nlp, tag_matches, notag_matches)
            save_tag_ngrams(tag_matches, CSV_TAG_FILENAME)
            save_notag_ngrams(notag_matches, CSV_NOTAG_FILENAME)

            
            
    with open(logs_path, 'a') as logs:
        timestring = time.strftime('%H:%M:%S')
        logs.write(f"Finished processing all files. Time: {timestring}.\n"
           +  f"Total time spent in seconds: {time.time() - starttime}")
        

