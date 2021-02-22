from classes import Detector
import spacy

with open("/home/cyril/Desktop/Python/projects/error_detection/data/C_sentences.txt") as f:
    text = f.read()

for sentence in text.split("\n\n"):
    sentence = sentence.split('\n')
    original = sentence[0][2:]
    corrections = sentence[1:]
    with open("/home/cyril/Desktop/Python/projects/error_detection/data/C_sentences.txt", 'a') as f:
        f.write(original + '\n\n')

detector = Detector()
detector(text, save_file="/home/cyril/Desktop/Python/projects/error_detection/data/C.train.gold_guess.m2")