import numpy as np
import re


REMAIN_PUNCT = [',', '.', '!', '?']  # punctuation that we leave
END_PUNC = ['.', '!', '?']
PUNCT_TO_TOKEN = {
    ' ' : 'S',  # space
    ',' : 'C',  # comma
    '.' : 'P',  # period
    '!' : 'EX', # exclamation
    '?' : 'Q'   # question
}


def process_raw_sentence(sentence):

    dashes = ['—', '–']
    
    subs_by_comma = [
        '? ', '! ', ': ',
        ', — ', ' — ',
        ', – ', ' – ',  # another dash
        '; '
    ]  # substitute by ', '
    subs_by_space = ['\xa0']
    subs_by_period = ['…']
    
    to_delete_arr= ['"', "'", '«', '»', '„', '“', '*']  # to delete
    cases_arr = ['(', ')', '[', ']']  # delete info in brackets

    sentence = sentence.lower()  # lowercase the sentence

    # deleting of wahat needs to be deleted
    for mark_to_del in to_delete_arr:
        sentence = sentence.replace(mark_to_del, '')

    # substitute bugs like '\xa0' by ' '
    for mark_to_subs in subs_by_space:
        sentence = sentence.replace(mark_to_subs, ' ')

    # substitute specific symbols by '.'
    for mark_to_subs in subs_by_period:
        sentence = sentence.replace(mark_to_subs, '.')
        
    # deleting of a dash in the beginning of the sentence
    if sentence[0] in dashes:
        for ind, char in enumerate(sentence):
            if bool(re.search('[а-яА-Я]', char)):  # first letter
                sentence = sentence[ind:]
                break
    
    # deleting of text in cases: '[*]' and the '(*)'
    sentence_no_cases = re.sub('[\(\[].*?[\)\]]', '', sentence)
    if len(sentence_no_cases) < 2:  # check if we have not deleted complete sentence
        for case in cases_arr:
            sentence = sentence.replace(case, '')
    else:
        sentence = sentence_no_cases

    # punctuation in the end of the sentence can be multiple
    # for example: '...', '!..', '?..'
    # convert to:  '.', '!' and '?'
    n_of_last_punct = 0
    for char in sentence[::-1]:
        if bool(re.search('[а-яА-Я]', char)):
            break
        else:
            n_of_last_punct += 1
    if n_of_last_punct > 0:
        for mark in sentence[-n_of_last_punct:]:
            if not mark == ' ':  
                last_punct = mark  # only one punctuation mark in the end
                break
        if last_punct == ';':
            last_punct = '.'  # last mark of the sentence cant be a ';'
        sentence = sentence[:-n_of_last_punct]  # delete final punctuation (add it in the end)
    else:
        last_punct = '.'  # if there is no punctoation in the end of the sentence
    
    # substitute of all punctuation within the sentence by ',' (comma)
    for mark in subs_by_comma:
        sentence = sentence.replace(mark, ', ')

    # FINAL CLEANING
    # delete all end tokens within the sentence
    for end_punc in END_PUNC:
        sentence = sentence.replace('.', '')

    sentence = re.sub(r'\s+,', ',', sentence)
    
    # check if the first symbol is a space (accidentally) - delete it
    if sentence[0] == ' ':
        sentence = sentence[1:]

    # replace all multiple spaces by a single space ' '
    sentence = re.sub(r'\s+', ' ', sentence)
    
    # save edited sentence (adding the punctuation in the end of sentence)
    return sentence + last_punct


def tokenize(sentence, tokens_dict=PUNCT_TO_TOKEN):
    tokens_sent = []
    for word in sentence.split(' '):
        
        if word[-1] not in tokens_dict.keys():
            tokens_sent.append(tokens_dict[' '])
        else:
            tokens_sent.append(tokens_dict[word[-1]])

    return ' '.join(tokens_sent)

