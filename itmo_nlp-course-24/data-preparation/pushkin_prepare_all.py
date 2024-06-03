import numpy as np
import pandas as pd

from razdel import sentenize

import random
import re
import os

from processing import REMAIN_PUNCT, END_PUNC, PUNCT_TO_TOKEN
from processing import process_raw_sentence, tokenize



if __name__ == '__main__':

    log = True  # print some info in process
    
    pushkin_raw_dir = '../data/raw/pushkin'
    
    pushkin_prepared_dir = '../data/prepared/pushkin'

    books = []  # all books in dir
    for file in os.listdir(pushkin_raw_dir):  # through files in selected directory            
        filename = os.fsdecode(file)
        if filename.endswith(".txt"):  # looking for .txt files in the directory
            books.append(filename)

    for book_filename in books:
        book_filename_csv = book_filename.split('.')[0] + '.csv'
        
        if log :
            print(book_filename)

        with open(os.path.join(pushkin_raw_dir, book_filename), 'r') as file:
            book_raw = file.read()

        #---------------------------------------------------chapters
        chapter_length_th = 1000  # chapter length threshold
        min_sent_len = 5  # in chars
        
        chapters_list = []  # list with chapters text (with replaced '\n' by ' ')

        for text_part in re.split('\n\n', book_raw):  # includes paragraphs with chapter's numbers and epigraphs
            if len(text_part) > chapter_length_th:
                text_part = text_part.replace('\n', ' ')
                text_part = re.sub('[\[].*?[\]]', '', text_part)
                chapters_list.append(text_part)  # add chapter text without '\n'

        if log:
            print(f'\t{len(chapters_list)} chapters')
        #---------------------------------------------------sentences
        all_sentences_raw = []
        all_sentences_processed = []
        all_sentences_lengths = []

        for chapter_number in range(len(chapters_list)):
            for substring in sentenize(chapters_list[chapter_number]):
                sentence = substring.text
                
                if len(sentence) > min_sent_len and not bool(re.search('[a-zA-Z]', sentence)):
                    # delete too short sentences and sentences with non-russian letters
                    all_sentences_raw.append(sentence)
                    
                    processed_sentence = process_raw_sentence(sentence)  # process sentence
                    len_processed = len(processed_sentence.split(' '))  # find length of the sentence
                    all_sentences_processed.append(processed_sentence)
                    all_sentences_lengths.append(len_processed)
        if log:
            print(f'\t{len(all_sentences_raw)} sentences')

        #---------------------------------------------------dataframe
        book_sent_df = pd.DataFrame()

        book_sent_df['raw'] = all_sentences_raw
        book_sent_df['processed'] = all_sentences_processed
        book_sent_df['len'] = all_sentences_lengths
        #---------------------------------------------------targets
        if log:
            print('\t\tTargets: ', end='')

        all_inputs = []
        all_targets = []
        
        for ind_row in range(book_sent_df.shape[0]):
            sentence_this = book_sent_df.iloc[ind_row]['processed']
        
            input_this = sentence_this
            for mark in REMAIN_PUNCT:
                input_this = input_this.replace(mark, '')
        
            target_this = tokenize(sentence_this)
        
            assert len(target_this.split(' ')) == len(input_this.split(' '))
            
            all_inputs.append(input_this)
            all_targets.append(target_this)

        if log:
            print('done!')
        #---------------------------------------------------saving dataframe
        if log:
            print(f'\t\tSaving: ', end='')

        # add new rows to DF
        book_sent_df['input'] = all_inputs
        book_sent_df['target'] = all_targets
        # saving dataframe (only 'input' and 'target' columns)
        book_sent_df[['input', 'target']].to_csv(os.path.join(pushkin_prepared_dir, book_filename_csv))

        if log:
            print(f'done ({os.path.join(pushkin_prepared_dir, book_filename_csv)})')

