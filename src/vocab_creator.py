#!/usr/bin/env python3

# imports go here
from pycocotools.coco import COCO
from collections import Counter
import re

import os
import numpy as np


class VocabCreate(object):
    """Class to create the vocabulary for our project"""

    def __init__(self, data_pth, output_path):
        """Constructor
        """
        if os.path.exists(output_path):
            loaded_data = np.load(output_path, allow_pickle=True)
            self.one_hot_inds = loaded_data["one_hot_indices"].item()
            self.dict = loaded_data["dictionary"]
            return

        # the path to the json file containing the annotations
        self.data_pth = data_pth

        # data holds the dictionary corresponding to the JSON file
        self.data = None

        self.max_cap_len = 0

        # dict contains the final word list with words of frequency >= threshold
        self.dict = list()

        # One hot indices required for embedded layer of RNN
        self.one_hot_inds = dict()

        # words and its frequency in the original dataset
        self.vocab_list = Counter()
        self.load_data()
        self.frequency_list()
        self.remove_lowest(thresh=5)
        # print(len(self.vocab_list.keys()))
        # vals = [self.vocab_list[v] for v in self.dict]
        self.create_inds()

        # Creates a list of tuples
        # self.one_hot_inds = np.array(list(self.one_hot_inds.items()))
        np.savez(output_path, one_hot_indices=self.one_hot_inds, dictionary=self.dict)
        # print(set(vals), len(vals))

    def load_data(self):
        """Function to load the data"""
        self.data = COCO(self.data_pth)

    def frequency_list(self):
        """Creates a list of words"""
        # iterate through every caption
        cap_ids = self.data.anns.keys()
        for id in cap_ids:
            cap = str(self.data.anns[id]["caption"])
            clean_cap = re.sub(r'[^a-zA-Z0-9 ]+', '', cap)
            word_list = clean_cap.lower().strip().split()
            self.max_cap_len = max(self.max_cap_len, len(word_list))
            self.vocab_list.update(word_list)

    def __len__(self):
        return len(self.one_hot_inds)

    def remove_lowest(self, thresh):
        """
        Only keeps those words whose frequency is greater than threshold
        :argument thresh minimum frequency to keep in the vocabulary
        """
        ret = [w for w in self.vocab_list.keys() if self.vocab_list[w] >= thresh]
        self.dict = ret

    def create_inds(self):
        """
        Creates a dictionary with the index returned for each word
        :return:
        """
        # add start and end vectors to the vocabulary
        start_vec = "start_vec"
        end_vec = "end_vec"
        unknown = "unk_vec"
        pad_vec = "pad_vec"
        self.dict.insert(0, pad_vec)
        self.dict.append(start_vec)
        self.dict.append(end_vec)
        self.dict.append(unknown)
        
        # Generate distinct indices for each word in the vocabulary
        for i in range(len(self.dict)):
            self.one_hot_inds[self.dict[i]] = i

    def maximum_caption_length(self):
        return self.max_cap_len


def main():
    vocab = VocabCreate("../datasets/COCO/annotations/captions_train2014.json", "../outputs/vocab.npz")
    print("Loaded successfully!")
    print("There are ", len(vocab), " unique words with frequency >=5 in the dataset")
    # print(vocab.maximum_caption_length())


if __name__ == "__main__":
    main()
