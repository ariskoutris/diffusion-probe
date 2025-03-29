"""
Utility functions for processing WordNet synsets and extracting their attributes.
"""
from nltk.corpus import wordnet as wn

def get_synset_name(synset):
    return synset.lemmas()[0].name().replace("_", " ").title()

def get_synset_frequency(synset):
    return synset.lemmas()[0].count()

def get_synset_definition(synset):
    return synset.definition()

def get_synset_attributes(synset):
    return {
        'name': get_synset_name(synset),
        'frequency': get_synset_frequency(synset),
        'definition': get_synset_definition(synset)
    }

def find_root_synset(word, pos=None, index=0):
    synsets = wn.synsets(word, pos=pos)
    if not synsets:
        raise ValueError(f"No synsets found for '{word}'")
    return synsets[index]

def calculate_similarity(synset1, synset2):
    return synset1.wup_similarity(synset2)
