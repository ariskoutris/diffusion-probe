"""
Utility functions for processing WordNet synsets and extracting their attributes.
"""
from nltk.corpus import wordnet as wn

def get_synset_name(synset):
    """Extract a human-readable name from a synset."""
    return synset.lemmas()[0].name().replace("_", " ").title()

def get_synset_frequency(synset):
    """Get the frequency count of a synset's primary lemma."""
    return synset.lemmas()[0].count()

def get_synset_definition(synset):
    """Get the definition of a synset."""
    return synset.definition()

def get_synset_attributes(synset):
    """Get a dictionary of synset attributes."""
    return {
        'name': get_synset_name(synset),
        'frequency': get_synset_frequency(synset),
        'definition': get_synset_definition(synset)
    }

def find_root_synset(word, pos=None, index=0):
    """Find a synset by word and optional part of speech."""
    synsets = wn.synsets(word, pos=pos)
    if not synsets:
        raise ValueError(f"No synsets found for '{word}'")
    return synsets[index]

def calculate_similarity(synset1, synset2):
    """Calculate the semantic similarity between two synsets."""
    return synset1.wup_similarity(synset2)
