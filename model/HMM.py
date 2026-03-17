from components.viterbi import viterbi

from components.start_prob import find_start_probabilities
from components.transition_prob import find_transition_probabilities
from components.emission_prob import find_emission_probabilities

class HMM:
    def __init__(self, states):
        self.states = states
        self.start_probabilities = None
        self.transition_probabilities = None
        self.emission_probabilities = None
        self.train_observed_sequences = None

    def compute_start_probabilities(self, ner_tags):
        self.start_probabilities = find_start_probabilities(len(self.states), ner_tags)


    def compute_transition_probabilities(self, ner_tags):
        self.transition_probabilities = find_transition_probabilities(len(self.states), ner_tags)


    def compute_emission_probabilities(self, ner_tags, observed_sequences):
        self.emission_probabilities = find_emission_probabilities(len(self.states), ner_tags, observed_sequences)


    def viterbi_algorithm(self, obs):
        return viterbi(obs, self.states, self.start_probabilities, self.transition_probabilities, self.emission_probabilities)
    
    def train_hmm(self, taged_sequences, observed_sequences):
        self.train_observed_sequences = [list(seq) for seq in observed_sequences]

        # Remap potentially sparse tag ids to a compact range for matrix indexing.
        tag_sequences = [list(seq) for seq in taged_sequences]
        unique_tags = sorted(set(tag for seq in tag_sequences for tag in seq))
        tag_to_compact = {tag: idx for idx, tag in enumerate(unique_tags)}
        compact_tag_sequences = [[tag_to_compact[tag] for tag in seq] for seq in tag_sequences]

        # Keep external state values for decoded predictions.
        self.states = unique_tags

        self.compute_start_probabilities(compact_tag_sequences)
        self.compute_transition_probabilities(compact_tag_sequences)
        self.compute_emission_probabilities(compact_tag_sequences, self.train_observed_sequences)
        