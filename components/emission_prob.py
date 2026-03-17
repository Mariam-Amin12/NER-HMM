import numpy as np

def find_emission_probabilities(num_states, ner_tags, observed_sequences):
    observed_sequences = list(observed_sequences)
    num_observations = len(set(obs for obs_seq in observed_sequences for obs in obs_seq))
    emission_probabilities = np.zeros((num_states, num_observations))
    observation_to_index = {obs: idx for idx, obs in enumerate(set(obs for obs_seq in observed_sequences for obs in obs_seq))}
    
    for i, sequence in enumerate(ner_tags):
        observed_sequence = observed_sequences[i]
        for state, token in zip(sequence, observed_sequence):
            emission_probabilities[state, observation_to_index[token]] += 1

    row_sums = emission_probabilities.sum(axis=1)
    emission_probabilities = emission_probabilities / row_sums[:, np.newaxis]

    return emission_probabilities

