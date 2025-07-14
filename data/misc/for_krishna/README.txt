7/3/25:
=======
Substitution model: general time reversible
Indel models tested: TKF91, TKF92


Folder contents:
================
- GTR_equilibriums.tsv: equilibrium distribution (observed amino acid frequencies from training set)
- GTR_exchangeabilities.tsv: exchangeability parameters (from GTR+TKF91 training run)
- tkf91_indel_params.txt: TKF91 indel rates
- tkf92_indel_params.txt: TKF92 indel rates, TKF92 fragment extension probability
- loglikelihoods.xlsx: log-likelihoods on unseen test set, for both models


Dataset:
=============
- Original multiple sequence alignments from PFam v36
- Following the protocol from Prillo et. al. 2023 (CherryML): for each multiple sequence alignments, exclusive pairs of sequences were greedily chosen based on branch lengths provided by Pfam. Authors proved that this composite likelihood is a good approximate for the full likelihood.
- Branch lengths provided by PFam, originally from FastTree. One unique branch length per pairwise alignment.
- training and test sets split the pairwise alignments by pfam and clan. That is, all members of a clan can appear either in training or in test, but not both. Same with pfam.
- Data augmentation: since we assume models are reversible, I double my dataset by training on both "forward" and "reverse" pairs

	for two sequences: seq1, seq2
	forward pair:   ancestor = seq1, descendant = seq2
	reverse pair: descendant = seq1,   ancestor = seq2

training set: 424,417 unique pairs; 848,834 alignments
test set: 125,866 unique pairs; 251,732 alignments


Training:
===========
- Maximize log-likelihood via Adam optimizer
- Batch size: 25,000
- Learning rate: constant, 0.005
