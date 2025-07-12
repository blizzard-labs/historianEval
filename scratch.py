import pandas as pd

df = pd.DataFrame({'col1': [1, 2, 3], 'col2': ['A', 'B', 'C']}, index=['row_a', 'row_b', 'row_c'])
row_indices = df.index

af = pd.read_csv('data/model_gen/mamX10k/simulated_phylo_parameters.csv')
af.insert(loc=0, column = 'sequence_name', value= ['seq_' + str(k) for k in (af.index + 1)])
ri = af.index


print(ri)
print(af)

print(row_indices)
print(df)