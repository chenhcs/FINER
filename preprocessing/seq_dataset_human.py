from keras.preprocessing import sequence
import numpy as np
from sys import argv

if len(argv) > 2:
    folder = argv[1]
    dataset = '/' + argv[2]
else:
    folder = argv[1]
    dataset = ''

training_gene_ids = []
training_iso_ids = []
test_gene_ids = []
test_iso_ids = []
fr = open('../' + folder + '/input' + dataset + '/train_gene_list.txt')
while True:
    line = fr.readline()
    if not line:
        break
    training_gene_ids.append(line.split('\n')[0])
fr.close()
fr = open('../' + folder + '/input' + dataset + '/train_isoform_list.txt')
while True:
    line = fr.readline()
    if not line:
        break
    training_iso_ids.append(line.split('\n')[0])
fr.close()
fr = open('../' + folder + '/input' + dataset + '/test_gene_list.txt')
while True:
    line = fr.readline()
    if not line:
        break
    test_gene_ids.append(line.split('\n')[0])
fr.close()
fr = open('../' + folder + '/input' + dataset + '/test_isoform_list.txt')
while True:
    line = fr.readline()
    if not line:
        break
    test_iso_ids.append(line.split('\n')[0])
fr.close()

training_gene_ids = np.array(training_gene_ids)
training_iso_ids = np.array(training_iso_ids)
test_gene_ids = np.array(test_gene_ids)
test_iso_ids = np.array(test_iso_ids)
np.save('../' + folder + '/input' + dataset + '/train_gene_list.npy', np.array(training_gene_ids))
np.save('../' + folder + '/input' + dataset + '/train_isoform_list.npy', np.array(training_iso_ids))
np.save('../' + folder + '/input' + dataset + '/test_gene_list.npy', np.array(test_gene_ids))
np.save('../' + folder + '/input' + dataset + '/test_isoform_list.npy', np.array(test_iso_ids))

aa_num_dict = {}

aa_num_dict['F'] = 1
aa_num_dict['L'] = 2
aa_num_dict['I'] = 3
aa_num_dict['M'] = 4
aa_num_dict['V'] = 5
aa_num_dict['S'] = 6
aa_num_dict['P'] = 7
aa_num_dict['T'] = 8
aa_num_dict['A'] = 9
aa_num_dict['Y'] = 10
aa_num_dict['H'] = 11
aa_num_dict['Q'] = 12
aa_num_dict['N'] = 13
aa_num_dict['K'] = 14
aa_num_dict['D'] = 15
aa_num_dict['E'] = 16
aa_num_dict['C'] = 17
aa_num_dict['W'] = 18
aa_num_dict['R'] = 19
aa_num_dict['G'] = 20

print('Convert isoform sequences to numpy arrays...')
X_train = []
X_test = []
gene_index = []
test_gene_id = []
train_gene_id = []
train_iso_id = []
test_iso_id = []
gene_count = 0
id_numseq_map = {}
longest = 0

fr = open('../' + folder + '/sequences/isoform_aa_sequences.txt', 'r')
while True:
    line = fr.readline()
    if not line:
        break
    if '>' in line:
        gene_name = line.split()[0].split('>')[1]
        prot_id = line.split()[1]
        prot_seq = fr.readline().split()[0]
        numseq = []
        for j in range(len(prot_seq) - 2):
            ngram = (aa_num_dict[prot_seq[j]] - 1) * 400 + (aa_num_dict[prot_seq[j + 1]] - 1) * 20 + (aa_num_dict[prot_seq[j + 2]])
            numseq.append(ngram)
        id_numseq_map[prot_id] = numseq
        if len(numseq) > longest:
            longest = len(numseq)

for id in training_iso_ids:
    X_train.append(id_numseq_map[id])

for id in test_iso_ids:
    X_test.append(id_numseq_map[id])

print('Done')
X_train_np = np.array(sequence.pad_sequences(X_train, longest))
X_test_np = np.array(sequence.pad_sequences(X_test, longest))
print(X_train_np.shape)
print(X_test_np.shape)
np.savez_compressed('../' + folder + '/input' + dataset + '/human_sequence_train.npz', data=X_train_np)
np.savez_compressed('../' + folder + '/input' + dataset + '/human_sequence_test.npz', data=X_test_np)
