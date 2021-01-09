from keras.preprocessing import sequence
import numpy as np
from sys import argv

script, dataset = argv

training_gene_ids = []
training_iso_ids = []
test_gene_ids = []
test_iso_ids = []
fr = open('../data/input/' + dataset + '/train_gene_list.txt')
while True:
    line = fr.readline()
    if not line:
        break
    training_gene_ids.append(line.split('\n')[0])
fr.close()
fr = open('../data/input/' + dataset + '/train_isoform_list.txt')
while True:
    line = fr.readline()
    if not line:
        break
    training_iso_ids.append(line.split('\n')[0])
fr.close()
fr = open('../data/input/' + dataset + '/test_gene_list.txt')
while True:
    line = fr.readline()
    if not line:
        break
    test_gene_ids.append(line.split('\n')[0])
fr.close()
fr = open('../data/input/' + dataset + '/test_isoform_list.txt')
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
np.save('../data/input/' + dataset + '/train_gene_list.npy', np.array(training_gene_ids))
np.save('../data/input/' + dataset + '/train_isoform_list.npy', np.array(training_iso_ids))
np.save('../data/input/' + dataset + '/test_gene_list.npy', np.array(test_gene_ids))
np.save('../data/input/' + dataset + '/test_isoform_list.npy', np.array(test_iso_ids))

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

print('loading Sequence...')
X_train = []
X_test = []
gene_index = []
test_gene_id = []
train_gene_id = []
train_iso_id = []
test_iso_id = []
gene_count = 0
fr = open('../data/sequences/isoform_aa_sequences.txt', 'r')

entry = fr.readline()
id_numseq_map = {}
longest = 0
while entry != '':
    entry = entry.split('\n')[0]
    if '>' in entry:
        gene_name = entry.split('\t')[0].split('>')[1]
        isonumber = int(entry.split('\t')[1].split(' ')[0])
        id_seq_dic = {}
        id_len_dic = {}
        valid = 1
        for i in range(isonumber):
            item = fr.readline().split('\n')[0]
            prot_id, prot_len = item.split('\t')
            prot_len = int(prot_len)
            prot_seq = fr.readline().split('\n')[0]
            id_seq_dic[prot_id] = prot_seq
            id_len_dic[prot_id] = prot_len
        if valid:
            for key in id_seq_dic.keys():
                prot_len = id_len_dic[key]
                seq = id_seq_dic[key]
                numseq = []
                for j in range(len(seq) - 2):
                    ngram = (aa_num_dict[seq[j]] - 1) * 400 + (aa_num_dict[seq[j + 1]] - 1) * 20 + (aa_num_dict[seq[j + 2]])
                    numseq.append(ngram)
                id_numseq_map[key] = numseq
                if len(numseq) > longest:
                    longest = len(numseq)
    entry = fr.readline()

for id in training_iso_ids:
    X_train.append(id_numseq_map[id])

for id in test_iso_ids:
    X_test.append(id_numseq_map[id])

print('loading complete...')
X_train_np = np.array(sequence.pad_sequences(X_train, longest))
X_test_np = np.array(sequence.pad_sequences(X_test, longest))
print(X_train_np.shape)
print(X_test_np.shape)
np.savez_compressed('../data/input/' + dataset + '/human_sequence_train.npz', data=X_train_np)
np.savez_compressed('../data/input/' + dataset + '/human_sequence_test.npz', data=X_test_np)
