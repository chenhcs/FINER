from keras.preprocessing import sequence
import numpy as np
from sys import argv

script, dataset = argv

fr = open('../data/domains/domain_id_mapping.txt')
entry = fr.readline().split('\n')[0]
domain_dic = {}
while entry != '':
    value, key = entry.split('\t')
    domain_dic[key] = int(value)
    entry = fr.readline().split('\n')[0]
fr.close()

domain_dic['0'] = 0

fr = open('../data/domains/human_isoform_domains.txt')
entry = fr.readline().split('\n')[0]
isofDmDic = {}
while entry != '':
    isof, domains = entry.split('\t')[1:]
    domains += '0'
    tmpD = [domain_dic[key] for key in domains.split(' ')]
    if len(tmpD) > 1:
        tmpD.pop()
    print(tmpD)
    isofDmDic[isof] = tmpD
    entry = fr.readline().split('\n')[0]
fr.close()

HUMANX_train_domain = []
HUMANX_test_domain = []
HUMANtrain_iso_id_ho_s = np.load('../data/input/' + dataset + '/train_isoform_list.npy')
HUMANtest_iso_id_ho_s = np.load('../data/input/' + dataset + '/test_isoform_list.npy')
max_length = 0
for isoid in HUMANtrain_iso_id_ho_s:
    if isoid in isofDmDic.keys():
        HUMANX_train_domain.append(isofDmDic[isoid])
        if len(isofDmDic[isoid]) > max_length:
            max_length = len(isofDmDic[isoid])
    else:
        print(isoid)
        HUMANX_train_domain.append([0])

for isoid in HUMANtest_iso_id_ho_s:
    if isoid in isofDmDic.keys():
        HUMANX_test_domain.append(isofDmDic[isoid])
        if len(isofDmDic[isoid]) > max_length:
            max_length = len(isofDmDic[isoid])
    else:
        print(isoid)
        HUMANX_test_domain.append([0])

print(max_length)
HUMANX_train_np = np.array(sequence.pad_sequences(HUMANX_train_domain, max_length))
HUMANX_test_np = np.array(sequence.pad_sequences(HUMANX_test_domain, max_length))

print('loading complete...')
print(HUMANX_train_np)
print(HUMANX_test_np)
print(HUMANX_train_np.shape)
print(HUMANX_test_np.shape)
np.save('../data/input/' + dataset + '/human_domain_train.npy', HUMANX_train_np)
np.save('../data/input/' + dataset + '/human_domain_test.npy', HUMANX_test_np)
