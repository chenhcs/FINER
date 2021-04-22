from keras.preprocessing import sequence
import numpy as np
from sys import argv

if len(argv) > 2:
    folder = argv[1]
    dataset = '/' + argv[2]
else:
    folder = argv[1]
    dataset = ''

print('Convert isoform domains to numpy arrays...')
fr = open('../' + folder + '/domains/domain_id_mapping.txt')
entry = fr.readline().split('\n')[0]
domain_dic = {}
while entry != '':
    value, key = entry.split('\t')
    domain_dic[key] = int(value)
    entry = fr.readline().split('\n')[0]
fr.close()

domain_dic['0'] = 0

fr = open('../' + folder + '/domains/human_isoform_domains.txt')
entry = fr.readline().split('\n')[0]
isofDmDic = {}
while entry != '':
    isof = entry.split('\t')[1]
    if len(entry.split('\t')) > 2:
        domains = entry.split('\t')[2]
        tmpD = [domain_dic[key] for key in domains.split(' ')]
        #print(tmpD)
        isofDmDic[isof] = tmpD
    entry = fr.readline().split('\n')[0]
fr.close()

HUMANX_train_domain = []
HUMANX_test_domain = []
HUMANtrain_iso_id_ho_s = np.load('../' + folder + '/input' + dataset + '/train_isoform_list.npy')
HUMANtest_iso_id_ho_s = np.load('../' + folder + '/input' + dataset + '/test_isoform_list.npy')
max_length = 0
for isoid in HUMANtrain_iso_id_ho_s:
    if isoid in isofDmDic.keys():
        HUMANX_train_domain.append(isofDmDic[isoid])
        if len(isofDmDic[isoid]) > max_length:
            max_length = len(isofDmDic[isoid])
    else:
        #print(isoid)
        HUMANX_train_domain.append([0])

for isoid in HUMANtest_iso_id_ho_s:
    if isoid in isofDmDic.keys():
        HUMANX_test_domain.append(isofDmDic[isoid])
        if len(isofDmDic[isoid]) > max_length:
            max_length = len(isofDmDic[isoid])
    else:
        #print(isoid)
        HUMANX_test_domain.append([0])

print(max_length)
HUMANX_train_np = np.array(sequence.pad_sequences(HUMANX_train_domain, max_length))
HUMANX_test_np = np.array(sequence.pad_sequences(HUMANX_test_domain, max_length))

print('Done')
print(HUMANX_train_np)
print(HUMANX_test_np)
print(HUMANX_train_np.shape)
print(HUMANX_test_np.shape)
np.save('../' + folder + '/input' + dataset + '/human_domain_train.npy', HUMANX_train_np)
np.save('../' + folder + '/input' + dataset + '/human_domain_test.npy', HUMANX_test_np)
