import numpy as np
from numpy import genfromtxt
from scipy import sparse
from sys import argv
import os
import sys

if len(argv) > 2:
    folder = argv[1]
    dataset = '/' + argv[2]
else:
    folder = argv[1]
    dataset = ''

def ppi_genes(ppi_file):
    genes = []
    fr = open(ppi_file)
    while True:
        line = fr.readline()
        if not line:
            break
        nd1, nd2 = line.split('\n')[0].split(',')
        genes.append(nd1)
        genes.append(nd2)
    fr.close()
    return set(genes)


def coexpression_net(tissue, ppi_gene_list):
    if dataset == '/brain':
        if tissue.split('_')[1] in ['0000232', '0000233', '0001279']:
            X_train_isoid = np.load('../' + folder + '/input' + dataset + '/train_isoform_list.npy')
            X_test_isoid = np.load('../' + folder + '/input' + dataset + '/test_isoform_list.npy')
        else:
            return 0, 0
    elif dataset == '/major':
        if tissue.split('_')[1] in ['0000763', '0000562', '0001103', '0001379', '0001078', '0001363', '0000775', '0001487', '0000141', '0000648', '0001253', '0001422']:
            X_train_isoid = np.load('../' + folder + '/input' + dataset + '/train_isoform_list.npy')
            X_test_isoid = np.load('../' + folder + '/input' + dataset + '/test_isoform_list.npy')
        else:
            return 0, 0
    else:
        X_train_isoid = np.load('../' + folder + '/input' + dataset + '/train_isoform_list.npy')
        X_test_isoid = np.load('../' + folder + '/input' + dataset + '/test_isoform_list.npy')
    iso_id = np.hstack((X_train_isoid, X_test_isoid))
    pos_map = {}
    cnt = 0
    for iso in iso_id:
        pos_map[iso] = cnt
        cnt += 1
    gene_list = []
    iso_list = []
    fr = open('../' + folder + '/expression/' + tissue + '.txt')
    line = fr.readline()
    while True:
        line = fr.readline()
        if not line:
            break
        iso, gene = line.split('\t')[0:2]
        iso_list.append(iso)
        gene_list.append(gene)
    fr.close()
    gene_list =np.array(gene_list)
    #print(gene_list)
    #print(gene_list.shape)
    iso_list = np.array(iso_list)
    #print(iso_list)
    #print(iso_list.shape)
    exp_mat = genfromtxt('../' + folder + '/expression/' + tissue + '.txt', delimiter='\t')
    exp_mat = exp_mat[1:, 2:]
    iso_mean_exp = np.mean(exp_mat, axis=1)
    #small_idx = np.where(iso_mean_exp <= 0.5)[0]
    #exp_mat[small_idx, :] = 0
    print(iso_mean_exp.shape)
    ppi_idx = []
    gene_in_ppi = []
    iso_in_ppi = []
    iso_exp_map = {}
    for i in range(len(gene_list)):
        if gene_list[i] in ppi_gene_list:
            ppi_idx.append(i)
            gene_in_ppi.append(gene_list[i])
            iso_in_ppi.append(iso_list[i])
            iso_exp_map[iso_list[i]] = iso_mean_exp[i]

    exp_mat = exp_mat[ppi_idx, :]

    print(exp_mat.shape)
    #print(exp_mat)

    cor_net_min = np.ones([exp_mat.shape[0], exp_mat.shape[0]])
    # Calculate co-exp net
    for i in range(exp_mat.shape[1]):
        exp_mat_leave_on_out = np.delete(exp_mat, i, axis=1)
        print(exp_mat_leave_on_out.shape)
        cor_net = np.corrcoef(exp_mat_leave_on_out)
        # Set nan to be zero
        nan_where = np.isnan(cor_net)
        cor_net[nan_where] = 0
        cor_net[cor_net > 0.999] = 0.999
        # Diagnal to be zero
        for i in range(cor_net.shape[0]):
            cor_net[i, i] = 0
        min_index = np.argmin(np.absolute(np.vstack((np.expand_dims(cor_net_min, axis=0), np.expand_dims(cor_net, axis=0)))), axis=0)
        cor_net_min = cor_net_min * (1 - min_index) + cor_net * min_index
        #print(cor_net_min)
    print(np.max(cor_net_min))
    cor_net_min = np.absolute(cor_net_min)

    all_pcc = cor_net_min.flatten()
    par = int(len(all_pcc) * 0.05)
    threshold = all_pcc[np.argpartition(all_pcc, -par)[-par]]
    print(threshold)
    cor_net_min[cor_net_min < threshold] = 0
    print(cor_net_min)
    print(np.shape(cor_net_min))
    idx_r, _ = np.nonzero(cor_net_min)

    cor_mat_all = np.zeros((len(iso_id), len(iso_id)))
    dataset_list_pos = []
    for iso in iso_in_ppi:
        dataset_list_pos.append(pos_map[iso])
    row = np.repeat([dataset_list_pos], len(dataset_list_pos), axis=0)
    col = np.transpose(row)
    #print(row)
    #print(col)
    cor_mat_all[row, col] = cor_net_min

    cor_mat_sparse = sparse.csr_matrix(cor_mat_all)
    dir = '../' + folder + '/input' + dataset + '/co_expression_net/'
    if not os.path.isdir(dir):
        os.mkdir(dir)
    sparse.save_npz('../' + folder + '/input' + dataset + '/co_expression_net/' + tissue + '_coexp_net.npz', cor_mat_sparse)

    return len(idx_r), exp_mat.shape[0] * exp_mat.shape[0]



##############################################################################################################
fr = open('../' + folder + '/GO_terms/tissue_specific_GOs.txt')
tissues = []
while True:
    line = fr.readline()
    if not line:
        break
    tissue = line.split('\t')[0]
    tissues.append(tissue)
fr.close()
print(tissues)

for tissue in tissues:
    ppi_gene_list = ppi_genes('../' + folder + '/tissue_specific_PPIs/' + tissue + '.txt')
    #print(ppi_gene_list)
    print('Construct the co-expression network for ' + tissue + '...')
    num_edges, full_net_edges = coexpression_net(tissue, ppi_gene_list)
    print(tissue, num_edges, full_net_edges, num_edges / full_net_edges)
