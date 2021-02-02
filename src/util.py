import numpy as np
import tensorflow as tf
import matplotlib
import os
from scipy import sparse
from numpy import genfromtxt
import time
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def get_data(tissue_id, dataset):
    #Load dataset
    if dataset == 'major':
        X_train_seq = np.load('../data/input/major/human_sequence_train.npz')['data']
        X_train_dm = np.load('../data/input/major/human_domain_train.npy')
        X_test_seq = np.load('../data/input/major/human_sequence_test.npz')['data']
        X_test_dm = np.load('../data/input/major/human_domain_test.npy')
        X_train_geneid = np.load('../data/input/major/train_gene_list.npy')
        X_train_isoid = np.load('../data/input/major/train_isoform_list.npy')
        X_test_geneid = np.load('../data/input/major/test_gene_list.npy')
        X_test_isoid = np.load('../data/input/major/test_isoform_list.npy')

        X_expression = sparse.load_npz('../data/input/major/co_expression_net/' + tissue_id + '_coexp_net.npz')

    elif dataset == 'brain':
        X_train_seq = np.load('../data/input/brain/human_sequence_train.npz')['data']
        X_train_dm = np.load('../data/input/brain/human_domain_train.npy')
        X_test_seq = np.load('../data/input/brain/human_sequence_test.npz')['data']
        X_test_dm = np.load('../data/input/brain/human_domain_test.npy')
        X_train_geneid = np.load('../data/input/brain/train_gene_list.npy')
        X_train_isoid = np.load('../data/input/brain/train_isoform_list.npy')
        X_test_geneid = np.load('../data/input/brain/test_gene_list.npy')
        X_test_isoid = np.load('../data/input/brain/test_isoform_list.npy')

        X_expression = sparse.load_npz('../data/input/brain/co_expression_net/' + tissue_id + '_coexp_net.npz')

    X_train_dm = X_train_dm[:, -15:]
    X_test_dm = X_test_dm[:, -15:]
    print(X_train_seq.shape)
    print(X_test_seq.shape)
    print(X_train_dm.shape)
    print(X_test_dm.shape)

    return X_train_seq, X_train_dm, X_test_seq, X_test_dm, X_train_geneid, X_train_isoid, X_test_geneid, X_test_isoid, X_expression


def sample_by_edges_predictor(adjacency_list, nb_num):
    sample_size = len(adjacency_list.keys())
    batches = []
    batch_indexes = []
    batch_neighor_idx = []
    batch_neighor_mask = []
    min_batch_size = 800
    random_indexes = np.random.permutation(sample_size)
    for i in random_indexes:
        samples = sample_instance_edges(adjacency_list, i, nb_num)
        batch_indexes.extend(samples)
        batch_neighor_idx.append(len(samples))
        batch_indexes = list(set(batch_indexes))
        if len(batch_indexes) >= min_batch_size:
            batches.append(batch_indexes)
            batch_indexes = []
    if batch_indexes:
        batches.append(batch_indexes)
    return batches


def sample_instance_edges(adjacency_list, idx, nb_num):
    samples = [idx]
    ad_list = adjacency_list[idx]
    if ad_list:
        ad_list_len = len(ad_list)
        if ad_list_len <= nb_num:
            samples.extend(ad_list)
        else:
            random_nbr = []
            while len(random_nbr) < nb_num:
                nbr = np.random.randint(ad_list_len)
                if nbr not in random_nbr and ad_list[nbr] != idx:
                    random_nbr.append(nbr)
                    samples.append(ad_list[nbr])
    return samples


def group_samples_by_lengths(indexes, seq_features):
    group_s = []
    group_m = []
    group_l = []
    group_xl = []
    nonspace = np.sign(seq_features)
    lengths = np.sum(nonspace, 1)
    for i in range(len(indexes)):
        if lengths[i] < 2000:
            group_s.append(indexes[i])
        elif lengths[i] < 4000:
            group_m.append(indexes[i])
        elif lengths[i] < 8000:
            group_l.append(indexes[i])
        else:
            group_xl.append(indexes[i])
    return group_s, group_m, group_l, group_xl


#Generate labels
def generate_multi_label(tissue_id, X_train_geneid, X_test_geneid, positive_gene_map):
    def generate_label(X_train_geneid, X_test_geneid, positive_gene):
        y_train = np.array([])
        y_test = np.array([])
        train_pos_iso_num = 0
        test_pos_num = 0
        test_gene_num = 0

        last_gID = ''
        for gID in X_train_geneid:
            if gID != last_gID:
                if gID in positive_gene:
                    y_train = np.hstack((y_train, np.ones(1)))
                    train_pos_iso_num += 1
                else:
                    y_train = np.hstack((y_train, np.zeros(1)))
                last_gID = gID
            else:
                y_train = np.hstack((y_train, y_train[-1]))
                if y_train[-1] == 1:
                    train_pos_iso_num += 1

        for gID in X_test_geneid:
            if gID != last_gID:
                test_gene_num += 1
                if gID in positive_gene:
                    test_pos_num += 1
                    y_test = np.hstack((y_test, np.ones(1)))
                else:
                    y_test = np.hstack((y_test, np.zeros(1)))
                last_gID = gID
            else:
                y_test = np.hstack((y_test, y_test[-1]))

        eval_pos_repeat = int(np.ceil((test_gene_num - test_pos_num) / (test_pos_num * 9.)))

        neg_pos_ratio = (len(X_train_geneid) - train_pos_iso_num) / train_pos_iso_num
        return y_train, y_test, eval_pos_repeat, neg_pos_ratio

    label_path = "../data/tmp_data/" + tissue_id + "_labels.npy"
    if label_path and os.path.exists(label_path):
        y_train, y_test, np_ratios, eval_repeats = np.load(
            label_path, allow_pickle=True)
    else:
        y_train = np.array([])
        y_test = np.array([])
        np_ratios = []
        eval_repeats = []
        for go in positive_gene_map.keys():
            print(go)
            y_tr, y_te, eval_pos_repeat, neg_pos_ratio = generate_label(
                X_train_geneid, X_test_geneid, positive_gene_map[go])
            eval_repeats.append(eval_pos_repeat)
            np_ratios.append(neg_pos_ratio)
            if len(y_train) == 0:
                y_train = np.expand_dims(y_tr, 1)
                y_test = np.expand_dims(y_te, 1)
            else:
                y_train = np.hstack((y_train, np.expand_dims(y_tr, -1)))
                y_test = np.hstack((y_test, np.expand_dims(y_te, -1)))

        eval_repeats = np.array(eval_repeats)
        np_ratios = np.array(np_ratios)
    if label_path:
        np.save(
            label_path, np.array([y_train, y_test, np_ratios, eval_repeats]))

    go_ancestors = np.load('../data/GO_terms/go_ancestors.npy', allow_pickle=True)
    go_ancestors = go_ancestors[0]
    num_terms = len(positive_gene_map)
    go_hier = np.zeros([num_terms, num_terms])
    gos = [go for go in positive_gene_map.keys()]
    for i in range(num_terms):
        for j in range(num_terms):
            if gos[i] in go_ancestors[gos[j]]:
                go_hier[i, j] = 1.0

    return y_train, y_test, np_ratios, eval_repeats, gos, go_hier


def pos_gene_set(selected_tissue_gos):
    def parse_annotation_file(goa_file, positive_gene_map):
        fr = open(goa_file)
        while True:
            line = fr.readline()
            if not line:
                break
            line = line.split('\n')[0]
            gene = line.split('\t')[0]
            GO = line.split('\t')[1:]
            for selected_go in selected_tissue_gos:
                if selected_go in GO:
                    if selected_go not in positive_gene_map.keys():
                        positive_gene_map[selected_go] = [gene]
                    else:
                        gene_set = positive_gene_map[selected_go]
                        gene_set.append(gene)
                        positive_gene_map[selected_go] = gene_set
        fr.close()

    positive_gene_map = {}
    parse_annotation_file('../data/GO_annotations/human_annotations.txt', positive_gene_map)

    return positive_gene_map


def get_tissue_go(tissue_idx):
    fr = open('../data/GO_terms/tissue_specific_GOs.txt')
    for i in range(tissue_idx):
        line = fr.readline()
        if not line:
            break
        columns = line.split('\n')[0].split('\t')
        tissue_id, tissue_name = columns[0:2]
        tissue_name = '_'.join(tissue_name.split('(')[0][:-1].split(' '))
        go_terms = columns[2:]
    fr.close()
    return tissue_id, tissue_name, go_terms


def find_tissue_enhanced_isoforms(tissue_id, dataset):
    iso_list = []
    fr = open('../data/expression/' + tissue_id + '.txt')
    line = fr.readline()
    while True:
        line = fr.readline()
        if not line:
            break
        iso, _ = line.split('\t')[0:2]
        iso_list.append(iso)

    tissues = []
    if dataset == 'major':
        fold = 4.0
        fr = open('../data/expression/major_tissue_dataset.txt')
        while True:
            line = fr.readline()
            if not line:
                break
            tissue = '_'.join(line.split('\n')[0].split(' '))
            tissues.append(tissue)
        fr.close()
    elif dataset == 'brain':
        fold = 2.0
        fr = open('../data/expression/brain_dataset.txt')
        while True:
            line = fr.readline()
            if not line:
                break
            tissue = '_'.join(line.split('\n')[0].split(' '))
            tissues.append(tissue)
        fr.close()

    tissue_exp_map = {}
    other_tissue_exp_map = {}
    for tissue in tissues:
        exp_mat = genfromtxt('../data/expression/' + tissue + '.txt', delimiter='\t')
        exp_mat = exp_mat[1:, 2:]
        iso_mean_exp = np.mean(exp_mat, axis=1)
        if tissue != tissue_id:
            for i in range(len(iso_list)):
                if iso_list[i] not in other_tissue_exp_map:
                    other_tissue_exp_map[iso_list[i]] = [iso_mean_exp[i]]
                else:
                    other_tissue_exp_map[iso_list[i]].append(iso_mean_exp[i])
        else:
            for i in range(len(iso_list)):
                tissue_exp_map[iso_list[i]] = iso_mean_exp[i]

    tissue_enhanced_iso = []
    for iso in iso_list:
        if tissue_exp_map[iso] and tissue_exp_map[iso] >= fold * np.mean(other_tissue_exp_map[iso]):
            tissue_enhanced_iso.append(iso)

    return tissue_enhanced_iso


def write_result(tissue_id, prediction, positive_gene_map, geneid, isoid, aucs, prcs):
    cnt = 0
    print(prediction.shape)
    for go in positive_gene_map.keys():
        fw = open('../results/GO_predictions/predictions_' + tissue_id + '_'+ go + '.txt', 'w')
        for j in range(len(isoid)):
            fw.write(isoid[j] + '\t')
            fw.write(geneid[j] + '\t')
            fw.write(str(1. / (1. + np.exp(-prediction[j, cnt]))) + '\n')
        fw.close()
        cnt += 1
    fw = open('../results/perf_eval/' + tissue_id + '.txt', 'w')
    i = 0
    fw.write('GO term\tAUC\tAUPRC\n')
    for go in positive_gene_map.keys():
        fw.write(go + '\t' + str(aucs[i]) + '\t' + str(prcs[i]) + '\n')
        i += 1
    fw.close()
    return
