"""main function
"""

import numpy as np
import tensorflow as tf
from encoder import Encoder
from predictor import Predictor
from scipy.sparse import csr_matrix
import util
import net_util
import networkx as nx
import copy
from sys import argv
import os
import matplotlib
import time
matplotlib.use('Agg')
import matplotlib.pyplot as plt

script, tissue_idx, dataset = argv


def train_predictor(predictor, X_train_seq, X_train_dm, y_train, X_test_seq,
                    X_test_dm, y_test, X_train_geneid, X_test_geneid, go,
                    np_ratios, eval_repeats, bag_indexes, go_hier, pre_ce_loss,
                    geneid, cur_iter, iii_net, learning_curve_auc,
                    learning_curve_auprc, tissue_id):
    learning_curve_auc, learning_curve_auprc, func_feature, aucs, prcs, pre_ce_loss = predictor.train(
        X_train_seq, X_train_dm, y_train, X_test_seq, X_test_dm, y_test,
        X_train_geneid, X_test_geneid, go, np_ratios, eval_repeats,
        bag_indexes, go_hier, pre_ce_loss, geneid, cur_iter, iii_net,
        learning_curve_auc, learning_curve_auprc)

    prediction, _ = predictor.inference(X_train_seq, X_train_dm, go_hier)

    label_update = np.zeros(y_train.shape)
    for i in range(int(np.max(bag_indexes))):
        idx = np.where(bag_indexes == i)[0]
        if len(idx) == 0:
            continue
        bag_labels = np.max(y_train[idx, :], axis=0)
        for lb in range(y_train.shape[-1]):
            if bag_labels[lb] == 1:
                pos_idx = np.where(prediction[idx, lb] >= 0.0)[0]
                if len(pos_idx) == 0:
                    pos_idx = np.argmax(prediction[idx, lb])
                label_update[idx[pos_idx], lb] = 1
            elif bag_labels[lb] == -1:
                label_update[idx, lb] = -1

    return predictor, learning_curve_auc, learning_curve_auprc, label_update, prediction, func_feature, aucs, prcs, pre_ce_loss


def train_encoder(encoder, X_train_expression, batch_indexes, cur_iter,
                  pos_iso_idx, non_functional_set, geneid, isoid, y_train,
                  y_test, tissue_enhanced_iso, func_feature):
    iii_net = encoder.train(X_train_expression, batch_indexes, cur_iter,
                            pos_iso_idx, non_functional_set, geneid, isoid,
                            y_train, y_test, tissue_enhanced_iso, func_feature)
    return iii_net


def pos_gene_stats(train_labels, gene_ids):
    labels_sum = np.sum(train_labels, axis=1)
    pos_iso_idx = np.where(labels_sum > 0)[0]
    non_functional_set = np.where(labels_sum == 0)[0]
    return pos_iso_idx, non_functional_set


def main():
    tf.compat.v1.disable_eager_execution()
    model_save_dir = '../saved_models'
    iterations = 4
    tissue_id, tissue_name, tissue_gos = util.get_tissue_go(int(tissue_idx))
    print('tissue: ' + tissue_id + '(' + tissue_name + ')')

    X_train_seq, X_train_dm, X_test_seq, X_test_dm, X_train_geneid, \
    X_train_isoid, X_test_geneid, X_test_isoid, X_train_expression = util.get_data(
        tissue_id + '_' + tissue_name, dataset)

    positive_gene_map = util.pos_gene_set(tissue_gos)
    y_train, y_test, np_ratios, eval_repeats, go, go_hier = \
    util.generate_multi_label(
        tissue_id, X_train_geneid, X_test_geneid, positive_gene_map)
    pos_iso_idx, non_functional_set = pos_gene_stats(
        y_train, X_train_geneid)

    X_train_seq = np.vstack((X_train_seq, X_test_seq))
    X_train_dm = np.vstack((X_train_dm, X_test_dm))
    y_train = np.vstack((y_train, -1 * np.ones(y_test.shape)))
    geneid = np.hstack((X_train_geneid, X_test_geneid))
    isoid = np.hstack((X_train_isoid, X_test_isoid))
    geneid_set = list(set(list(geneid)))

    instance_to_bag = np.zeros(len(geneid))
    gene_num = 0
    for id in geneid_set:
        idx = np.where(geneid == id)
        instance_to_bag[idx] = gene_num
        gene_num += 1
    instance_to_bag = instance_to_bag.astype(int)

    print(y_train.shape, y_test.shape)
    print('Training model for ' + tissue_id)

    fr = open('../hyper_prms/' + tissue_id + '_predictor_hprms.txt')
    predictor_config = eval(fr.read())
    fr.close()
    fr = open('../hyper_prms/' + tissue_id + '_encoder_hprms.txt')
    encoder_config = eval(fr.read())
    fr.close()

    print('predictor_config', predictor_config)
    print('encoder_config', encoder_config)

    predictor = Predictor(predictor_config)
    saver = tf.compat.v1.train.Saver()

    ckpt_path = '../saved_models/' + tissue_id + '_predictor_pretrain'

    #Load the model pretrained on SwissProt
    ckpt = tf.train.get_checkpoint_state(ckpt_path)
    if ckpt and ckpt.model_checkpoint_path:
        print('Loading the model pretrained on SwissProt...')
        saver.restore(predictor.sess, ckpt.model_checkpoint_path)
    else:
        print('Train from scratch...')

    # Read tissue specific ppi
    ppi_file_name = '../data/tissue_specific_PPIs/' + tissue_id + '.txt'
    iii_net, genes_with_edges = net_util.read_net(ppi_file_name, len(geneid),
                                                  geneid)

    tissue_enhanced_iso = util.find_tissue_enhanced_isoforms(
        tissue_id + '_' + tissue_name, dataset)

    encoder = Encoder(encoder_config, iii_net)

    print('training model...')
    learning_curve_auc = []
    learning_curve_auprc = []
    pre_ce_loss = np.float('inf')
    for it in range(iterations):
        print('Iteration:', it)

        # Train predictor
        predictor.set_parameters(it)
        predictor, learning_curve_auc, learning_curve_auprc, y_train, prediction, func_feature, aucs, prcs, pre_ce_loss = train_predictor(
            predictor, X_train_seq, X_train_dm, y_train, X_test_seq, X_test_dm,
            y_test, X_train_geneid, X_test_geneid, go, np_ratios,
            eval_repeats, instance_to_bag, go_hier, pre_ce_loss, geneid, it,
            iii_net, learning_curve_auc, learning_curve_auprc, tissue_id)

        ckpt_path = "../saved_models/saved_ckpt/" + tissue_id + "_iter" + str(it)
        if not os.path.exists(ckpt_path):
            os.makedirs(ckpt_path)
        saver.save(predictor.sess, save_path=ckpt_path + "/predictor.ckpt")

        if it + 1 < iterations:
            pos_iso_idx, non_functional_set = pos_gene_stats(
                y_train, geneid)
            iii_net = train_encoder(
                encoder, X_train_expression, instance_to_bag, it, pos_iso_idx,
                non_functional_set, geneid, isoid, y_train, y_test,
                tissue_enhanced_iso, func_feature)

    print('Saving model and results...')
    util.write_result(tissue_id, prediction, positive_gene_map,
                      geneid, isoid, aucs, prcs)
    ckpt_path = '../saved_models/' + tissue_id + '_predictor_final'
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)
    saver.save(predictor.sess, save_path=ckpt_path + '/predictor.ckpt')

    predictor.sess.close()
    encoder.sess.close()


if __name__ == "__main__":
    main()
