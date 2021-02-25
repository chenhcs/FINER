"""functional predictor
"""

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import tensorflow as tf
from PyramidPooling import PyramidPooling
from tensorflow.keras.layers import Dense, Dropout, SpatialDropout1D
from tensorflow.keras.layers import Activation, Flatten, Input, Masking
from tensorflow.keras.layers import Conv1D, MaxPool1D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Concatenate
from tensorflow.keras import regularizers
import copy
import util
import net_util
import networkx as nx


class Predictor(object):
    def __init__(self, config):
        print("initializing predictor...")
        self.ft_learning_rate = config["finetune_learning_rate"]
        self.ft_learning_rate_decay_factor = config[
            "finetune_learning_rate_decay_factor"]
        self.ft_decay_step = config["finetune_decay_step"]
        self.neighbor_loss_hp = config["neighbor_loss_hp"]
        self.lamda_increments = config["lamda_increments"]
        self.epochs = config["epochs"]
        self.initial_epochs = config["initial_epochs"]
        self.class_num = config["class_num"]
        self.nb_num = config["nb_num"]
        self.train_size = config["train_size"]
        self.test_size = config["test_size"]
        self.iteration_lr_decay_factor = config["iteration_lr_decay_factor"]
        self.finetune = False

        print("launching session for predictor")
        self.sess = tf.compat.v1.Session()

        self.seq_features_s = tf.compat.v1.placeholder(tf.float32,
                                                       [None, None])
        self.dm_features_s = tf.compat.v1.placeholder(tf.float32, [None, None])
        self.seq_features_m = tf.compat.v1.placeholder(tf.float32,
                                                       [None, None])
        self.dm_features_m = tf.compat.v1.placeholder(tf.float32, [None, None])
        self.seq_features_l = tf.compat.v1.placeholder(tf.float32,
                                                       [None, None])
        self.dm_features_l = tf.compat.v1.placeholder(tf.float32, [None, None])
        self.seq_features_xl = tf.compat.v1.placeholder(
            tf.float32, [None, None])
        self.dm_features_xl = tf.compat.v1.placeholder(tf.float32,
                                                       [None, None])
        self.labels = tf.compat.v1.placeholder(tf.float32, [None, None])
        self.np_ratios = tf.compat.v1.placeholder(tf.float32, [None, None])
        self.instance_weight = tf.compat.v1.placeholder(
            tf.float32, [None, None])
        self.go_hier = tf.compat.v1.placeholder(tf.float32, [None, None])
        self.unlabeled_mask = tf.compat.v1.placeholder(tf.float32, [
            None,
        ])
        self.neighbor_sim = tf.compat.v1.placeholder(tf.float32, [None, None])
        self.output_s, self.last_layer_embedding_s = self.model(
            self.seq_features_s, self.dm_features_s, reuse=False)
        self.output_m, self.last_layer_embedding_m = self.model(
            self.seq_features_m, self.dm_features_m)
        self.output_l, self.last_layer_embedding_l = self.model(
            self.seq_features_l, self.dm_features_l)
        self.output_xl, self.last_layer_embedding_xl = self.model(
            self.seq_features_xl, self.dm_features_xl)
        self.output = tf.concat(
            [self.output_s, self.output_m, self.output_l, self.output_xl],
            axis=0)
        self.last_layer_embedding = tf.concat([
            self.last_layer_embedding_s, self.last_layer_embedding_m,
            self.last_layer_embedding_l, self.last_layer_embedding_xl], axis=0)

        self.global_step = tf.Variable(0, trainable=False)
        self.train_op = self.apply_loss_function(self.global_step)
        self.sess.run(tf.compat.v1.global_variables_initializer())

        return

    def model(self, seq_features, dm_features, reuse=True):
        def go_hier_pred(prediction_mat):
            prediction_mat_tile = tf.tile(tf.expand_dims(prediction_mat, 1),
                                          [1, tf.shape(self.go_hier)[-1], 1])
            go_hier_tile = tf.tile(tf.expand_dims(self.go_hier, 0),
                                   [tf.shape(prediction_mat)[0], 1, 1])
            prediction_tile_masked = tf.multiply(
                prediction_mat_tile, go_hier_tile) - tf.multiply(
                    tf.fill(tf.shape(go_hier_tile), 1e8), 1 - go_hier_tile)
            hier_prediction_mat = tf.reduce_max(prediction_tile_masked,
                                                axis=-1)
            return hier_prediction_mat

        with tf.compat.v1.variable_scope("predictor", reuse=reuse):
            x1 = Embedding(input_dim=8001,
                           output_dim=8,
                           name="layers/seq_embedding")(seq_features)
            x1 = Conv1D(filters=64,
                        kernel_size=64,
                        strides=1,
                        padding='same',
                        activation='relu',
                        name="layers/seq_conv")(x1)
            x1 = PyramidPooling([1, 2, 4, 8])(x1)
            x1 = Dense(128,
                       kernel_regularizer=regularizers.l2(0.15),
                       name="layers/seq_dense")(x1)
            x1 = Activation('relu')(x1)
            seq_output = Activation('relu')(x1)

            x2 = Embedding(input_dim=17331,
                           output_dim=8,
                           input_length=15,
                           mask_zero=True,
                           name="layers/dm_embedding")(dm_features)
            domain_output = LSTM(128, name="layers/dm_lstm")(x2)
            x = Concatenate()([seq_output, domain_output])

            dense = Dense(128,
                          kernel_regularizer=regularizers.l2(0.15),
                          name="layers/fc_1")(x)
            x = Activation('relu')(dense)
            output = Dense(self.class_num,
                           kernel_regularizer=regularizers.l2(0.15),
                           name="last_layers/fc_2")(x)
            hier_output = go_hier_pred(output)

        return hier_output, dense

    def apply_loss_function(self, global_step):
        unlabeled_mask = tf.tile(tf.expand_dims(self.unlabeled_mask, -1),
                                 [1, self.class_num])
        weight_mask = tf.multiply(tf.sigmoid(self.instance_weight),
                                  unlabeled_mask)
        weight_mask_upsample = tf.multiply(self.labels,
                                           (self.np_ratios - 1)) + 1
        weight_mask = tf.multiply(weight_mask, weight_mask_upsample)
        cross_entropy = tf.multiply(
            tf.maximum(self.output, 0) -
            tf.multiply(self.output, self.labels) +
            tf.math.log(1 + tf.exp(-tf.abs(self.output))), weight_mask)
        self.ce_loss = tf.reduce_mean(input_tensor=cross_entropy)

        def pairwise_l2_dis(mat):
            r = tf.reduce_sum(input_tensor=tf.multiply(mat, mat), axis=1)
            r = tf.reshape(r, [-1, 1])
            sqr_d = r - 2 * tf.matmul(mat, tf.transpose(a=mat)) + tf.transpose(
                a=r)
            return sqr_d

        self.l2_dis = pairwise_l2_dis(self.last_layer_embedding)
        pairwise_unlabeled_mask = tf.reshape(self.unlabeled_mask,
                                             [-1, 1]) + self.unlabeled_mask
        pairwise_unlabeled_mask = tf.clip_by_value(pairwise_unlabeled_mask, 0,
                                                   1)
        nb_loss = tf.reduce_mean(input_tensor=tf.multiply(
            self.l2_dis, tf.multiply(self.neighbor_sim,
                                     pairwise_unlabeled_mask)))

        self.lamda = tf.Variable(self.neighbor_loss_hp,
                                 trainable=False,
                                 name="neighbor_loss_hp")
        self.neighbor_loss = tf.multiply(self.lamda, nb_loss)
        self.loss = self.ce_loss + self.neighbor_loss

        self.learning_rate = tf.Variable(self.ft_learning_rate,
                                         trainable=False,
                                         name="learning_rate")
        self.learning_rate_decay_factor = tf.Variable(
            self.ft_learning_rate_decay_factor,
            trainable=False,
            name="learning_rate_decay_factor")
        self.decay_step = tf.Variable(self.ft_decay_step,
                                      trainable=False,
                                      name="decay_step")
        self.lr = tf.compat.v1.train.exponential_decay(
            self.learning_rate,
            global_step,
            self.decay_step,
            self.learning_rate_decay_factor,
            staircase=True)
        opt = tf.compat.v1.train.AdamOptimizer(learning_rate=self.lr)

        return opt.minimize(self.loss, global_step=global_step)

    def set_parameters(self, iter):
        self.finetune = True
        global_step = self.global_step.assign(0)
        learning_rate = self.learning_rate.assign(
            self.ft_learning_rate *
            np.power(self.iteration_lr_decay_factor, np.clip(iter - 1, 0 , None)))
        decay_step = self.decay_step.assign(self.ft_decay_step)
        learning_rate_decay_factor = self.learning_rate_decay_factor.assign(
            self.ft_learning_rate_decay_factor)
        assign_lamda = self.lamda.assign(
            self.neighbor_loss_hp *
            np.clip(np.power(self.lamda_increments, iter), 0, 1000))
        self.sess.run([global_step, learning_rate, decay_step, learning_rate_decay_factor, assign_lamda])

    def train(self, seq_features, dm_features, labels, seq_features_test,
              dm_features_test, labels_test, X_train_geneid, X_test_geneid,
              go, np_ratios, eval_repeats, bag_indexes, go_hier,
              pre_ce_loss_epoch, gene_list=[], cur_iter=0, iii_net=None,
              learning_curve_auc=[], learning_curve_auprc=[]):
        print("training predictor...")
        print(tf.compat.v1.trainable_variables())

        num_epochs = self.initial_epochs
        sparse_iii_net = nx.adjacency_matrix(iii_net)

        train_idx_map = {}
        test_idx_map = {}
        for i in range(len(X_train_geneid)):
            gid = X_train_geneid[i]
            if gid not in train_idx_map:
                train_idx_map[gid] = [i]
            else:
                train_idx_map[gid].append(i)
        for i in range(len(X_test_geneid)):
            gid = X_test_geneid[i]
            if gid not in test_idx_map:
                test_idx_map[gid] = [i]
            else:
                test_idx_map[gid].append(i)
        if cur_iter == 0:
            prediction_test, _ = self.inference(seq_features_test,
                                                dm_features_test, go_hier)
            print(prediction_test.shape)
            aucs = []
            prcs = []
            for i in range(prediction_test.shape[1]):
                y_test_gene = []
                y_pred_gene = []
                y_test_gene_prc = []
                y_pred_gene_prc = []
                y_test = labels_test[:, i]
                y_pred = prediction_test[:, i]
                for gid in set(list(X_test_geneid)):
                    idx = test_idx_map[gid]
                    y_test_gene.append(np.max(y_test[idx]))
                    y_pred_gene.append(np.max(y_pred[idx]))
                    if np.max(y_test[idx]) == 1:
                        y_test_gene_prc.extend([np.max(y_test[idx])] *
                                               eval_repeats[i])
                        y_pred_gene_prc.extend([np.max(y_pred[idx])] *
                                               eval_repeats[i])
                    else:
                        y_test_gene_prc.append(np.max(y_test[idx]))
                        y_pred_gene_prc.append(np.max(y_pred[idx]))
                auc_roc = roc_auc_score(y_test_gene, y_pred_gene)
                aucs.append(auc_roc)
                auprc = average_precision_score(y_test_gene_prc,
                                                y_pred_gene_prc)
                prcs.append(auprc)
            for i in range(len(go)):
                print(go[i], aucs[i], prcs[i])
            print('AUC:' + str(np.mean(aucs)))
            print('AUPRC:' + str(np.mean(prcs)))
            learning_curve_auc.append(np.mean(aucs))
            learning_curve_auprc.append(np.mean(prcs))
        else:
            num_epochs = self.epochs

        for epoch in range(num_epochs):
            if iii_net is None:
                batch_size = 800
                batch_indexes = []
                random_indexes = np.random.permutation(len(seq_features))
                for i in range(0, len(random_indexes), batch_size):
                    batch_indexes.append(random_indexes[i:i + batch_size])
            else:
                print("Start sampling batches")
                batch_indexes = util.sample_by_edges_predictor(
                    net_util.adjacency(iii_net), self.nb_num)
                print("Finish sampling batches")
            ce_loss_epoch = 0
            neighbor_loss_epoch = 0
            for i in range(len(batch_indexes)):
                indexes = batch_indexes[i]
                group_s, group_m, group_l, group_xl = util.group_samples_by_lengths(
                    indexes, seq_features[indexes])
                klen_s = 2000
                klen_m = 4000
                klen_l = 8000
                klen_xl = 16000
                indexes_groups = np.int_(
                    np.hstack((group_s, group_m, group_l, group_xl)))
                labels_batch = labels[indexes_groups]
                seq_features_batch_s = seq_features[
                    group_s][:, len(seq_features[0]) - klen_s:]
                seq_features_batch_m = seq_features[
                    group_m][:, len(seq_features[0]) - klen_m:]
                seq_features_batch_l = seq_features[
                    group_l][:, len(seq_features[0]) - klen_l:]
                seq_features_batch_xl = seq_features[
                    group_xl][:, len(seq_features[0]) - klen_xl:]
                dm_features_batch_s = dm_features[group_s]
                dm_features_batch_m = dm_features[group_m]
                dm_features_batch_l = dm_features[group_l]
                dm_features_batch_xl = dm_features[group_xl]

                np_ratios_tile = np.tile(np_ratios, [len(indexes_groups), 1])
                unlabeled_mask_batch = copy.deepcopy(labels_batch)
                unlabeled_mask_batch[unlabeled_mask_batch >= 0] = 1
                unlabeled_mask_batch[unlabeled_mask_batch < 0] = 0
                unlabeled_mask_batch = unlabeled_mask_batch[:, 0]
                if iii_net is None:
                    neighbor_sim_batch = np.zeros(
                        [len(indexes_groups),
                         len(indexes_groups)])
                    instance_weight = self.sess.run(
                        self.output,
                        feed_dict={
                            self.seq_features_s: seq_features_batch_s,
                            self.seq_features_m: seq_features_batch_m,
                            self.seq_features_l: seq_features_batch_l,
                            self.seq_features_xl: seq_features_batch_xl,
                            self.dm_features_s: dm_features_batch_s,
                            self.dm_features_m: dm_features_batch_m,
                            self.dm_features_l: dm_features_batch_l,
                            self.dm_features_xl: dm_features_batch_xl,
                            self.go_hier: go_hier
                        })
                else:
                    neighbor_sim_batch = sparse_iii_net[
                        indexes_groups, :][:, indexes_groups].todense()
                    instance_weight = self.sess.run(
                        self.output,
                        feed_dict={
                            self.seq_features_s: seq_features_batch_s,
                            self.seq_features_m: seq_features_batch_m,
                            self.seq_features_l: seq_features_batch_l,
                            self.seq_features_xl: seq_features_batch_xl,
                            self.dm_features_s: dm_features_batch_s,
                            self.dm_features_m: dm_features_batch_m,
                            self.dm_features_l: dm_features_batch_l,
                            self.dm_features_xl: dm_features_batch_xl,
                            self.go_hier: go_hier
                        })

                _, ce_loss_batch, neighbor_loss_batch = self.sess.run(
                    [self.train_op, self.ce_loss, self.neighbor_loss],
                    feed_dict={
                        self.seq_features_s: seq_features_batch_s,
                        self.seq_features_m: seq_features_batch_m,
                        self.seq_features_l: seq_features_batch_l,
                        self.seq_features_xl: seq_features_batch_xl,
                        self.dm_features_s: dm_features_batch_s,
                        self.dm_features_m: dm_features_batch_m,
                        self.dm_features_l: dm_features_batch_l,
                        self.dm_features_xl: dm_features_batch_xl,
                        self.labels: labels_batch,
                        self.np_ratios: np_ratios_tile,
                        self.unlabeled_mask: unlabeled_mask_batch,
                        self.neighbor_sim: neighbor_sim_batch,
                        self.instance_weight: instance_weight,
                        self.go_hier: go_hier
                    })
                ce_loss_epoch += ce_loss_batch
                neighbor_loss_epoch += neighbor_loss_batch
            lr_curr = self.sess.run([self.lr])
            print("epoch:", epoch, "ce loss =", ce_loss_epoch,
                  "neighbor loss =", neighbor_loss_epoch, "current_lr = ",
                  lr_curr)

            prediction_test, _ = self.inference(seq_features_test,
                                                dm_features_test, go_hier)
            if epoch == num_epochs - 1:
                prediction, last_layer_embedding = self.inference(
                    seq_features, dm_features, go_hier)
                prediction_train = prediction[:len(X_train_geneid)]

            aucs = []
            prcs = []
            for i in range(prediction_test.shape[1]):
                y_test_gene = []
                y_pred_gene = []
                y_test_gene_prc = []
                y_pred_gene_prc = []
                y_test = labels_test[:, i]
                y_pred = prediction_test[:, i]
                for gid in set(list(X_test_geneid)):
                    idx = test_idx_map[gid]
                    y_test_gene.append(np.max(y_test[idx]))
                    y_pred_gene.append(np.max(y_pred[idx]))
                    if np.max(y_test[idx]) == 1:
                        y_test_gene_prc.extend([np.max(y_test[idx])] *
                                               eval_repeats[i])
                        y_pred_gene_prc.extend([np.max(y_pred[idx])] *
                                               eval_repeats[i])
                    else:
                        y_test_gene_prc.append(np.max(y_test[idx]))
                        y_pred_gene_prc.append(np.max(y_pred[idx]))

                auc_roc = roc_auc_score(y_test_gene, y_pred_gene)
                aucs.append(auc_roc)
                auprc = average_precision_score(y_test_gene_prc,
                                                y_pred_gene_prc)
                prcs.append(auprc)

            for i in range(len(go)):
                print(go[i], aucs[i], prcs[i])
            print('AUC:' + str(np.mean(aucs)))
            print('AUPRC:' + str(np.mean(prcs)))
            learning_curve_auc.append(np.mean(aucs))
            learning_curve_auprc.append(np.mean(prcs))
            if ce_loss_epoch > pre_ce_loss_epoch:
                cur_learning_rate = self.sess.run(self.learning_rate)
                new_learning_rate = self.learning_rate.assign(
                    cur_learning_rate * 0.5)
                self.sess.run(new_learning_rate)
            pre_ce_loss_epoch = ce_loss_epoch

        return learning_curve_auc, learning_curve_auprc, last_layer_embedding, aucs, prcs, pre_ce_loss_epoch

    def inference(self, seq_features, dm_features, go_hier):
        batch_size = 256
        prediction_all = np.array([])
        last_layer_embedding_all = np.array([])
        all_indexes = np.arange(len(seq_features))
        group_s, group_m, group_l, group_xl = util.group_samples_by_lengths(
            all_indexes, seq_features)
        seql = 1
        for gp in [group_s, group_m, group_l, group_xl]:
            seql *= 2000
            for idx in range(0, len(gp), batch_size):
                seq_features_batch = seq_features[
                    gp[idx:idx +
                       batch_size]][:, len(seq_features[0]) - seql:]
                dm_features_batch = dm_features[gp[idx:idx + batch_size]]
                prediction_batch, last_layer_embedding_batch = self.sess.run(
                    [self.output_s, self.last_layer_embedding_s],
                    feed_dict={
                        self.seq_features_s: seq_features_batch,
                        self.dm_features_s: dm_features_batch,
                        self.go_hier: go_hier
                    })
                if len(prediction_all) == 0:
                    prediction_all = prediction_batch
                    last_layer_embedding_all = last_layer_embedding_batch
                else:
                    prediction_all = np.vstack(
                        (prediction_all, prediction_batch))
                    last_layer_embedding_all = np.vstack(
                        (last_layer_embedding_all, last_layer_embedding_batch))
        index_order = np.int_(np.hstack((group_s, group_m, group_l, group_xl)))
        original_place = np.argsort(index_order)
        prediction_all = prediction_all[original_place]
        return prediction_all, last_layer_embedding_all
