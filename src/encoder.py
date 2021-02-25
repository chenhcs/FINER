"""III network refinement module
"""

import random
import numpy as np
import tensorflow as tf
import copy
import util
import net_util
import networkx as nx
from tensorflow.keras.layers import Embedding
from sklearn.preprocessing import normalize

class Encoder(object):
    def __init__(self,
                 config,
                 ppi_net,
                 num_walks=8,
                 walk_length=10,
                 window_size=10):
        # Initialize setting
        print("initializing encoder")
        self.emb_dim = config["embedding_dim"]
        self.N_size = config["node_size"]
        self.link_thresh = config["link_thresh"]
        self.train_size = config["train_size"]
        self.test_size = config["test_size"]

        self.learning_rate = config["learning_rate"]
        self.learning_rate_decay_factor = config["learning_rate_decay_factor"]
        self.decay_step = config["decay_step"]

        self.epochs = config["epochs"]
        self.first_epochs = config["first_epochs"]

        self.num_walks = num_walks
        self.walk_length = walk_length
        self.p = config["p"]
        self.q = config["q"]
        self.positive = config["positive"]
        self.negative = config["negative"]
        self.min_degree = config["min_degree"]
        self.lambda_func = config["lambda_func"]
        self.window = window_size

        self.ppi_net = ppi_net
        self.iii_net = copy.deepcopy(self.ppi_net)

        # Setup session
        print("launching session for encoder")
        self.sess = tf.compat.v1.Session()
        self.center_input = tf.compat.v1.placeholder(tf.float32, [None,])
        self.center_repeat_link = tf.compat.v1.placeholder(tf.float32, [None,])
        self.center_repeat_nolink = tf.compat.v1.placeholder(tf.float32, [None,])
        self.link_dst_input = tf.compat.v1.placeholder(tf.float32, [None,])
        self.nolink_dst_input = tf.compat.v1.placeholder(tf.float32, [None,])
        self.center_repeat_ctx = tf.compat.v1.placeholder(tf.float32, [None,])
        self.center_repeat_neg = tf.compat.v1.placeholder(tf.float32, [None,])
        self.context_input = tf.compat.v1.placeholder(tf.float32, [None,])
        self.negative_input = tf.compat.v1.placeholder(tf.float32, [None,])
        self.link_labels = tf.compat.v1.placeholder(tf.float32, [None,])
        self.nolink_labels = tf.compat.v1.placeholder(tf.float32, [None,])
        self.context_labels = tf.compat.v1.placeholder(tf.float32, [None,])
        self.negative_labels = tf.compat.v1.placeholder(tf.float32, [None,])
        self.W = tf.Variable(
            tf.random.normal([self.emb_dim,], mean=1.0, stddev=0.01), name="W")
        self.W_const = tf.compat.v1.placeholder(tf.float32, [None,])
        self.coexp_weight = tf.compat.v1.placeholder(tf.float32, [None,])

        self.emb_center, self.emb_center_repeat_link, \
        self.emb_center_repeat_nolink, self.emb_link, self.emb_nolink, \
        self.emb_center_repeat_ctx, self.emb_center_repeat_neg, \
        self.emb_context, self.emb_negative \
        = self.model1(reuse=False)

        self.emb_center_repeat_link_const = tf.compat.v1.placeholder(
            tf.float32, [None, None])
        self.emb_link_const = tf.compat.v1.placeholder(tf.float32, [None, None])
        self.emb_center_repeat_nolink_const = tf.compat.v1.placeholder(
            tf.float32, [None, None])
        self.emb_nolink_const = tf.compat.v1.placeholder(
            tf.float32, [None, None])
        self.func_feature_sim = tf.compat.v1.placeholder(tf.float32, [None,])

        self.embWemb_1, self.link_probs_1 = self.model2(
            self.W_const, self.emb_center_repeat_link, self.emb_link,
            self.emb_center_repeat_nolink, self.emb_nolink)

        self.embWemb_2, self.link_probs_2 = self.model2(
            self.W, self.emb_center_repeat_link_const, self.emb_link_const,
            self.emb_center_repeat_nolink_const, self.emb_nolink_const)
        self.link_weights = tf.compat.v1.placeholder(tf.float32, [None,])

        self.cur_iter = tf.Variable(0, trainable=False)
        self.global_step = tf.Variable(0, trainable=False)
        self.train_op_emb = self.apply_loss_function_emb(self.global_step)
        self.train_op_w = self.apply_loss_function_w(self.global_step)
        self.sess.run(tf.compat.v1.global_variables_initializer())

        return

    def model1(self, reuse=True):
        """Embedding layers"""
        with tf.compat.v1.variable_scope("encoder/model1", reuse=reuse):
            center_embedding = Embedding(self.N_size,
                                         output_dim=self.emb_dim,
                                         input_length=1)
            context_embedding = Embedding(self.N_size,
                                          output_dim=self.emb_dim,
                                          input_length=1)
            emb_center = center_embedding(self.center_input)
            emb_center_repeat_link = center_embedding(self.center_repeat_link)
            emb_center_repeat_nolink = center_embedding(
                self.center_repeat_nolink)
            emb_link = center_embedding(self.link_dst_input)
            emb_nolink = center_embedding(self.nolink_dst_input)
            emb_center_repeat_ctx = center_embedding(self.center_repeat_ctx)
            emb_center_repeat_neg = center_embedding(self.center_repeat_neg)
            emb_context = context_embedding(self.context_input)
            emb_negative = context_embedding(self.negative_input)
        return emb_center, emb_center_repeat_link, emb_center_repeat_nolink, emb_link, emb_nolink, emb_center_repeat_ctx, emb_center_repeat_neg, emb_context, emb_negative

    def model2(self, W, emb_center_repeat_link, emb_link,
               emb_center_repeat_nolink, emb_nolink):
        """Link prediction layers"""
        Wdiag = tf.linalg.tensor_diag(W)
        embW_link = tf.matmul(emb_center_repeat_link, Wdiag)
        embWemb_link = tf.reduce_sum(input_tensor=tf.multiply(
            embW_link, emb_link), axis=-1)

        embW_nolink = tf.matmul(emb_center_repeat_nolink, Wdiag)
        embWemb_nolink = tf.reduce_sum(input_tensor=tf.multiply(
            embW_nolink, emb_nolink), axis=-1)

        embWemb = tf.concat([embWemb_link, embWemb_nolink], axis=0)
        link_probs = tf.sigmoid(embWemb)

        return embWemb, link_probs

    def apply_loss_function_emb(self, global_step):
        pos_logits = tf.reduce_sum(input_tensor=tf.multiply(
            self.emb_center_repeat_ctx, self.emb_context),
                                   axis=-1)
        neg_logits = tf.reduce_sum(input_tensor=tf.multiply(
            self.emb_center_repeat_neg, self.emb_negative),
                                   axis=-1)

        nb_loss = self.neighbor_loss(pos_logits, neg_logits)
        lp_loss = self.linkpred_loss(self.embWemb_1, self.link_weights)
        coexp_loss = self.coexp_sim_loss(self.emb_center_repeat_link,
                                         self.emb_link, self.coexp_weight)
        func_loss = self.func_sim_loss(self.link_probs_1,
                                       self.func_feature_sim)

        self.emb_loss = 1.0 * nb_loss + 1.0 * lp_loss + 100.0 * coexp_loss + self.lambda_func * func_loss
        self.lr = tf.compat.v1.train.exponential_decay(
            self.learning_rate,
            global_step,
            self.decay_step,
            self.learning_rate_decay_factor,
            staircase=True)
        opt = tf.compat.v1.train.AdamOptimizer(learning_rate=self.lr).minimize(
            self.emb_loss, global_step=global_step)

        return opt

    def apply_loss_function_w(self, global_step):
        func_loss = self.func_sim_loss(self.link_probs_2,
                                       self.func_feature_sim)
        self.lp_loss = self.linkpred_loss(
            self.embWemb_2,
            self.link_weights) + tf.nn.l2_loss(self.W - 1) + self.lambda_func * func_loss
        self.lr = tf.compat.v1.train.exponential_decay(
            self.learning_rate,
            global_step,
            self.decay_step,
            self.learning_rate_decay_factor,
            staircase=True)
        opt = tf.compat.v1.train.AdamOptimizer(learning_rate=self.lr).minimize(
            self.lp_loss, global_step=global_step)
        return opt

    def coexp_sim_loss(self, emb_center, emb_nb, coexp_weight):
        def l2_dis_mat(mat1, mat2):
            sqr_d = tf.reduce_sum(tf.math.square((mat1 - mat2)), 1)
            return sqr_d

        return tf.reduce_mean(
            tf.multiply(l2_dis_mat(emb_center, emb_nb), coexp_weight))

    def neighbor_loss(self, pos_logits, neg_logits):
        labels = tf.concat([self.context_labels, self.negative_labels], axis=0)
        loss = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=labels, logits=tf.concat([pos_logits, neg_logits], axis=0))
        return tf.reduce_mean(input_tensor=loss)

    def func_sim_loss(self, link_probs, func_feature_sim):
        return -tf.reduce_mean(tf.multiply(func_feature_sim, link_probs))

    def linkpred_loss(self, embWemb, link_weights):
        labels = tf.concat([self.link_labels, self.nolink_labels], axis=0)
        weighted_cross_entropy = tf.multiply(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=labels,
                                                    logits=embWemb),
            link_weights)
        link_loss = tf.reduce_mean(input_tensor=weighted_cross_entropy)
        return link_loss

    def simulate_walks(self, cur_iter, instance_to_bag, gene_list):
        all_walks = []
        nd_num_map = {}
        if cur_iter == 0:
            walks = net_util.Walks(self.iii_net, self.p, self.q, True,
                                   instance_to_bag)
        else:
            walks = net_util.Walks(self.iii_net, self.p, self.q, False, None,
                                   self.neighbor_changed, self.alias_edges)
        sim_walks = walks.simulate_walks(self.num_walks, self.walk_length)
        rnd_idx = np.random.permutation(len(sim_walks))
        for i in rnd_idx:
            walk = sim_walks[i]
            for nd in walk:
                if nd in nd_num_map:
                    nd_num_map[nd] += 1
                else:
                    nd_num_map[nd] = 1
        for i in rnd_idx:
            walk = sim_walks[i]
            new_walk = []
            for nd in walk:
                new_walk.append(nd)
            all_walks.append(new_walk)
        self.alias_edges = walks.alias_edges
        return all_walks

    def skip_gram_sample(self, walks):
        sg_pair_dic = {}
        for node in self.iii_net.nodes():
            sg_pair_dic[node] = []

        for i, walk in enumerate(walks):
            nodes = [n for n in walk]
            for pos, node in enumerate(nodes):
                reduced_window = np.random.randint(
                    self.window)  # `b` in the original word2vec code

                # now go over all nodes from the (reduced) window, predicting
                # each one in turn
                start = max(0, pos - self.window + reduced_window)
                for pos2, node2 in enumerate(
                        nodes[start:(pos + self.window + 1 - reduced_window)],
                        start):
                    # don't train on the `node` itself
                    if pos2 != pos:
                        sg_pair_dic[node].append(node2)
        return sg_pair_dic

    def neg_sample(self, neg_table, num_neg):
        counter = 0
        neg_list = []
        while counter < num_neg:
            indexes = np.random.randint(low=0,
                                        high=len(neg_table),
                                        size=num_neg)
            for i in indexes:
                counter += 1
                neg_list.append(neg_table[i])
                if counter >= num_neg:
                    break
        return neg_list

    def train(self, co_exp_net, instance_to_bag, cur_iter, pos_nodes_set,
              non_functional_set, gene_list, iso_list, y_train, y_test,
              tissue_enhanced_iso, func_feature):
        print("train encoder...")

        coexp_mask = []
        for i in range(len(iso_list)):
            if i < len(y_train) - len(y_test):
                if i in pos_nodes_set:
                    coexp_mask.append(1)
                else:
                    coexp_mask.append(0)
            else:
                iso = iso_list[i]
                if iso in tissue_enhanced_iso:
                    coexp_mask.append(1)
                else:
                    coexp_mask.append(0)
        coexp_mask = np.array(coexp_mask)

        co_exp_array = co_exp_net.toarray()

        cur_iter_assign = self.cur_iter.assign(cur_iter)
        self.sess.run([cur_iter_assign])

        all_walks = self.simulate_walks(cur_iter, instance_to_bag, gene_list)

        sg_pair_dic = self.skip_gram_sample(all_walks)
        n_table = net_util.create_neg_table(self.ppi_net, non_functional_set,
                                            gene_list)
        nodes_with_edges = []
        for node in self.iii_net.nodes():
            if self.iii_net.degree(node) > 0:
                nodes_with_edges.append(node)

        degree = net_util.get_degree(cur_iter, nodes_with_edges, pos_nodes_set,
                                     self.iii_net)

        if cur_iter == 0:
            epochs = self.first_epochs
        else:
            epochs = self.epochs
        for epoch in range(epochs):
            emb_loss_total = 0
            lp_loss_total = 0
            batch_size = 512

            batch_indexes = []
            random_indexes = np.random.permutation(nodes_with_edges)
            for i in range(0, len(nodes_with_edges), batch_size):
                batch_indexes.append(random_indexes[i:i + batch_size])

            if epoch == 0:
                W = np.zeros(self.emb_dim)
            print("# of batches:", len(batch_indexes))

            center_repeat_ind_link_map = {}
            center_repeat_ind_nolink_map = {}
            link_indexes_map = {}
            nolink_indexes_map = {}
            center_repeat_ind_ctx_map = {}
            center_repeat_ind_neg_map = {}
            context_indexes_map = {}
            negative_indexes_map = {}
            func_feature_sim_map = {}

            for i in range(len(batch_indexes)):
                #Sampling interacting pairs and neighbor node pairs for each batch
                center_repeat_ind_link = []
                center_repeat_ind_nolink = []
                link_indexes = []
                nolink_indexes = []
                center_repeat_ind_ctx = []
                center_repeat_ind_neg = []
                context_indexes = []
                negative_indexes = []
                for idx in batch_indexes[i]:
                    # Sample interacting pairs
                    link_sample = []
                    nb = np.array(list(self.iii_net.neighbors(idx)))
                    nb_idx = np.random.permutation(len(nb))
                    link_sample = list(nb[nb_idx[0:self.positive]])
                    link_indexes.extend(link_sample)
                    center_repeat_ind_link.extend([idx] * len(link_sample))
                    # Sample node pairs without interaction by negative sampling
                    nolink_sample = self.neg_sample(
                        n_table, self.negative * degree[idx])
                    nolink_indexes.extend(nolink_sample)
                    center_repeat_ind_nolink.extend([idx] * len(nolink_sample))

                    # Sample neighbor nodes by random walk
                    ctx_smaple = []
                    context = np.array(sg_pair_dic[idx])
                    ctx_idx = np.random.permutation(len(context))
                    ctx_smaple = list(context[ctx_idx[0:self.positive]])
                    context_indexes.extend(ctx_smaple)
                    center_repeat_ind_ctx.extend([idx] * len(ctx_smaple))
                    # Sample non neighbor nodes by negative sampling
                    nonctx_sample = self.neg_sample(
                        n_table, self.negative * degree[idx])
                    negative_indexes.extend(nonctx_sample)
                    center_repeat_ind_neg.extend([idx] * len(nonctx_sample))

                center_repeat_ind_link_map[i] = center_repeat_ind_link
                center_repeat_ind_nolink_map[i] = center_repeat_ind_nolink
                link_indexes_map[i] = link_indexes
                nolink_indexes_map[i] = nolink_indexes
                center_repeat_ind_ctx_map[i] = center_repeat_ind_ctx
                center_repeat_ind_neg_map[i] = center_repeat_ind_neg
                context_indexes_map[i] = context_indexes
                negative_indexes_map[i] = negative_indexes

                b = 0.01
                func_feature_sim_map[i] = np.hstack(
                    (b - np.sum(np.square(
                        (func_feature[center_repeat_ind_link] -
                         func_feature[link_indexes])), axis=1),
                     b - np.sum(np.square(
                         (func_feature[center_repeat_ind_nolink] -
                          func_feature[nolink_indexes])), axis=1)))

            for i in range(len(batch_indexes)):
                # Update embeddings
                center_repeat_ind_link = center_repeat_ind_link_map[i]
                center_repeat_ind_nolink = center_repeat_ind_nolink_map[i]
                link_indexes = link_indexes_map[i]
                nolink_indexes = nolink_indexes_map[i]
                center_repeat_ind_ctx = center_repeat_ind_ctx_map[i]
                center_repeat_ind_neg = center_repeat_ind_neg_map[i]
                context_indexes = context_indexes_map[i]
                negative_indexes = negative_indexes_map[i]
                coexp_weight = co_exp_array[center_repeat_ind_link,
                                            link_indexes]
                pos_nodes_left = coexp_mask[center_repeat_ind_link]
                pos_nodes_right = coexp_mask[link_indexes]
                coexp_weight = np.multiply(
                    coexp_weight, np.multiply(pos_nodes_left, pos_nodes_right))

                link_weights = self.sess.run(self.link_probs_1,
                                             feed_dict={
                                                 self.center_repeat_link:
                                                 center_repeat_ind_link,
                                                 self.link_dst_input:
                                                 link_indexes,
                                                 self.center_repeat_nolink:
                                                 center_repeat_ind_nolink,
                                                 self.nolink_dst_input:
                                                 nolink_indexes,
                                                 self.W_const: W
                                             })

                _, emb_loss = self.sess.run(
                    [self.train_op_emb, self.emb_loss],
                    feed_dict={
                        self.center_repeat_link: center_repeat_ind_link,
                        self.link_dst_input: link_indexes,
                        self.center_repeat_nolink: center_repeat_ind_nolink,
                        self.nolink_dst_input: nolink_indexes,
                        self.link_labels: np.ones(len(link_indexes)),
                        self.nolink_labels: np.zeros(len(nolink_indexes)),
                        self.center_repeat_ctx: center_repeat_ind_ctx,
                        self.context_input: context_indexes,
                        self.center_repeat_neg: center_repeat_ind_neg,
                        self.negative_input: negative_indexes,
                        self.context_labels: np.ones(len(context_indexes)),
                        self.negative_labels: np.zeros(len(negative_indexes)),
                        self.W_const: W,
                        self.link_weights: link_weights,
                        self.coexp_weight: coexp_weight,
                        self.func_feature_sim: func_feature_sim_map[i]
                    })
                emb_loss_total += emb_loss

            for i in range(len(batch_indexes)):
                # Update W
                center_repeat_ind_link = center_repeat_ind_link_map[i]
                center_repeat_ind_nolink = center_repeat_ind_nolink_map[i]
                link_indexes = link_indexes_map[i]
                nolink_indexes = nolink_indexes_map[i]

                emb_center_repeat_link, emb_link, emb_center_repeat_nolink, emb_nolink = self.sess.run(
                    [
                        self.emb_center_repeat_link, self.emb_link,
                        self.emb_center_repeat_nolink, self.emb_nolink
                    ],
                    feed_dict={
                        self.center_repeat_link: center_repeat_ind_link,
                        self.link_dst_input: link_indexes,
                        self.center_repeat_nolink: center_repeat_ind_nolink,
                        self.nolink_dst_input: nolink_indexes
                    })

                link_weights = self.sess.run(self.link_probs_1,
                                             feed_dict={
                                                 self.center_repeat_link:
                                                 center_repeat_ind_link,
                                                 self.link_dst_input:
                                                 link_indexes,
                                                 self.center_repeat_nolink:
                                                 center_repeat_ind_nolink,
                                                 self.nolink_dst_input:
                                                 nolink_indexes,
                                                 self.W_const: W
                                             })

                _, lp_loss = self.sess.run(
                    [self.train_op_w, self.lp_loss],
                    feed_dict={
                        self.emb_center_repeat_link_const:
                        emb_center_repeat_link,
                        self.emb_link_const: emb_link,
                        self.emb_center_repeat_nolink_const:
                        emb_center_repeat_nolink,
                        self.emb_nolink_const: emb_nolink,
                        self.link_labels: np.ones(len(link_indexes)),
                        self.nolink_labels: np.zeros(len(nolink_indexes)),
                        self.link_weights: link_weights,
                        self.func_feature_sim: func_feature_sim_map[i]
                    })
                lp_loss_total += lp_loss

            W = self.sess.run(self.W)
            print("#train# epoch", epoch, "embedding loss = ", emb_loss_total,
                  "link prediction loss =", lp_loss_total)

        embeddings = self.emb_inference(np.arange(self.N_size))
        self.iii_net, self.neighbor_changed = self.predict_link(
            cur_iter, embeddings, W, gene_list, y_train, y_test)

        return self.iii_net

    def emb_inference(self, indexes):
        batch_size = 512
        embeddings = np.zeros((len(indexes), self.emb_dim), dtype=np.float32)
        for idx in range(0, len(indexes), batch_size):
            indexes_batch = indexes[idx:idx + batch_size]
            embeddings_batch = self.sess.run(
                self.emb_center_repeat_link,
                feed_dict={self.center_repeat_link: indexes_batch})
            embeddings[idx:idx + batch_size] = embeddings_batch
        return embeddings

    def predict_link(self, iter, embeddings, W, gene_list,
                     y_train, y_test):
        G = nx.Graph()
        nodes = np.arange(self.N_size)
        G.add_nodes_from(nodes)
        predicted_edges = []

        nodes_idx = []
        for i in range(self.N_size):
            if self.ppi_net.degree(i) > 0:
                nodes_idx.append(i)

        embeddings = normalize(embeddings, norm='l2')
        predicted_links = embeddings[nodes_idx, :].dot(np.diag(W)).dot(
            np.transpose(embeddings[nodes_idx, :]))

        link_thresh = self.link_thresh
        print("link_thresh", link_thresh)

        r, c = np.where(1.0 / (1.0 + np.exp(-predicted_links)) >= link_thresh)
        for i in range(len(r)):
            predicted_edges.append([nodes_idx[r[i]], nodes_idx[c[i]]])

        for r in range(len(nodes_idx)):
            c = np.argsort(-predicted_links[r, :])
            for i in range(self.min_degree):
                predicted_edges.append([nodes_idx[r], nodes_idx[c[i]]])

        G.add_edges_from(predicted_edges)
        print("Prediciton stats: %d edges in total" % G.number_of_edges())

        neighbor_changed = np.zeros(self.N_size)
        for node in G.nodes():
            if list(G.neighbors(node)) != list(self.iii_net.neighbors(node)):
                neighbor_changed[node] = 1
        print("Links of %d nodes were updated." % np.sum(neighbor_changed))

        return G, neighbor_changed
