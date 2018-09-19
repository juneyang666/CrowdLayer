
import numpy as np
import tensorflow as tf
import keras
from keras import backend as K
from keras.engine.topology import Layer
from keras.layers import merge, Dense, Reshape
from keras.layers import Activation, Dropout, Flatten, Dense, merge, Reshape, Permute, Multiply, Dot,dot, Concatenate, Add


def init_identities(shape, dtype=None):
    out = np.zeros(shape)
    for r in range(shape[2]):
        for i in range(shape[0]):
            out[i,i,r] = 1.0
    return out



class CrowdsClassification(Layer):

    def __init__(self, output_dim, num_annotators, conn_type="MW", **kwargs):
        self.output_dim = output_dim
        self.num_annotators = num_annotators
        self.conn_type = conn_type
        super(CrowdsClassification, self).__init__(**kwargs)

    def build(self, input_shape):
        if self.conn_type == "MW":
            # matrix of weights per annotator
            self.kernel = self.add_weight("CrowdLayer", (self.output_dim, self.output_dim, self.num_annotators),
                                            initializer=init_identities,
                                            trainable=True)
        elif self.conn_type == "VW":
            # vector of weights (one scale per class) per annotator
            self.kernel = self.add_weight("CrowdLayer", (self.output_dim, self.num_annotators),
                                            initializer=keras.initializers.Ones(),
                                            trainable=True)
        elif self.conn_type == "VB":
            # two vectors of weights (one scale and one bias per class) per annotator
            self.kernel = []
            self.kernel.append(self.add_weight("CrowdLayer", (self.output_dim, self.num_annotators),
                                            initializer=keras.initializers.Zeros(),
                                            trainable=True))
        elif self.conn_type == "VW+B":
            # two vectors of weights (one scale and one bias per class) per annotator
            self.kernel = []
            self.kernel.append(self.add_weight("CrowdLayer", (self.output_dim, self.num_annotators),
                                            initializer=keras.initializers.Ones(),
                                            trainable=True))
            self.kernel.append(self.add_weight("CrowdLayer", (self.output_dim, self.num_annotators),
                                            initializer=keras.initializers.Zeros(),
                                            trainable=True))
        elif self.conn_type == "SW":
            # single weight value per annotator
            self.kernel = self.add_weight("CrowdLayer", (self.num_annotators,1),
                                            initializer=keras.initializers.Ones(),
                                            trainable=True)
        else:
            raise Exception("Unknown connection type for CrowdsClassification layer!")

        super(CrowdsClassification, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        if self.conn_type == "MW":
            print(x)
            print(self.kernel)
            res = K.dot(x, self.kernel)
            print(res)
        elif self.conn_type == "VW" or self.conn_type == "VB" or self.conn_type == "VW+B" or self.conn_type == "SW":
            out = []
            for r in range(self.num_annotators):
                if self.conn_type == "VW":
                    out.append(x * self.kernel[:,r])
                elif self.conn_type == "VB":
                    out.append(x + self.kernel[0][:,r])
                elif self.conn_type == "VW+B":
                    out.append(x * self.kernel[0][:,r] + self.kernel[1][:,r])
                elif self.conn_type == "SW":
                    out.append(x * self.kernel[r,0])
            res = tf.stack(out)
            if len(res.shape) == 3:
                res = tf.transpose(res, [1, 2, 0])
            elif len(res.shape) == 4:
                res = tf.transpose(res, [1, 2, 3, 0])
            else:
                raise Exception("Wrong number of dimensions for output")
        else:
            raise Exception("Unknown connection type for CrowdsClassification layer!")

        return res

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim, self.num_annotators)


class CrowdsRegression(Layer):

    def __init__(self, num_annotators, conn_type="B", **kwargs):
        self.num_annotators = num_annotators
        self.conn_type = conn_type
        super(CrowdsRegression, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = []
        if self.conn_type == "S":
            # scale-only parameter
            self.kernel.append(self.add_weight("CrowdLayer", (1, self.num_annotators),
                                  initializer=keras.initializers.Ones(),
                                  trainable=True))
        elif self.conn_type == "B":
            # bias-only parameter
            self.kernel.append(self.add_weight("CrowdLayer", (1, self.num_annotators),
                                  initializer=keras.initializers.Zeros(),
                                  trainable=True))
        elif self.conn_type == "S+B" or self.conn_type == "B+S":
            # scale and bias parameters
            self.kernel.append(self.add_weight("CrowdLayer", (1, self.num_annotators),
                                      initializer=keras.initializers.Ones(),
                                      trainable=True))
            self.kernel.append(self.add_weight("CrowdLayer", (1, self.num_annotators),
                                      initializer=keras.initializers.Zeros(),
                                      trainable=True))
        else:
            raise Exception("Unknown connection type for CrowdsRegression layer!")

        super(CrowdsRegression, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        if self.conn_type == "S":
            #res = K.dot(x, self.kernel[0])
            res = x * self.kernel[0]
        elif self.conn_type == "B":
            res = x + self.kernel[0]
        elif self.conn_type == "S+B":
            #res = K.dot(x, self.kernel[0]) + self.kernel[1]
            res = x * self.kernel[0] + self.kernel[1]
        elif self.conn_type == "B+S":
            res = (x + self.kernel[1]) * self.kernel[0]
        else:
            raise Exception("Unknown connection type for CrowdsClassification layer!")

        return res

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.num_annotators)


class MaskedMultiCrossEntropy(object):

    def loss(self, y_true, y_pred):
        print('y_true: ', y_true)
        print('y_pred: ', y_pred)
        vec = tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y_true, dim=1)
        mask = tf.equal(y_true[:,0,:], -1)
        zer = tf.zeros_like(vec)
        loss = tf.where(mask, x=zer, y=vec)
        return loss


# convenience l2_norm function
def l2_norm(x, axis=None):
    """
    takes an input tensor and returns the l2 norm along specified axis
    """

    square_sum = K.sum(K.square(x), axis=axis, keepdims=True)
    norm = K.sqrt(K.maximum(square_sum, K.epsilon()))

    return norm


def pairwise_cosine_sim(A, B):
    """
    A [batch x d] tensor of n rows with d dimensions
    B [batch x d x m] tensor of n rows with d dimensions

    returns:
    D [batch x m] tensor of cosine similarity scores between each point i<n, j<m
    """

    print('cosine sim A: ', A)
    print('cosine sim B: ', B)

    A = K.reshape(A, [-1, 1, 8])
    A_mag = l2_norm(A, axis=2)
    B_mag = l2_norm(B, axis=1)
    num = K.batch_dot(A, B)
    den = (A_mag * B_mag)
    dist_mat = num / den
    dist_mat = K.squeeze(dist_mat, axis=1)

    print('dist_mat: ', dist_mat)

    return dist_mat


class MaskedMultiCrossEntropyCosSim(object):

    def __init__(self, y_pred_clean):
        self.y_pred_clean = y_pred_clean

    def loss(self, y_true, y_pred):
        vec = tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y_true, dim=1)
        mask = tf.equal(y_true[:,0,:], -1)
        cos_sim = pairwise_cosine_sim(self.y_pred_clean, y_pred)
        cos_sim = 1 - tf.divide(cos_sim, tf.reshape(tf.reduce_sum(cos_sim, axis=1), [-1, 1]))
        zer = tf.zeros_like(vec)
        cos_sim = tf.where(mask, x=zer, y=cos_sim)
        loss = tf.where(mask, x=zer, y=vec)
        loss = tf.add(loss, cos_sim)
        return loss

class MaskedMultiCrossEntropyBaseChannel(object):

    def __init__(self, y_pred_clean):
        self.y_pred_clean = y_pred_clean
        self.y_pred_broad = tf.reshape(tf.tile(y_pred_clean, [1, 59]), [-1, 8, 59])

    def loss(self, y_true, y_pred):

        vec = tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y_true, dim=1)
        vec_base_channel = tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=self.y_pred_broad, dim=1)
        mask = tf.equal(y_true[:,0,:], -1)

        zer = tf.zeros_like(vec)
        loss_base_channel = tf.where(mask, x=zer, y=vec_base_channel)
        loss = tf.where(mask, x=zer, y=vec)
        loss = tf.add(loss, loss_base_channel)
        return loss


class MaskedMultiCrossEntropyBaseChannelConst(object):

    def __init__(self, y_pred_clean, const):
        self.y_pred_clean = y_pred_clean
        self.y_pred_broad = tf.reshape(tf.tile(y_pred_clean, [1, 59]), [-1, 8, 59])
        self.const = const

    def loss(self, y_true, y_pred):

        vec = tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y_true, dim=1)
        vec_base_channel = tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=self.y_pred_broad, dim=1)
        mask = tf.equal(y_true[:,0,:], -1)

        zer = tf.zeros_like(vec)
        loss_base_channel = self.const * tf.where(mask, x=zer, y=vec_base_channel)
        loss = tf.where(mask, x=zer, y=vec)
        loss = tf.add(loss, loss_base_channel)
        return loss


class MaskedMultiCrossEntropyCurriculum(object):

    def __init__(self, y_pred_clean, const):
        self.y_pred_clean = y_pred_clean
        self.y_pred_broad = tf.reshape(tf.tile(y_pred_clean, [1, 59]), [-1, 8, 59])
        self.const = const

    def loss(self, y_true, y_pred):

        vec = tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y_true, dim=1)
        vec_base_channel = tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=self.y_pred_broad, dim=1)
        mask = tf.equal(y_true[:,0,:], -1)

        zer = tf.zeros_like(vec)
        loss_base_channel = self.const * tf.where(mask, x=zer, y=vec_base_channel)
        loss = tf.where(mask, x=zer, y=vec)
        loss = tf.add(loss, loss_base_channel)
        return loss


class MaskedMultiCrossEntropyCurriculumChannelMatrix(object):

    def __init__(self, model, a, b):
        self.t = tf.transpose(model.get_weights()[-1], perm=[2, 0, 1])
        self.a = a
        self.b = b

    def loss(self, y_true, y_pred):

        vec = tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y_true, dim=1)
        # vec_base_channel = tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=self.y_pred_broad, dim=1)
        trace = tf.trace(self.t)

        vec = vec - trace * self.b
        mask = tf.equal(y_true[:,0,:], -1)
        zer = tf.zeros_like(vec)
        loss = tf.where(mask, x=zer, y=vec)
        return loss


class MaskedMultiMSE(object):

    def loss(self, y_true, y_pred):
        vec = K.square(y_pred - y_true)
        mask = tf.equal(y_true[:,:], 999999999)
        zer = tf.zeros_like(vec)
        loss = tf.where(mask, x=zer, y=vec)
        return loss


class MaskedMultiSequenceCrossEntropy(object):

    def __init__(self, num_classes):
        self.num_classes = num_classes

    def loss(self, y_true, y_pred):
        mask_missings = tf.equal(y_true, -1)
        mask_padding = tf.equal(y_true, 0)

        # convert targets to one-hot enconding and transpose
        y_true = tf.transpose(tf.one_hot(tf.cast(y_true, tf.int32), self.num_classes, axis=-1), [0,1,3,2])

        # masked cross-entropy
        vec = tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y_true, dim=2)
        zer = tf.zeros_like(vec)
        vec = tf.where(mask_missings, x=zer, y=vec)
        vec = tf.where(mask_padding, x=zer, y=vec)
        loss = tf.reduce_mean(vec, axis=-1)
        return loss


class CrowdsAggregationCategoricalCrossEntropy(object):

    def __init__(self, num_classes, num_annotators, pi_prior=0.01):
        self.num_classes = num_classes
        self.num_annotators = num_annotators
        self.pi_prior = pi_prior

        # initialize pi_est (annotators' estimated confusion matrices) wit identities
        self.pi_est = np.zeros((self.num_classes,self.num_classes,self.num_annotators), dtype=np.float32)
        for r in xrange(self.num_annotators):
            self.pi_est[:,:,r] = np.eye(self.num_classes) + self.pi_prior
            self.pi_est[:,:,r] /= np.sum(self.pi_est[:,:,r], axis=1)

        self.init_suff_stats()

    def init_suff_stats(self):
        # initialize suff stats for M-step
        self.suff_stats = self.pi_prior * tf.ones((self.num_annotators,self.num_classes,self.num_classes))

    def loss_fc(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)

        #y_pred += 0.01
        #y_pred /= tf.reduce_sum(y_pred, reduction_indices=len(y_pred.get_shape()) - 1, keep_dims=True)

        #y_pred = tf.where(tf.less(y_pred, 0.001),
        #                        #0.01 * tf.ones_like(y_pred),
        #                        0.001 + y_pred,
        #                        y_pred)
        #y_pred += 0.01 # y_pred cannot be zero!
        eps = 1e-3
        #y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)
        y_pred = tf.clip_by_value(y_pred, eps, 9999999999)


        # E-step
        adjustment_factor = tf.ones_like(y_pred)
        for r in range(self.num_annotators):
            adj = tf.where(tf.equal(y_true[:,r], -1),
                                tf.ones_like(y_pred),
                                tf.gather(tf.transpose(self.pi_est[:,:,r]), y_true[:,r]))
            adjustment_factor = tf.multiply(adjustment_factor, adj)

        res = tf.multiply(adjustment_factor, y_pred)
        y_agg = res / tf.expand_dims(tf.reduce_sum(res, axis=1), 1)

        loss = -tf.reduce_sum(y_agg * tf.log(y_pred), reduction_indices=[1])

        # update suff stats
        upd_suff_stats = []
        for r in range(self.num_annotators):
            #print r
            suff_stats = []
            normalizer = tf.zeros_like(y_pred)
            for c in range(self.num_classes):
                suff_stats.append(tf.reduce_sum(tf.where(tf.equal(y_true[:,r], c),
                                    y_agg,
                                    tf.zeros_like(y_pred)), axis=0))
            upd_suff_stats.append(suff_stats)
        upd_suff_stats = tf.stack(upd_suff_stats)
        self.suff_stats += upd_suff_stats

        return loss

    def m_step(self):
        #print "M-step"
        self.pi_est = tf.transpose(self.suff_stats / tf.expand_dims(tf.reduce_sum(self.suff_stats, axis=2), 2), [1, 2, 0])

        return self.pi_est


class CrowdsAggregationBinaryCrossEntropy(object):

    def __init__(self, num_annotators, pi_prior=0.01, alpha=None, beta=None, update_freq=1):
        self.num_annotators = num_annotators
        self.pi_prior = pi_prior
        self.alpha = alpha
        self.beta = beta
        self.update_freq = update_freq

        # initialize alpha and beta (annotators' estimated sensitivity and specificity)
        if self.alpha == None:
            print("initializing alpha with unit...")
            self.alpha = 0.99*np.ones((self.num_annotators,1), dtype=np.float32)
        if self.beta == None:
            self.beta = 0.99*np.ones((self.num_annotators,1), dtype=np.float32)
        self.count = tf.ones(1)

        self.suff_stats_alpha = [self.pi_prior for r in xrange(self.num_annotators)]
        self.suff_stats_beta = [self.pi_prior for r in xrange(self.num_annotators)]
        self.suff_stats_alpha_norm = [self.pi_prior for r in xrange(self.num_annotators)]
        self.suff_stats_beta_norm = [self.pi_prior for r in xrange(self.num_annotators)]

    def init_suff_stats(self):
        # initialize suff stats for M-step
        pass

    def loss_fc(self, y_true, y_pred):
        #y_true = tf.cast(y_true, tf.int32)

        #y_pred += 0.01
        #y_pred /= tf.reduce_sum(y_pred, reduction_indices=len(y_pred.get_shape()) - 1, keep_dims=True)

        #y_pred = tf.where(tf.less(y_pred, 0.001),
        #                        #0.01 * tf.ones_like(y_pred),
        #                        0.001 + y_pred,
        #                        y_pred)
        #y_pred += 0.01 # y_pred cannot be zero!
        eps = 1e-3
        y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)
        #y_pred = tf.clip_by_value(y_pred, eps, 9999999999)

        p = y_pred[:,1]
        self.count += 1
        self.count = tf.Print(self.count, [self.count])
        #self.count += 1

        if False:
            print("M-step...")
            self.alpha = []
            self.beta = []
            for r in xrange(self.num_annotators):
                self.alpha.append(self.suff_stats_alpha[r] / self.suff_stats_alpha_norm[r])
                self.beta.append(self.suff_stats_beta[r] / self.suff_stats_beta_norm[r])
            self.count = 0
            self.suff_stats_alpha = [self.pi_prior for r in range(self.num_annotators)]
            self.suff_stats_beta = [self.pi_prior for r in range(self.num_annotators)]
            self.suff_stats_alpha_norm = [self.pi_prior for r in range(self.num_annotators)]
            self.suff_stats_beta_norm = [self.pi_prior for r in range(self.num_annotators)]
            self.alpha = tf.Print(self.alpha, [self.alpha])


        # E-step
        a = tf.ones_like(p)
        b = tf.ones_like(p)
        for r in xrange(self.num_annotators):
            a = a * tf.where(tf.equal(y_true[:,r], 1), self.alpha[r]*tf.ones_like(p), tf.ones_like(p))
            b = b * tf.where(tf.equal(y_true[:,r], 1), (1.0-self.beta[r])*tf.ones_like(p), tf.ones_like(p))
            a = a * tf.where(tf.equal(y_true[:,r], 0), (1.0-self.alpha[r])*tf.ones_like(p), tf.ones_like(p))
            b = b * tf.where(tf.equal(y_true[:,r], 0), self.beta[r]*tf.ones_like(p), tf.ones_like(p))

        mu = (a*p) / (a*p + b*(1.0-p))
        #mu = tf.Print(mu, [mu])
        loss = - (mu * tf.log(y_pred[:,1]) + (1.0-mu) * tf.log(y_pred[:,0]))

        # update suff stats
        for r in xrange(self.num_annotators):
            self.suff_stats_alpha[r] += tf.reduce_sum(tf.where(tf.equal(y_true[:,r], 1), mu, tf.zeros_like(p)))
            self.suff_stats_beta[r] += tf.reduce_sum(tf.where(tf.equal(y_true[:,r], 0), (1.0-mu), tf.zeros_like(p)))
            self.suff_stats_alpha_norm[r] += tf.reduce_sum(tf.where(tf.equal(y_true[:,r], -1), tf.zeros_like(p), mu))
            self.suff_stats_beta_norm[r] += tf.reduce_sum(tf.where(tf.equal(y_true[:,r], -1), tf.zeros_like(p), (1.0-mu)))

        return loss

    def m_step(self):
        print(dir(self))
        print("debug:", self.count.eval())
        #print "M-step"
        #self.count += 1
        #print "increment", self.count
        #if self.count >= self.update_freq:



        return (self.alpha, self.beta)


class CrowdsAggregationCallback(keras.callbacks.Callback):

    def __init__(self, loss):
        self.loss = loss

    def on_epoch_begin(self, epoch, logs=None):
        self.loss.init_suff_stats()

    def on_epoch_end(self, epoch, logs=None):
        # run M-step
        self.model.pi = self.loss.m_step()


APRIOR_NOISE=0.46
N_CLASSES=8
bias_weights = (
    np.array([np.array([np.log(1. - APRIOR_NOISE)
                        if i == j else
                        np.log(APRIOR_NOISE / (N_CLASSES - 1.))
                        for j in range(N_CLASSES)]) for i in
              range(N_CLASSES)])
    + 0.01 * np.random.random((N_CLASSES, N_CLASSES)))
# bias_weights = np.repeat(bias_weights, N_ANNOT, axis=1)


def init_bias(shape, dtype=None):
    out = np.zeros(shape)
    for r in range(shape[2]):
        for i in range(shape[0]):
            out[:,:,r] = bias_weights
    return out


def init_weight(shape, dtype=None):
    W = 0
    out = W*(np.random.random(shape) - 0.5)
    return out


class CrowdsClassificationSModel(Layer):

    def __init__(self, output_dim, num_annotators, conn_type="MW", **kwargs):
        self.output_dim = output_dim
        self.num_annotators = num_annotators
        self.conn_type = conn_type
        super(CrowdsClassificationSModel, self).__init__(**kwargs)

    def build(self, input_shape):
        if self.conn_type == "MW":
            # matrix of weights per annotator
            self.kernel = []
            self.kernel.append(self.add_weight("CrowdLayer", (self.output_dim, self.output_dim, self.num_annotators),
                                initializer=init_bias,
                                trainable=True))
        else:
            raise Exception("Unknown connection type for CrowdsClassification layer!")

        super(CrowdsClassificationSModel, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs):
        if self.conn_type == "MW":
            print('inputs: ', inputs)
            print('weights: ', self.kernel)
            channel_output_l = []
            for r in range(self.num_annotators):
                channel_matrix_w = self.kernel[0][:,:,r]
                channel_matrix_w_l = []
                for c in range(self.output_dim):
                    channel_matrix_w_c = K.softmax(channel_matrix_w[c,:])
                    channel_matrix_w_l.append(channel_matrix_w_c)
                channel_matrix_w = tf.stack(channel_matrix_w_l)
                channel_output_w = K.dot(inputs[1], channel_matrix_w)
                channel_output_w = K.dropout(channel_output_w, 0.4)
                channel_output_l.append(channel_output_w)
            channel_output = tf.stack(channel_output_l)
            channel_output = K.permute_dimensions(channel_output, (1,2,0))

#             res = K.batch_dot(inputs[1], channel_matrix)
        else:
            raise Exception("Unknown connection type for CrowdsClassification layer!")

        return channel_output

    def compute_output_shape(self, input_shape):
        return (input_shape[1][0], self.output_dim, self.num_annotators)



class CrowdsClassificationSModelChannelMatrix(Layer):

    def __init__(self, output_dim, num_annotators, conn_type="MW", **kwargs):
        self.output_dim = output_dim
        self.num_annotators = num_annotators
        self.conn_type = conn_type
        super(CrowdsClassificationSModelChannelMatrix, self).__init__(**kwargs)

    def build(self, input_shape):
        if self.conn_type == "MW":
            # matrix of weights per annotator
            self.kernel = []
            self.kernel.append(self.add_weight("CrowdLayer", (self.output_dim, self.output_dim, self.num_annotators),
                                initializer=init_bias,
                                trainable=True))
        else:
            raise Exception("Unknown connection type for CrowdsClassification layer!")

        super(CrowdsClassificationSModelChannelMatrix, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs):
        if self.conn_type == "MW":
            print('inputs: ', inputs)
            print('weights: ', self.kernel)
            channel_output_l = []
            channel_matrix_l = []
            for r in range(self.num_annotators):
                channel_matrix_w = self.kernel[0][:,:,r]
                channel_matrix_w_l = []
                for c in range(self.output_dim):
                    channel_matrix_w_c = K.softmax(channel_matrix_w[c,:])
                    channel_matrix_w_l.append(channel_matrix_w_c)
                channel_matrix_w = tf.stack(channel_matrix_w_l)
                channel_matrix_l.append(channel_matrix_w)
                channel_output_w = K.dot(inputs[1], channel_matrix_w)
                channel_output_w = K.dropout(channel_output_w, 0.4)
                channel_output_l.append(channel_output_w)
            channel_matrix = tf.stack(channel_matrix_l)
            channel_output = tf.stack(channel_output_l)
            channel_output = K.permute_dimensions(channel_output, (1,2,0))
            self.channel_matrix = channel_matrix

#             res = K.batch_dot(inputs[1], channel_matrix)
        else:
            raise Exception("Unknown connection type for CrowdsClassification layer!")

        return channel_output

    def get_channel_matrix(self):
        return self.channel_matrix

    def compute_output_shape(self, input_shape):
        return (input_shape[1][0], self.output_dim, self.num_annotators)


# Complex Model
class CrowdsClassificationCModel(Layer):

    def __init__(self, output_dim, num_annotators, conn_type="MW", **kwargs):
        self.output_dim = output_dim
        self.num_annotators = num_annotators
        self.conn_type = conn_type
        super(CrowdsClassificationCModel, self).__init__(**kwargs)

    def build(self, input_shape):
        if self.conn_type == "MW":
            # matrix of weights per annotator
            self.kernel = []
            self.kernel.append(self.add_weight("CrowdLayer", (128, self.output_dim*self.output_dim, self.num_annotators),
                                initializer=init_weight,
                                trainable=True))
            self.kernel.append(self.add_weight("CrowdLayer", (self.output_dim, self.output_dim, self.num_annotators),
                                initializer=init_bias,
                                trainable=True))
        else:
            raise Exception("Unknown connection type for CrowdsClassification layer!")

        super(CrowdsClassificationCModel, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs):
        if self.conn_type == "MW":
            print('inputs: ', inputs)
            print('weights: ', self.kernel)
            channel_output_l = []
            for r in range(self.num_annotators):
                dot_p = K.dot(inputs[0], self.kernel[0][:,:,r])
                dot_p = K.reshape(dot_p, (-1, 8, 8))
                channel_matrix_w = dot_p + self.kernel[1][:,:,r]
#                 channel_matrix_w = self.kernel[0][:,:,r]
                channel_matrix_w_l = []
                for c in range(self.output_dim):
                    channel_matrix_w_c = K.softmax(channel_matrix_w[:,c,:])
                    channel_matrix_w_l.append(channel_matrix_w_c)
                channel_matrix_w = tf.stack(channel_matrix_w_l)
                channel_matrix_w = K.permute_dimensions(channel_matrix_w, (1,0,2))
                channel_output_w = K.batch_dot(inputs[1], channel_matrix_w)
                channel_output_w = K.dropout(channel_output_w, 0.5)
                channel_output_l.append(channel_output_w)

            channel_output = tf.stack(channel_output_l)
            channel_output = K.permute_dimensions(channel_output, (1,2,0))
        else:
            raise Exception("Unknown connection type for CrowdsClassification layer!")

        return channel_output

    def compute_output_shape(self, input_shape):
        return (input_shape[1][0], self.output_dim, self.num_annotators)


class CrowdsClassificationCModelSingleWeight(Layer):

    def __init__(self, output_dim, num_annotators, conn_type="MW", **kwargs):
        self.output_dim = output_dim
        self.num_annotators = num_annotators
        self.conn_type = conn_type
        super(CrowdsClassificationCModelSingleWeight, self).__init__(**kwargs)

    def build(self, input_shape):
        if self.conn_type == "MW":
            # matrix of weights per annotator
            self.kernel = []
            self.kernel.append(self.add_weight("CrowdLayer", (64, self.output_dim*self.output_dim),
                                initializer=init_weight,
                                trainable=True))
            self.kernel.append(self.add_weight("CrowdLayer", (self.output_dim, self.output_dim, self.num_annotators),
                                initializer=init_bias,
                                trainable=True))
        else:
            raise Exception("Unknown connection type for CrowdsClassification layer!")

        super(CrowdsClassificationCModelSingleWeight, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs):
        if self.conn_type == "MW":
            print('inputs: ', inputs)
            print('weights: ', self.kernel)
            channel_output_l = []
            for r in range(self.num_annotators):
                dot_p = K.dot(inputs[0], self.kernel[0][:,:])
                dot_p = K.reshape(dot_p, (-1, 8, 8))
                channel_matrix_w = dot_p + self.kernel[1][:,:,r]
#                 channel_matrix_w = self.kernel[0][:,:,r]
                channel_matrix_w_l = []
                for c in range(self.output_dim):
                    channel_matrix_w_c = K.softmax(channel_matrix_w[:,c,:])
                    channel_matrix_w_l.append(channel_matrix_w_c)
                channel_matrix_w = tf.stack(channel_matrix_w_l)
                channel_matrix_w = K.permute_dimensions(channel_matrix_w, (1,0,2))
                channel_output_w = K.batch_dot(inputs[1], channel_matrix_w)
                channel_output_w = K.dropout(channel_output_w, 0.3)
                channel_output_l.append(channel_output_w)

            channel_output = tf.stack(channel_output_l)
            channel_output = K.permute_dimensions(channel_output, (1,2,0))

#             res = K.batch_dot(inputs[1], channel_matrix)
        else:
            raise Exception("Unknown connection type for CrowdsClassification layer!")

        return channel_output

    def compute_output_shape(self, input_shape):
        return (input_shape[1][0], self.output_dim, self.num_annotators)

