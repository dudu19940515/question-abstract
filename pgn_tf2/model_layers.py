# -*- coding:utf-8 -*-
# Created by LuoJie at 11/23/19

from utils.config import save_wv_model_path, vocab_path
import tensorflow as tf
from utils.gpu_utils import config_gpu
from tensorflow.keras.models import Model
import tensorflow as tf
from utils.wv_loader import load_embedding_matrix, Vocab



class Encoder(tf.keras.Model):
    def __init__(self,  embedding_matrix,enc_units, batch_sz):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        vocab_size, embedding_dim =  embedding_matrix.shape
        self.embedding =tf.keras.layers.Embedding(vocab_size, embedding_dim,
                                                  trainable = True)

        self.lstm = tf.keras.layers.LSTM(self.enc_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.bilstm = tf.keras.layers.Bidirectional(self.lstm)
        #self.W_h = tf.keras.layers.Dense(self.enc_units*2)
        #默认是concat
    def call(self, x):
        # code
        x = self.embedding(x)
        #output shape = batch, max_enc_len, 2*hidden_dim
        output, forward_state,f_cell_state, backward_state,b_cell_state = self.bilstm(x)
        #print("cell shape is {}".format(cell_state.shape))
        hidden_state = tf.keras.layers.concatenate([forward_state, backward_state], axis=-1)
        cell_state = tf.keras.layers.concatenate([f_cell_state, b_cell_state], axis=-1)
        state = [hidden_state, cell_state]
        #enc_features shape == (batch_size*max_len, 2*enc_units)
        #enc_features = tf.reshape(output, (-1,2*self.enc_units))
        #enc_features = self.W_h(enc_features)
        #output= self.bilstm(x)

        return output, state

    # def initialize_hidden_state(self):
    #     return tf.zeros(shape = [self.batch_sz, self.enc_units])
class ReduceState(tf.keras.Model):
    def __init__(self,enc_units,):
        super(ReduceState, self).__init__()
        self.reduce_h = tf.keras.layers.Dense(enc_units, activation = tf.keras.activations.relu)
            #nn.Linear(config.hidden_dim * 2, config.hidden_dim)
        self.reduce_c = tf.keras.layers.Dense(enc_units,activation =tf.keras.activations.relu)
            #nn.Linear(config.hidden_dim * 2, config.hidden_dim)

    def call(self, state):
        # h, c dim = [ batch, 2*enc_units]
        h, c = state

        hidden_reduced_h = self.reduce_h(h)

        hidden_reduced_c = self.reduce_c(c)

        # h, c dim = [1, batch, hidden_dim]
        return hidden_reduced_h, hidden_reduced_c

def masked_attention(enc_padding_mask, score):
    """Take softmax of e then apply enc_padding_mask and re-normalize"""
    #attn_dist shape = batch_size, max_len,1
    attn_dist = tf.squeeze(score, axis=2)
    attn_dist = tf.nn.softmax(attn_dist) #shape = (batch_size, max_len)
    mask = tf.cast(enc_padding_mask, dtype=attn_dist.dtype)
    attn_dist *= mask  # apply mask
    masked_sums = tf.reduce_sum(attn_dist, axis=1)  # shape (batch_size)
    attn_dist = attn_dist / tf.reshape(masked_sums, [-1, 1])  # re-normalize
    attn_dist = tf.expand_dims(attn_dist, axis=2)
    return attn_dist #shape batch_size, max_len,1


class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W_s = tf.keras.layers.Dense(2*units)
        self.W_h = tf.keras.layers.Dense(2*units)
        self.W_c = tf.keras.layers.Dense(2*units)
        self.V = tf.keras.layers.Dense(1)

    def __call__(self, dec_hidden, enc_output, enc_pad_mask, use_coverage=False, prev_coverage=None):
        """
         calculate attention and coverage from dec_hidden enc_output and prev_coverage
         one dec_hidden(word) by one dec_hidden
         dec_hidden or query is [batch_sz, enc_unit], enc_output or values is [batch_sz, max_train_x, enc_units],
         prev_coverage is [batch_sz, max_len_x, 1]
         dec_hidden is initialized as enc_hidden, prev_coverage is initialized as None
         output context_vector [batch_sz, enc_units] attention_weights & coverage [batch_sz, max_len_x, 1]
         """
        # hidden shape == (batch_size, hidden size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden size)
        # we are doing this to perform addition to calculate the score
        #dec_hidden  shape = (batch_size, 2*dec_units)
        hidden_with_time_axis = tf.expand_dims(dec_hidden, 1)

        if use_coverage and prev_coverage is not None:
            # Multiply coverage vector by w_c to get coverage_features.
            # self.W_s(values) [batch_sz, max_len, units] self.W_h(hidden_with_time_axis) [batch_sz, 1, 2*units]
            # self.W_c(prev_coverage) [batch_sz, max_len, 2*units]  score [batch_sz, max_len, 1]
            score = self.V(tf.nn.tanh(self.W_s(enc_output) + self.W_h(hidden_with_time_axis) + self.W_c(prev_coverage)))
            # attention_weights shape (batch_size, max_len, 1)

            # attention_weights shape== (batch_size, max_length, 1)
            #mask = tf.cast(enc_pad_mask, dtype=score.dtype)

            ##attention_weights = tf.nn.softmax(score, axis=1)
            attention_weights = masked_attention(enc_pad_mask, score)
            coverage = attention_weights + prev_coverage
        else:
            # score shape == (batch_size, max_length, 1)
            # we get 1 at the last axis because we are applying score to self.V
            # the shape of the tensor before applying self.V is (batch_size, max_length, units)
            # 计算注意力权重值
            score = self.V(tf.nn.tanh(
                self.W_s(enc_output) + self.W_h(hidden_with_time_axis)))

            # mask = tf.cast(enc_pad_mask, dtype=score.dtype)
            # masked_score = tf.squeeze(score, axis=-1) * mask
            # masked_score = tf.expand_dims(masked_score, axis=2)
            # attention_weights = tf.nn.softmax(masked_score, axis=1

            attention_weights = masked_attention(enc_pad_mask, score)
            if use_coverage:
                coverage = attention_weights
            else:
                coverage = []

        # # 使用注意力权重*编码器输出作为返回值，将来会作为解码器的输入
        # context_vector shape after sum == (batch_size, 2*hidden_dim)
        context_vector = attention_weights * enc_output
        context_vector = tf.reduce_sum(context_vector, axis=1)
        attention_weights = tf.squeeze(attention_weights, -1) #转换为 batch_size, max_len

        return context_vector, attention_weights, coverage
class Decoder(tf.keras.Model):
    def __init__(self, embedding_matrix, dec_units, attn_units, batch_sz, ):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        vocab_size, embedding_dim = embedding_matrix.shape
        self.attention = BahdanauAttention(attn_units)
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, weights=[embedding_matrix],
                                                   trainable=True)

        self.lstm = tf.keras.layers.LSTM(dec_units,
                                       return_sequences=True,
                                       # whether to return the last output sequence, or the full sequence
                                       return_state=True,  # whether to return the last state in additon to output
                                       recurrent_initializer='glorot_uniform')
        self.x_context = tf.keras.layers.Dense(embedding_dim)
        self.fc1 = tf.keras.layers.Dense(dec_units)
        self.fc2 = tf.keras.layers.Dense(vocab_size, activation=tf.keras.activations.softmax)
    def __call__(self, dec_input, hidden_state, enc_output,enc_pad_mask,
                context_vector_last, use_coverage,prev_coverage):
        #hidden_state 输入lstm的hidden_state
        #context_vector 上下文词向量 batch_size,enc_units*2

        x = self.embedding(dec_input)  # x的结果shape  =(batch_size, 1,embedding_dim)
        # 与 attention合并
        x = tf.concat([tf.expand_dims(context_vector_last, axis=1), x],
                      axis=-1)  # shape= (batch_size, 1, 2*enc_hidden_size+ embedding_dim)
        x = self.x_context(x) #重新映射为 embeding_dim

        output, hidden_state, cell_state = self.lstm(x,initial_state=hidden_state)

        # dec_state shape = 2* (batch_size,1*dec_units)
        dec_state  = [hidden_state, cell_state]
        #求本轮context_vector

        dec_state_concat = tf.concat([hidden_state, cell_state], axis =-1)
        context_vector, attentions, coverage_ret = self.attention(dec_state_concat,
                                                                  enc_output,
                                                                  enc_pad_mask,
                                                                  use_coverage,
                                                                  prev_coverage)
        #print("context vector shape", context_vector.shape)

        output = tf.reshape(output, (-1, output.shape[2]))  #batch_size, dec_units
       # 按照作者做法再与context_vector 合并  可加快训练速度
        output = tf.concat([output, context_vector], axis = -1)  #batch_size, dec_hidden_size +2*enc_units
        output = self.fc1(output)
        prediction = self.fc2(output)
        #prediction 维度 batch_size,vocab_size

        return x, prediction, dec_state,context_vector,attentions,coverage_ret

class Pointer(tf.keras.layers.Layer):

    def __init__(self):
        super(Pointer, self).__init__()
        self.w_s_reduce = tf.keras.layers.Dense(1)
        self.w_i_reduce = tf.keras.layers.Dense(1)
        self.w_c_reduce = tf.keras.layers.Dense(1)

    def call(self, context_vector, dec_hidden, dec_inp):
        #context_vector 为当前context_vector
        dec_hidden = tf.concat([dec_hidden[0], dec_hidden[1]], axis =-1)  #batch_size, 2*dec_units
        dec_inp = tf.squeeze(dec_inp, axis =1 ) #转换为 （batch_size, embedding_size)
        return tf.nn.sigmoid(self.w_s_reduce(dec_hidden) + self.w_c_reduce(context_vector) +
                             self.w_i_reduce(dec_inp))
    #shape= batch_size, 1


if __name__ == '__main__':
    # GPU资源配置
    config_gpu()
    # 读取vocab训练
    vocab = Vocab(vocab_path)
    # 计算vocab size
    vocab_size = vocab.count

    # 使用GenSim训练好的embedding matrix
    embedding_matrix = load_embedding_matrix()

    enc_max_len = 200
    dec_max_len = 41
    batch_size = 16
    embedding_dim = 300
    units = 128
    context_vector_last = tf.zeros(shape = (batch_size,2*units))
    # 编码器结构
    encoder = Encoder( embedding_matrix, units, batch_size)
    # encoder input
    enc_inp = tf.ones(shape=(batch_size, enc_max_len), dtype=tf.int32)
    # decoder input
    dec_inp = tf.ones(shape=(batch_size, dec_max_len), dtype=tf.int32)
    # enc pad mask
    enc_pad_mask = tf.ones(shape=(batch_size, enc_max_len), dtype=tf.int32)

    # encoder hidden
   # enc_hidden = encoder.initialize_hidden_state()

    enc_output, enc_hidden = encoder(enc_inp)
    # 打印结果
    print('Encoder output shape: (batch size, sequence length, units) {}'.format(enc_output.shape))
    print('Encoder Hidden state shape: (batch size, units) {}'.format(enc_hidden[0].shape))
    #
    reduce_state = ReduceState(units)
    enc_hidden = reduce_state(enc_hidden)
    # attention_layer = BahdanauAttention(units)
    # context_vector, attention_weights, coverage = attention_layer(enc_hidden, enc_output, enc_pad_mask)
    #
    # print("Attention context_vector shape: (batch size, units) {}".format(context_vector.shape))
    # print("Attention weights shape: (batch_size, sequence_length, 1) {}".format(attention_weights.shape))
    # print("Attention coverage: (batch_size, ) {}".format(coverage))

    decoder = Decoder( embedding_matrix, units,units, batch_size)

    dec_x, dec_out, dec_hidden, context_vector, attentions, coverage_ret= decoder(tf.random.uniform((batch_size, 1)),
                                          enc_hidden,enc_output,enc_pad_mask,
                                          context_vector_last, True, None)
    print('Decoder output shape: (batch_size, vocab size) {}'.format(dec_out.shape))
    print('Decoder dec_x shape: (batch_size, 1,embedding_dim + units) {}'.format(dec_x.shape))
    print("decoder dec_hidden shape:{}".format(dec_hidden[0].shape))
    print("context_vector shape:{}".format(context_vector.shape))
    pointer = Pointer()
    p_gen = pointer(context_vector, dec_hidden, dec_x)
    print('Pointer p_gen shape: (batch_size,1) {}'.format(p_gen.shape))
