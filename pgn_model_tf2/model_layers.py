import tensorflow as tf
from utils.wv_loader import get_vocab,load_word2vec_file,load_embedding_matrix
from utils.config import save_wv_model_path,vocab_path
from utils.wv_loader import Vocab


def mask_attention(enc_padding_mask, attn_dist):
    """
    softmax后重新 enc_pad_ mask 后renomarlize
    """
    attn_dist = tf.squeeze(attn_dist, axis=2)
    mask = tf.cast(enc_padding_mask, dtype=attn_dist.dtype)
    attn_dist *= mask  # apply mask
    masked_sums = tf.reduce_sum(attn_dist, axis=1)  # shape (batch_size)
    attn_dist = attn_dist / tf.reshape(masked_sums, [-1, 1])  # re-normalize
    attn_dist = tf.expand_dims(attn_dist, axis=2)
    return attn_dist

class Encoder(tf.keras.Model):
    def __init__(self,  embedding_matrix,enc_units, batch_sz):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        vocab_size, embedding_dim =  embedding_matrix.shape
        self.embedding =tf.keras.layers.Embedding(vocab_size, embedding_dim,weights = [embedding_matrix],
                                                  trainable = False)

        self.lstm = tf.keras.layers.LSTM(self.enc_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.bilstm = tf.keras.layers.Bidirectional(self.lstm)
        #默认是concat
    def call(self, x):
        # code
        x = self.embedding(x)
        #hidden = tf.split(hidden, num_or_size_splits=2, axis=1)
        output, forward_state,f_cell_state, backward_state,b_cell_state = self.bilstm(x)
        #print("cell shape is {}".format(cell_state.shape))
        hidden = tf.keras.layers.concatenate([forward_state, backward_state], axis=-1)
        cell_state = tf.keras.layers.concatenate([f_cell_state, b_cell_state], axis=-1)
        #output= self.bilstm(x)
        return output, hidden, cell_state

    # def initialize_hidden_state(self):
    #     return tf.zeros(shape = [self.batch_sz, self.enc_units])


class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W_s = tf.keras.layers.Dense(units)
        self.W_h = tf.keras.layers.Dense(units)
        self.W_c = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values, enc_pad_mask, use_coverage=False, prev_coverage=None):

        # query为上次的GRU隐藏层 dec_hidden = (batch_size, units)
        # values为编码器的编码结果enc_output = (batch_size, max_len_enc, units)


        # hidden shape == (batch_size, hidden size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden size)
        # we are doing this to perform addition to calculate the score
        hidden_with_time_axis = tf.expand_dims(query, 1)

        # score shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        # the shape of the tensor before applying self.V is (batch_size, max_length, units)
        # 计算注意力权重值

        if use_coverage and prev_coverage is not None:

            e = self.V(tf.nn.tanh(
                self.W_h(values) + self.W_s(hidden_with_time_axis)+ self.W_c(prev_coverage)))
            #e.shape = [batch_size, max_enc_len, 1)
        # attention_weights shape== (batch_size, max_length, 1)
            mask = tf.cast(enc_pad_mask, dtype = e.dtype)
            mask_score = tf.squeeze(e, axis = -1) *mask  #先转换为shape = (batch_size,max_len_enc) 才能计算
            masked_score = tf.expand_dims(mask_score, axis = 2)
            attention_weights = tf.nn.softmax(masked_score, axis=1)
            #attention_weights shape = (batch_size,max_len_enc,1)
            attention_weights = mask_attention(enc_pad_mask, attention_weights)
            coverage = attention_weights+ prev_coverage
        else:
            e = self.V(tf.nn.tanh( self.W_s(values) + self.W_h(hidden_with_time_axis)))
            mask = tf.cast(enc_pad_mask, dtype=e.dtype)
            mask_score = tf.squeeze(e, axis=-1) * mask  # 先转换为shape = (batch_size,max_len_enc) 才能计算
            masked_score = tf.expand_dims(mask_score, axis=2)
            attention_weights = tf.nn.softmax(masked_score, axis=1)
            attention_weights = mask_attention(enc_pad_mask, attention_weights)
            if use_coverage:
                coverage = attention_weights
            else:
                coverage = []

        # # 使用注意力权重*编码器输出作为返回值，将来会作为解码器的输入
        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights, coverage


class Decoder(tf.keras.Model):
    def __init__(self, embedding_matrix, dec_units, batch_sz, ):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        vocab_size, embedding_dim = embedding_matrix.shape
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, weights=[embedding_matrix],
                                                   trainable=False)
        #         self.gru = tf.keras.layers.GRU(self.dec_units,
        #                                        return_sequences=True,
        #                                        return_state=True,
        #                                        recurrent_initializer='glorot_uniform')

        self.lstm = tf.keras.layers.LSTM(self.dec_units,
                                       return_sequences=True,
                                       # whether to return the last output sequence, or the full sequence
                                       return_state=True,  # whether to return the last state in additon to output
                                       recurrent_initializer='glorot_uniform')
        #self.bigru = tf.keras.layers.Bidirectional(self.gru, merge_mode='concat')
        self.fc = tf.keras.layers.Dense(vocab_size,activation=tf.keras.activations.softmax)

    def __call__(self, dec_input, hidden, cell_state, enc_output,context_vector):
        # code


        x = self.embedding(dec_input)  # x的结果shape  =(batch_size, embedding_dim)
        # 与 attention合并
        x = tf.concat([tf.expand_dims(context_vector, axis=1), x],
                      axis=-1)  # shape= (batch_size, 1, enc_hidden_size+ embedding_dim)

        output, state,_ = self.lstm(x,initial_state=[hidden,cell_state])

        #state = tf.concat([forward_state, backward_state], axis=1)
        #print("变换维度之前 output的维度：", (output.shape))
        output = tf.reshape(output, (-1, output.shape[2]))  #batch_size, dec_hidden_size
       # 按照作者做法再与context_vector 合并  可加快训练速度

        output = tf.concat([output, context_vector], axis = -1)  #batch_size, dec_hidden_size +enc_hidden_size

        prediction = self.fc(output)
        #prediction 维度 batch_size, 1, vocab_size

        return x, prediction, state

class Pointer(tf.keras.layers.Layer):

    def __init__(self):
        super(Pointer, self).__init__()
        self.w_s_reduce = tf.keras.layers.Dense(1)
        self.w_i_reduce = tf.keras.layers.Dense(1)
        self.w_c_reduce = tf.keras.layers.Dense(1)

    def call(self, context_vector, dec_hidden, dec_inp):
        dec_inp = tf.squeeze(dec_inp, axis =1 ) #转换为 （batch_size, embedding_size+enc_hidden_size)

        return tf.nn.sigmoid(self.w_s_reduce(dec_hidden) + self.w_c_reduce(context_vector) +
                             self.w_i_reduce(dec_inp))
    #shape= batch_size, 1

if __name__ == '__main__':
    #config_gpu()
    # 读取vocab训练
    vocab = Vocab(vocab_path)
    # 计算vocab size
    vocab_size = vocab.count

    # 使用GenSim训练好的embedding matrix
    embedding_matrix = load_embedding_matrix()

    enc_max_len = 200
    dec_max_len = 41
    batch_size = 64
    embedding_dim = 300
    units = 1024

    # 编码器结构
    encoder = Encoder(embedding_matrix, units//2, batch_size)
    # encoder input
    enc_inp = tf.ones(shape=(batch_size, enc_max_len), dtype=tf.int32)
    # decoder input
    dec_inp = tf.ones(shape=(batch_size, dec_max_len), dtype=tf.int32)
    # enc pad mask
    enc_pad_mask = tf.ones(shape=(batch_size, enc_max_len), dtype=tf.int32)

    # encoder hidden
    #enc_hidden = encoder.initialize_hidden_state()

    enc_output,enc_hidden,enc_cell= encoder(enc_inp)
    # 打印结果
    #print('Encoder output shape: (batch size, sequence length, units) {}'.format(len(enc_output)))
    print('Encoder output shape: (batch size, sequence length, units) {}'.format(enc_output.shape))
    print('Encoder Hidden state shape: (batch size, units) {}'.format(enc_hidden.shape))

    attention_layer = BahdanauAttention(1024)
    context_vector, attention_weights, coverage = attention_layer(enc_hidden, enc_output, enc_pad_mask, use_coverage = True)

    print("Attention context_vector shape: (batch size, units) {}".format(context_vector.shape))
    print("Attention weights shape: (batch_size, sequence_length, 1) {}".format(attention_weights.shape))
    print("Attention coverage: (batch_size, ) {}".format(coverage.shape))

    decoder = Decoder(embedding_matrix, units, batch_size)

    dec_x, dec_out, dec_hidden, = decoder(tf.random.uniform((64, 1)),
                                          enc_hidden,
                                          enc_cell,
                                          enc_output,
                                          context_vector)
    print('Decoder output shape: (batch_size, vocab size) {}'.format(dec_out.shape))
    print('Decoder dec_x shape: (batch_size, 1,embedding_dim + units) {}'.format(dec_x.shape))

    pointer = Pointer()
    p_gen = pointer(context_vector, dec_hidden, dec_x)
    print('Pointer p_gen shape: (batch_size,1) {}'.format(p_gen.shape))

