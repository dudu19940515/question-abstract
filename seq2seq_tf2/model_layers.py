import tensorflow as tf
from utils.wv_loader import get_vocab,load_word2vec_file
from utils.config import model_path_trained
class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, embedding_matrix,enc_units, batch_sz):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding =tf.keras.layers.Embedding(vocab_size, embedding_dim,weights = [embedding_matrix],
                                                  trainable = False)

        self.gru = tf.keras.layers.GRU(self.enc_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.bigru = tf.keras.layers.Bidirectional(self.gru, merge_mode='concat')

    def call(self, x, hidden):
        # code
        x = self.embedding(x)
        hidden = tf.split(hidden, num_or_size_splits=2, axis=1)
        output, forward_state, backward_state = self.bigru(x, initial_state=hidden)
        state = tf.concat([forward_state, backward_state], axis=1)
        return output, state

    def initialize_hidden_state(self):
        return tf.zeros(shape = [self.batch_sz, 2*self.enc_units])


class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        # query为上次的GRU隐藏层
        # values为编码器的编码结果enc_output
        # 在seq2seq模型中，St是后面的query向量，而编码过程的隐藏状态hi是values。

        # hidden shape == (batch_size, hidden size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden size)
        # we are doing this to perform addition to calculate the score
        hidden_with_time_axis = tf.expand_dims(query, 1)

        # score shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        # the shape of the tensor before applying self.V is (batch_size, max_length, units)
        # 计算注意力权重值
        score = self.V(tf.nn.tanh(
            self.W1(values) + self.W2(hidden_with_time_axis)))

        # attention_weights sha== (batch_size, max_length, 1)
        attention_weights = tf.nn.softmax(score, axis=1)

        # # 使用注意力权重*编码器输出作为返回值，将来会作为解码器的输入
        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights


class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, embedding_matrix, dec_units, batch_sz, ):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, weights=[embedding_matrix],
                                                   trainable=False)
        #         self.gru = tf.keras.layers.GRU(self.dec_units,
        #                                        return_sequences=True,
        #                                        return_state=True,
        #                                        recurrent_initializer='glorot_uniform')

        self.gru = tf.keras.layers.GRU(self.dec_units,
                                       return_sequences=True,
                                       # whether to return the last output sequence, or the full sequence
                                       return_state=True,  # whether to return the last state in additon to output
                                       recurrent_initializer='glorot_uniform')
        #self.bigru = tf.keras.layers.Bidirectional(self.gru, merge_mode='concat')
        self.fc = tf.keras.layers.Dense(vocab_size)



        # code

    def call(self, x, hidden, enc_output,context_vector):
        # code


        x = self.embedding(x)  # x的结果shape  =(batch_size,1, embedding_dim)
        # 与 attention合并
        x = tf.concat([tf.expand_dims(context_vector, axis=1), x],
                      axis=-1)  # shape= (batch_size, 1, hidden_size+ embedding_dim)

        output, state = self.gru(x)

        #state = tf.concat([forward_state, backward_state], axis=1)
        #print("变换维度之前 output的维度：", (output.shape))
        output = tf.reshape(output, (-1, output.shape[2]))
       # print("output变换维度后的：", (output.shape))

        prediction = self.fc(output)

        return prediction, state

class Pointer(tf.keras.layers.Layer):

    def __init__(self):
        super(Pointer, self).__init__()
        self.w_s_reduce = tf.keras.layers.Dense(1)
        self.w_i_reduce = tf.keras.layers.Dense(1)
        self.w_c_reduce = tf.keras.layers.Dense(1)

    def call(self, context_vector, dec_hidden, dec_inp):
        return tf.nn.sigmoid(self.w_s_reduce(dec_hidden) + self.w_c_reduce(context_vector) + self.w_i_reduce(dec_inp))

if __name__ == '__main__':
    vocab, reverse_vocab = get_vocab(model_path_trained)
    # 计算vocab size
    vocab_size = len(vocab)
    # 使用GenSim训练好的embedding matrix
    embedding_matrix = load_word2vec_file(model_path_trained)

    input_sequence_len = 250
    BATCH_SIZE = 64
    embedding_dim = 300
    units = 1024

    # 编码器结构
    encoder = Encoder(vocab_size, embedding_dim, embedding_matrix, units//2, BATCH_SIZE)
    # example_input
    example_input_batch = tf.ones(shape=(BATCH_SIZE, input_sequence_len), dtype=tf.int32)
    # sample input
    sample_hidden = encoder.initialize_hidden_state()

    sample_output, sample_hidden = encoder(example_input_batch, sample_hidden)
    # 打印结果
    print('Encoder output shape: (batch size, sequence length, units) {}'.format(sample_output.shape))
    print('Encoder Hidden state shape: (batch size, units) {}'.format(sample_hidden.shape))

    attention_layer = BahdanauAttention(10)
    attention_result, attention_weights = attention_layer(sample_hidden, sample_output)

    print("Attention result shape: (batch size, units) {}".format(attention_result.shape))
    print("Attention weights shape: (batch_size, sequence_length, 1) {}".format(attention_weights.shape))

    decoder = Decoder(vocab_size, embedding_dim, embedding_matrix, units, BATCH_SIZE)
    sample_decoder_output, _, = decoder(tf.random.uniform((64, 1)),
                                        sample_hidden, sample_output, attention_result)

    print('Decoder output shape: (batch_size, vocab size) {}'.format(sample_decoder_output.shape))

