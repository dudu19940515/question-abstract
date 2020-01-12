import tensorflow as tf
class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, embedding_matrix,enc_units, batch_sz):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding =tf.keras.layers.Embedding(vocab_size, embedding_dim,weights = [embedding_matrix], trainable = False)
        #self.gru = tf.keras.layers.GRU(self.enc_units, return_sequences=True, return_state=True)
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            self.gru = tf.keras.layers.CuDNNGRU(self.enc_units,
                                                return_sequences=True,
                                                return_state=True,
                                                recurrent_initializer='glorot_uniform')
        else:
            self.gru = tf.keras.layers.GRU(self.enc_units,
                                           return_sequences=True, #whether to return the last output sequence, or the full sequence
                                           return_state=True, #whether to return the last state in additon to output
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
        # code
        self.W1 = tf.keras.layers.Dense(units)  # 用于 encoder units
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        # hidden shape == (batch_size, hidden size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden size)
        # we are doing this to perform addition to calculate the score
        # code
        hidden_with_time_axis = tf.expand_dims(query, 1)
        #print("hidden_with_time_axis shape", hidden_with_time_axis.shape)
        # score shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        # the shape of the tensor before applying self.V is (batch_size, max_length, units)
        # code
        score = self.V(tf.nn.tanh(self.W1(values) + self.W2(hidden_with_time_axis)))
        # vaues 是encoder的值
        # shape = (batch_size, max_length, 1)

        # attention_weights shape == (batch_size, max_length, 1)
        # code
        #print("scores 的维度", score.shape)
        attention_weights = tf.nn.softmax(score, axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        # code
        context_vector = attention_weights * values  # values 的shape = (batch_size, maxlength, hidden_size)
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights


class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, embedding_matrix, dec_units, batch_sz, ):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, weights=[embedding_matrix])
        #         self.gru = tf.keras.layers.GRU(self.dec_units,
        #                                        return_sequences=True,
        #                                        return_state=True,
        #                                        recurrent_initializer='glorot_uniform')
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            self.gru = tf.keras.layers.CuDNNGRU(self.dec_units,
                                                return_sequences=True,
                                                return_state=True,
                                                recurrent_initializer='glorot_uniform')
        else:
            self.gru = tf.keras.layers.GRU(self.dec_units,
                                           return_sequences=True,
                                           # whether to return the last output sequence, or the full sequence
                                           return_state=True,  # whether to return the last state in additon to output
                                           recurrent_initializer='glorot_uniform')
        self.bigru = tf.keras.layers.Bidirectional(self.gru, merge_mode='concat')
        self.fc = tf.keras.layers.Dense(vocab_size)

        # used for attention
        self.attention = BahdanauAttention(self.dec_units)

        # code

    def call(self, x, hidden, enc_output):
        # code
        context_vector, attention_weights = self.attention(hidden, enc_output)

        x = self.embedding(x)  # x的结果shape  =(batch_size,1, embedding_dim)
        # 与 attention合并
        x = tf.concat([tf.expand_dims(context_vector, axis=1), x],
                      axis=-1)  # shape= (batch_size, 1, hidden_size+ embedding_dim)

        output, forward_state, backward_state = self.bigru(x)

        state = tf.concat([forward_state, backward_state], axis=1)
        #print("变换维度之前 output的维度：", (output.shape))
        output = tf.reshape(output, (-1, output.shape[2]))
       # print("output变换维度后的：", (output.shape))

        x = self.fc(output)

        return x, state, attention_weights

if __name__ == '__main__':
    embedding_matrix = tf.ones([20000, 300])
    encoder = Encoder(vocab_size=20000, embedding_dim=300, embedding_matrix=embedding_matrix, enc_units=150,
                      batch_sz=64)
    simple_hidden = encoder.initialize_hidden_state()
    example_input_batch = tf.ones([64, 88], dtype=tf.int32)
    sample_output, sample_hidden = encoder(example_input_batch, simple_hidden)
    print('Encoder output shape: (batch size, sequence length, units) {}'.format(sample_output.shape))
    print('Encoder Hidden state shape: (batch size, units) {}'.format(sample_hidden.shape))

    attention_layer = BahdanauAttention(units=128)
    attention_results, attention_weights = attention_layer(sample_hidden, sample_output)
    print("Attention result shape: (batch size, units) {}".format(attention_results.shape))
    print("Attention weights shape: (batch_size, sequence_length, 1) {}".format(attention_weights.shape))

    decoder = Decoder(vocab_size=20000, embedding_dim=300, embedding_matrix=embedding_matrix, dec_units=150,
                      batch_sz=64)

    sample_decoder_output, _, _ = decoder(tf.random.uniform((64, 1)),
                                          sample_hidden, sample_output)

    print('Decoder output shape: (batch_size, vocab size) {}'.format(sample_decoder_output.shape))