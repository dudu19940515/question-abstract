import tensorflow as tf
from utils.wv_loader import load_embedding_matrix, get_vocab
from seq2seq_tf2.model_layers import Encoder,Decoder,BahdanauAttention
from utils.config import model_path_trained

class Seq2Seq(tf.keras.Model):
    def __init__(self, params):
        super(Seq2Seq, self).__init__()
        self.embedding_matrix = load_embedding_matrix()
        self.params = params
        self.encoder = Encoder(params["vocab_size"],
                               params["embed_size"],
                               self.embedding_matrix,
                               params["enc_units"],
                               params["batch_size"])

        self.attention_layer = BahdanauAttention(self.params['attn_units'])
        self.decoder = Decoder(self.params['vocab_size'],
                               self.params['embed_size'],
                               self.embedding_matrix,
                               self.params['dec_units'],
                               self.params['batch_size'])

    def call_encoder(self,enc_inp):
        enc_hidden = self.encoder.initialize_hidden_state()
        output,enc_hidden= self.encoder(enc_inp, enc_hidden)
        return output, enc_hidden

    def call_decoder_onestep(self,dec_inp, dec_hidden, enc_output):

        context_vector, attention_weight = self.attention_layer(dec_hidden, enc_output)

        predictions, state = self.decoder(dec_inp,None, None, context_vector)

        return predictions, state, context_vector, attention_weight

    def call(self, decoder_inp, dec_hidden, enc_output, dec_target):
        predictions = []
        attentions = []
        context_vector,_ = self.attention_layer(dec_hidden, enc_output)

        for t in range(1,dec_target.shape[1]):
            prediction, dec_hidden = self.decoder(decoder_inp,
                                          dec_hidden,
                                          enc_output,
                                          context_vector)
            context_vector, attn = self.attention_layer(dec_hidden, enc_output)
            decoder_inp = tf.expand_dims(dec_target[:,t],1)

            predictions.append(prediction)
            attentions.append(attn)

        return tf.stack(predictions,1), dec_hidden


if __name__ == '__main__':
    # GPU资源配置
    #config_gpu()
    # 读取vocab训练
    vocab, reverse_vocab = get_vocab(model_path_trained)
    # 计算vocab size
    vocab_size = len(vocab)
    batch_size = 128
    input_sequence_len = 200

    params = {}
    params["vocab_size"] = vocab_size
    params["embed_size"] = 300
    params["enc_units"] = 256
    params["attn_units"] = 512
    params["dec_units"] = 512
    params["batch_size"] = batch_size

    model = Seq2Seq(params)

    # example_input
    example_input_batch = tf.ones(shape=(batch_size, input_sequence_len), dtype=tf.int32)

    # sample input
    sample_hidden = model.encoder.initialize_hidden_state()

    sample_output, sample_hidden = model.encoder(example_input_batch, sample_hidden)
    # 打印结果
    print('Encoder output shape: (batch size, sequence length, units) {}'.format(sample_output.shape))
    print('Encoder Hidden state shape: (batch size, units) {}'.format(sample_hidden.shape))

    attention_layer = BahdanauAttention(10)
    context_vector, attention_weights = attention_layer(sample_hidden, sample_output)

    print("Attention context_vector shape: (batch size, units) {}".format(context_vector.shape))
    print("Attention weights shape: (batch_size, sequence_length, 1) {}".format(attention_weights.shape))

    sample_decoder_output, _, = model.decoder(tf.random.uniform((batch_size, 1)),
                                              sample_hidden, sample_output, context_vector)

    print('Decoder output shape: (batch_size, vocab size) {}'.format(sample_decoder_output.shape))






