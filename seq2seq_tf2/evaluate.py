
import tensorflow as tf

from utils.preprocess import preprocess_sentence

def evaluate(model,vocab,reverse_vocab, sentence, params):

    attention_plot = tf.zeros((vocab['max_length_targ'], vocab['max_length_inp']))
    results = ''
    #batch_size = params['batch_size']
    inputs = tf.convert_to_tensor(sentence)
    start_index = vocab['<START>']
    enc_units= params['enc_units']
    hidden = tf.zeros((1, enc_units))

    enc_output,enc_hidden = model.encoder(inputs, hidden)
    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([start_index], 0)

    for t in range(params['max_length_targ']):

        context_vector, attention_weights= model.attention(dec_hidden, enc_output)

        predictions, dec_hidden = model.decoder(dec_input, dec_hidden, enc_output, context_vector)
        attention_weights = tf.reshape(attention_weights, (-1,))

        attention_plot[t] = attention_weights.numpy()

        prediction_id = tf.argmax(predictions[0]).numpy()

        results = results+reverse_vocab[prediction_id] +' '

        if reverse_vocab[prediction_id] == '<STOP>':

            return results,  attention_plot

        dec_input = tf.expand_dims([prediction_id],0)


    return results, attention_plot

def translate(sentence):

    sentence = preprocess_sentence(sentence)

    result,_ = evaluate(model,vocab, reverse_vocab, params)

    print("predicted translation: {}".format(reuslt))





