import warnings
warnings.filterwarnings("ignore")
import sys

#sys.path.append('/Users/doumengge/Desktop/开课吧/开课吧/project1')
import pandas as pd
import numpy as np
from utils.data_loader import build_dataset,load_data
from utils.wv_loader import load_vocab
from utils.config import *
from seq2seq_tf2.batcher import train_batch_generator
from seq2seq_tf2.seq2seq_model2 import Seq2Seq
import tensorflow as tf
import time


def train_model(model, vocab, params,checkpoint_manager):
    epochs = params['epochs']
    batch_size = params['batch_size']
    pad_index = vocab.word2id[vocab.PAD_TOKEN]
    start_index = vocab.word2id[vocab.START_DECODING]
    unk_index = vocab['<UNK>']

    #计算vocab_size
    params['vocab_size'] = vocab.count


    optimizer = tf.keras.optimizers.Adam(name = 'Adam',learning_rate=0.01)
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

    def loss_function(real, pred, padding_mask=None):
        loss = 0
        #real shape: (batch_size, max_len_y)

        for t in range(real.shape[1]):
            if padding_mask:

                loss_ = loss_object(real[:,t], pred[:,t,:])
                mask = tf.cast(padding_mask[:,t],dtype= loss_.dtype)
                loss_ = mask*loss_
                loss_ = tf.reduce_sum(loss_, axis = 0)
                num_no_pad = tf.reduce_sum(mask)
                loss+=loss_/num_no_pad
            else:
                loss_ = loss_object(real[:, t], pred[:, t, :])
                loss_ = tf.reduce_mean(loss_, axis=0)  # batch-wise
                loss += loss_
        return loss/real.shape[1]

    @tf.function(input_signature=(tf.TensorSpec(shape=[params["batch_size"], params["max_enc_len"]], dtype=tf.int64),
                                  tf.TensorSpec(shape=[params["batch_size"], params["max_dec_len"]], dtype=tf.int64)))
    def train_step(input_batch, target_batch):
        """
        计算一个时间步上面的loss
        """
        batch_loss = 0
        with tf.GradientTape() as tape:

            enc_output, enc_hidden = model.call_encoder(input_batch)

            decoder_input = tf.expand_dims([start_index] * batch_size, 1)

            dec_hidden = enc_hidden

            predictions, _ = model(decoder_input, dec_hidden, enc_output, target_batch)
            #print(predictions.shape)
            batch_loss += loss_function(target_batch[:,1:], predictions)

            variables = model.encoder.trainable_variables + model.attention.trainable_variables+model.decoder.trainable_variables

            gradients = tape.gradient(batch_loss, variables)

            optimizer.apply_gradients(zip(gradients, variables))

            return batch_loss

    dataset, steps_per_epoch = train_batch_generator(batch_size)

    for epoch in range(epochs):
        start_time = time.time()
        total_loss = 0


        for batch, (input_batch, target_batch) in enumerate(dataset.take(steps_per_epoch)):
            batch_loss = train_step(input_batch, target_batch,)
            #print(target_batch.shape)
            total_loss += batch_loss
            if batch % 50 == 0:
                print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                             batch,
                                                             batch_loss.numpy()))
        # saving (checkpoint) the model every 2 epochs
        if (epoch + 1) % 2 == 0:
            ckpt_save_path = checkpoint_manager.save()
            print('Saving checkpoint for epoch {} at {}'.format(epoch + 1,
                                                                ckpt_save_path))

        print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                            total_loss / steps_per_epoch))
        print('Time taken for 1 epoch {} sec\n'.format(time.time() - start_time))

