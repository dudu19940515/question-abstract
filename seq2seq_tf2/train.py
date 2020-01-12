import tensorflow as tf
from seq2seq_tf2.seq2seq_model import Seq2Seq
from utils.wv_loader import get_vocab
from utils.config import model_path_trained,checkpoint_dir
from utils.gpu_utils import config_gpu
from seq2seq_tf2.train_helper import  train_model
from utils.param_utils import get_params

def train(params):
    config_gpu()

    vocab,_ = get_vocab(model_path_trained)

    params['vocab_size'] =len(vocab)

    print("Building the model")
    model = Seq2Seq(params)

    checkpoint = tf.train.Checkpoint(Seq2Seq = model)

    check_point_manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir,max_to_keep=5)

    train_model(model,vocab, params, check_point_manager)


if __name__ == '__main__':
    # 获得参数
    params = get_params()
    # 训练模型
    train(params)
