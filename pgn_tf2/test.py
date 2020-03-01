# -*- coding:utf-8 -*-
import tensorflow as tf
from pgn_tf2.batcher import batcher
from pgn_tf2.pgn_model import PGN
from tqdm import tqdm

from pgn_tf2.test_helper import beam_decode
from seq2seq_tf2.test import save_predict_result
from utils.config import checkpoint_dir
from utils.gpu_utils import config_gpu
from utils.wv_loader import Vocab
from utils.params_utils import get_params
from utils.file_utils import get_result_filename
from multiprocessing import Pool, set_start_method, Lock

def test(params):
    assert params["mode"].lower() in ["test", "eval"], "change training mode to 'test' or 'eval'"
    assert params["beam_size"] == params["batch_size"], "Beam size must be equal to batch_size, change the params"
    # GPU资源配置
    config_gpu(use_cpu=True)

    print("Building the model ...")
    model = PGN(params)

    print("Creating the vocab ...")
    vocab = Vocab(params["vocab_path"], params["vocab_size"])
    params['vocab_size'] = vocab.count

    print("Creating the batcher ...")
    b = batcher(vocab, params)

    print("Creating the checkpoint manager")
    checkpoint = tf.train.Checkpoint(PGN=model)
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=5)
    checkpoint.restore(checkpoint_manager.latest_checkpoint)
    if checkpoint_manager.latest_checkpoint:
        print("Restored from {}".format(checkpoint_manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")
    print("Model restored")

    if params['mode'] == 'eval' or params['mode'] == 'test':
        for batch in b:
            #print("batch is:\n", batch)

            yield beam_decode(model, batch, vocab, params)

    else:
        for batch in b:
            print(beam_decode(model, batch, vocab, params))


def test_and_save(params):
    assert params["test_save_dir"], "provide a dir to save the results"
    gen = test(params)
    results = []
    with tqdm(total=params["num_to_test"], position=0, leave=True) as pbar:
        for i in range(params["num_to_test"]):
            trial = next(gen)
            results.append(trial.abstract)
            pbar.update(1)
    return results


def predict_result(params, result_save_path):
    # 预测结果
    results = test_and_save(params)
    # 保存结果
    save_predict_result(results, result_save_path)



if __name__ == '__main__':
    # 获得参数
    params = get_params()
    params['batch_size'] = 3
    params['beam_size'] = 3
    params['mode'] = 'test'
    result_save_path = get_result_filename(params['batch_size'], 30, 400, 300)
    predict_result(params,result_save_path)
