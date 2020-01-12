import tensorflow as tf
from utils.config import  checkpoint_dir
from seq2seq_tf2.seq2seq_model import Seq2Seq
import math
def load_model(params):
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    model = Seq2Seq(params)
    ckpt = tf.train.Checkpoint(Seq2Seq = model)
    ckpt.restore(latest)

    return ckpt.Seq2Seq


def batch_predict(model, inps, params):
    # 判断输入长度
    batch_size = len(inps)
    # 开辟结果存储list
    preidicts = [''] * batch_size

    inps = tf.convert_to_tensor(inps)
    enc_units = params['enc_units'] *2
    # 0. 初始化隐藏层输入
    hidden = tf.zeros((batch_size, enc_units))
    # 1. 构建encoder
    enc_output, enc_hidden = model.encoder(inps, hidden)
    # 2. 复制
    dec_hidden = enc_hidden
    # 3. <START> * BATCH_SIZE
    dec_input = tf.expand_dims([vocab['<START>']] * batch_size, 1)

    context_vector, _ = model.attention(dec_hidden, enc_output)
    # Teacher forcing - feeding the target as the next input
    for t in range(max_length_targ):
        # 计算上下文
        context_vector, attention_weights = model.attention(dec_hidden, enc_output)
        # 单步预测
        predictions, dec_hidden = model.decoder(dec_input,
                                                dec_hidden,
                                                enc_output,
                                                context_vector)

        # id转换 贪婪搜索
        predicted_ids = tf.argmax(predictions, axis=1).numpy()

        for index, predicted_id in enumerate(predicted_ids):
            preidicts[index] += reverse_vocab[predicted_id] + ' '

        # using teacher forcing
        dec_input = tf.expand_dims(predicted_ids, 1)

    results = []
    for preidict in preidicts:
        # 去掉句子前后空格
        preidict = preidict.strip()
        # 句子小于max len就结束了 截断
        if '<STOP>' in preidict:
            # 截断stop
            preidict = preidict[:preidict.index('<STOP>')]
        # 保存结果
        results.append(preidict)
    return results

def model_predict(model,data_X,batch_size, params):
    # 存储结果
    results=[]
    # 样本数量
    sample_size=len(data_X)
    # batch 操作轮数 math.ceil向上取整 小数 +1
    # 因为最后一个batch可能不足一个batch size 大小 ,但是依然需要计算
    steps_epoch = math.ceil(sample_size/batch_size)
    # [0,steps_epoch)
    for i in tqdm(range(steps_epoch)):
        batch_data = data_X[i*batch_size:(i+1)*batch_size]
        results+=batch_predict(model,batch_data, params)
    return results