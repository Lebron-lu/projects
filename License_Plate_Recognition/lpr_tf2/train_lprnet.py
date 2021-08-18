import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from dataload import *
from model import *
import tensorflow as tf
import numpy as np
import argparse
import time

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)



def get_parser():
    parser = argparse.ArgumentParser(description='parameters to train lprnet')
    parser.add_argument('--epochs', default=3000, help='epochs to train the lprnet')
    parser.add_argument('--train_img_dir', default="", help='the train images path')
    parser.add_argument('--test_img_dir', default="", help='the test images path')
    parser.add_argument('--initial_lr', default=0.1, help='the initial learning rate')
    parser.add_argument('--dropout_rate', default=0.5, help='the dropout rate for layer')
    parser.add_argument('--lpr_len', default=7, help='the max length of license plate number')
    parser.add_argument('--train_batch_size', default=128, help='training batch size')
    parser.add_argument('--test_batch_size', default=128, help='testing batch size')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum for optimizer')
    parser.add_argument('--saved_model_folder', default="./saved_model", help='Location to save lprnet')
    parser.add_argument('--pretrained_model', default="", help='pretrained base lprnet')
    args = parser.parse_args()

    return args

# 评价函数
def metric(test_dataset, lprnet):
    tp, tp_error = 0, 0  # tp为正确预测数量, tp_error是错误预测数量
    start_time = time.time()
    for cur_batch, (test_imgs, test_labels) in enumerate(test_dataset):
        test_labels = tf.cast(test_labels, tf.int32)
        prebs = lprnet(test_imgs)
        preb_labels = list()
        for i in range(prebs.shape[0]):
            preb = prebs[i, :, :]
            preb_label = list()
            for j in range(preb.shape[1]):
                preb_label.append(np.argmax(preb[:, j]))
            no_repeat_blank_label = list()
            pre_c = preb_label[0]
            if pre_c != len(CHARS) - 1:
                no_repeat_blank_label.append(pre_c)
            for c in preb_label:
                if (pre_c == c) or (c == len(CHARS) - 1):
                    if c == (len(CHARS) - 1):
                        pre_c = c
                    continue
                no_repeat_blank_label.append(c)
                pre_c = c
            preb_labels.append(no_repeat_blank_label)
        for i, label in enumerate(preb_labels):
            if len(label) != len(test_labels[i]):
                tp_error += 1
                continue
            if (np.asarray(test_labels[i]) == np.asarray(label)).all():
                tp += 1
            else:
                tp_error += 1
                
    end_time = time.time()
    t = end_time - start_time
    acc = tp * 1.0 / (tp + tp_error)  # 准确率
    
    return acc, tp, tp_error, t
                                    

# 训练模型
def train():
    args = get_parser()

    if not os.path.exists(args.saved_model_folder):
        os.mkdir(args.saved_model_folder)

    # 实例化模型
    lprnet = LPRNet(lpr_len=args.lpr_len, class_num=len(CHARS), dropout_rate=args.dropout_rate)
    print("\n ********** Successful to build network! ********** \n")

    # 加载预训练模型
    if args.pretrained_model:
        lprnet.load_weights(args.pretrained_model)
        print("\n ********** Successful to load pretrained model! ********** \n")

    # 优化器使用 RMSprop or Adam
#     optimizer = tf.keras.optimizers.RMSprop(learning_rate=args.initial_lr, momentum=args.momentum)
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.initial_lr)

    # 加载训练数据集
    tr_imgs, tr_labels, train_imgs_num = load_data(img_dir=train_img_dir, lpr_len=lpr_len)
    tr_imgs = tr_imgs.map(preprocess,num_parallel_calls=4)
    train_dataset = tf.data.Dataset.zip((tr_imgs,tr_labels)).shuffle(train_imgs_num).batch(train_batch_size).prefetch(tf.data.experimental.AUTOTUNE).cache()

    # 加载测试数据集
    te_imgs, te_labels, test_imgs_num = load_data(img_dir=test_img_dir, lpr_len=lpr_len)
    te_imgs = te_imgs.map(preprocess,num_parallel_calls=4)
    test_dataset = tf.data.Dataset.zip((te_imgs, te_labels)).batch(test_batch_size).prefetch(tf.data.experimental.AUTOTUNE).cache()
    
    # 模型训练
    top_acc = 0.
    for cur_epoch in range(1, args.epochs + 1):
        for batch_index, (train_imgs, train_labels) in enumerate(train_dataset):
            start_time = time.time()
            with tf.GradientTape() as tape:
                train_logits = lprnet(train_imgs) #[N,66,18]
                train_labels = tf.cast(train_labels, tf.int32) #[N,7]
                train_logits = tf.transpose(train_logits, [2, 0, 1]) #[18,N,66]
                logits_shape = train_logits.shape
                logit_length = tf.fill([logits_shape[1]], logits_shape[0]) #(N)
                label_length = tf.fill([logits_shape[1]], args.lpr_len)
                loss = tf.nn.ctc_loss(labels=train_labels,
                                      logits=train_logits,
                                      label_length=label_length,
                                      logit_length=logit_length,
                                      logits_time_major=False,
                                      blank_index=len(CHARS) - 1)
                loss = tf.reduce_mean(loss)
            grads = tape.gradient(loss, lprnet.variables)
            optimizer.apply_gradients(grads_and_vars=zip(grads, lprnet.variables))
            end_time = time.time()
            print('\r' + "Epoch {0}/{1} || ".format(cur_epoch, epochs) 
                  + "Batch {0}/{1} || ".format(batch, train_imgs_num) # 修改的地方
                  + "Loss:{} || ".format(loss)
                  + "A Batch time:{0:.4f}s || ".format(end_time - start_time)
                  + "Learning rate:{0:.8f} || ".format(optimizer.lr.numpy().item()), end=''*20)
        acc, tp, tp_error, t = metric(args.test_img_dir, lprnet)
        print("\n******* Prediction accuracy: {0}/{1} || Acc:{2:.2f}%".format(tp, tp + tp_error, acc*100))
        print("******* Test speed: {}s 1/{}\n".format(t / (tp + tp_error), tp + tp_error))
        # 保存模型
        if acc >= top_acc:
            top_acc = acc
            lprnet.save(args.saved_model_folder, save_format='tf')

    # 将.pb模型转为.tflite
    cvtmodel = tf.keras.models.load_model(args.saved_model_folder)
    converter = tf.lite.TFLiteConverter.from_keras_model(cvtmodel)
    tflite_model = converter.convert()
    with open('lprnet' + '{}'.format(np.around(top_acc * 100)) + '.tflite', "wb") as f:
        f.write(tflite_model)
    print("\n ********** Successful to convert tflite model! ********** \n")

if __name__ == '__main__':
    train()
