import pickle as pkl
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data')

##データ格納する関数
def model_inputs(real_dim, z_dim):
    input_real = tf.placeholder(tf.float32, (None, real_dim), name='input_real')
    input_z = tf.placeholder(tf.float32, (None, z_dim), name='input_z')
    
    return input_real, input_z


##ジェネレータ（生成器）
def generator(z, out_dim, n_units=128, reuse=False, alpha=0.01):
    with tf.variable_scope('generator', reuse=reuse):
        
        #denseで128のノードに結合z➡h1
        h1 = tf.layers.dense(z, n_units, activation=None)
        #Leaky ReLU適用
        h1 = tf.maximum(alpha*h1, h1)
        
        logits = tf.layers.dense(h1, out_dim, activation=None)
        out = tf.tanh(logits)
        
        return out


##ディスクリミネータ（弁別器）
def discriminator(x, n_units=128, reuse=False, alpha=0.01):
    with tf.variable_scope('discriminator', reuse=reuse):
        
        h1 = tf. layers.dense(x, n_units, activation=None)
        h1 = tf.maximum(alpha*h1, h1)

        logits = tf.layers.dense(h1, 1, activation=None)
        out = tf.sigmoid(logits)
        
        return out , logits


##hyper parametersの初期化
input_size = 784
z_size = 100
#各中間層ノード数
g_hidden_size = 128
d_hidden_size = 128
#Leaky ReLUの傾きα
alpha = 0.01
smooth = 0.1


##計算グラフの定義
#描画リセット
tf.reset_default_graph()
#入力変数
input_real, input_z = model_inputs(input_size, z_size)
#ジェネレータモデル定義
g_model = generator(input_z, input_size, n_units= g_hidden_size, alpha=alpha)
#ディスクリミネータモデル定義
d_model_real, d_logits_real = discriminator(input_real, n_units=d_hidden_size, alpha=alpha)
d_model_fake, d_logits_fake = discriminator(g_model,  n_units= d_hidden_size, reuse=True,alpha=alpha)


##損失関数の定義
#2つの値を与え、差がどれくらいかを対数で表すシグモイドクロスエントロピー関数
d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_real, 
                                                                                                                    labels=tf.ones_like(d_logits_real)*(1 - smooth)))
d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake,
                                                                                                                     labels=tf.zeros_like(d_logits_real)))

d_loss = d_loss_real + d_loss_fake

g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake,
                                                                                                            labels=tf.ones_like(d_logits_fake)))


##最適化の定義
learning_rate = 0.002
#トレーニング中の変数一式を取り出す
t_vars = tf.trainable_variables()
#分類
d_vars = [var for var in t_vars if var.name.startswith('discriminator')]
g_vars = [var for var in t_vars if var.name.startswith('generator')]

d_train_optimize = tf.train.AdamOptimizer(learning_rate).minimize(d_loss, var_list=d_vars)
g_train_optimize = tf.train.AdamOptimizer(learning_rate).minimize(g_loss, var_list=g_vars)

batch_size = 100


##トレーニング
epochs = 100
losses = []#初期化
samples = []
#途中経過を保存
saver = tf.train.Saver(var_list=g_vars)
with tf.Session() as sess:
    #初期化
    sess.run(tf.global_variables_initializer())
    for e in range(epochs):
        #5万5千件の訓練データ÷100回
        for i in range(mnist.train.num_examples//batch_size):
            #100個無作為生成
            batch = mnist.train.next_batch(batch_size)
            batch_images = batch[0].reshape((batch_size, 784))
            batch_images = batch_images *2 -1
            
            #一様分布で乱数発生
            batch_z = np.random.uniform(-1,1,size=(batch_size, z_size))
            
            #feed_dictで特定の変数の値を上書き
            _= sess.run(d_train_optimize, feed_dict= {input_real: batch_images, input_z: batch_z})
            _= sess.run(g_train_optimize, feed_dict= {input_z: batch_z})

        #ロスの計算、2通りの書き方
        train_loss_d = sess.run(d_loss, {input_z: batch_z, input_real: batch_images})
        train_loss_g = g_loss.eval({input_z: batch_z})
            
        print("エポック {}/{} 回目".format(e+1, epochs),#例：1/100回目
                "Discriminator Loss: {:.4f}".format(train_loss_d),
                "Generator Loss     : {:.4f}".format(train_loss_g))
        
        losses.append((train_loss_d, train_loss_g))
            
        #16個ｘ100個のノイズでサンプルデータ作成
        sample_z = np.random.uniform(-1, 1, size=(16,z_size))
        gen_samples = sess.run(generator(input_z, input_size, n_units= g_hidden_size, reuse=True, alpha=alpha),
                              feed_dict={input_z: sample_z})
        samples.append(gen_samples)
        saver.save(sess, "./checkpoints/generator.ckpt")

with open("training_samples.pkl" , "wb") as f :
    pkl.dump(samples, f)


##学習精度（ロス）の可視化
#複数のグラフを描画
fig, ax = plt.subplots()
#損失を保存する変数
losses = np.array(losses)
#このままではDとG両方混ざってるので転置
plt.plot(losses.T[0], label="D")
plt.plot(losses.T[1], label="G")
plt.title("Train Loss")
plt.legend()


##チェックポイント時の数値から画像生成
def view_samples(epochs, samples):
    #7x7pixec, 4行4列, xy軸も表示
    fig, axes =plt.subplots(figsize=(7,7), nrows=4, ncols=4, sharex=True, sharey=True)
    for ax, img in zip(axes.flatten(), samples[epochs]):
        #axの軸非表示
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        #28x28のグレースケールで描画
        im = ax.imshow(img.reshape(28,28), cmap="Greys_r")

    return fig, axes

with open("training_samples.pkl", "rb") as f:
    samples = pkl.load(f)

_ = view_samples(-1, samples)

rows, cols = 10, 6
fig, axes = plt.subplots(figsize=(7,12),nrows=rows, ncols=cols, sharex=True, sharey=True)

for sample, ax_row in zip(samples[::int(len(samples)/rows)], axes):
    for img, ax in zip(sample[::int(len(sample)/cols)], ax_row):
        #28x28のグレースケールで描画
        im = ax.imshow(img.reshape(28,28), cmap="Greys_r")
        #ax(各グラフ）の軸非表示
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)


##チェックポイントファイルから機械に手書き画像を生成させる
saver = tf.train.Saver(var_list= g_vars)
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('checkpoints'))
    #一様分布を与える
    sample_z = np.random.uniform(-1,1, size=(16, z_size))
    #↑のノイズでまた一次元配列のデータをつくる
    gen_samples = sess.run(generator(input_z, input_size, n_units=g_hidden_size, reuse=True, alpha=alpha),
                          feed_dict= {input_z: sample_z})
_= view_samples(0, [gen_samples])
