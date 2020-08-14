import pickle as pkl
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from scipy.io import loadmat

#Load
trainset = loadmat('data/train_32x32.mat')
testset = loadmat('data/test_32x32.mat')


##Test ( Sample試し読み)-------------------------------------
idx = np.random.randint(0, trainset['X'].shape[3], size=36)
#描画
fig, axes = plt.subplots(6, 6, sharex=True, sharey=True, figsize=(5,5),)
for ii , ax in zip(idx, axes.flatten()):
    ax.imshow(trainset['X'][:,:,:,ii], aspect='equal')
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
#余白0
plt.subplots_adjust(wspace=0, hspace=0)

#------------------------------------   ↓Main  ------------------------------------ #

##正規化
def scale(x, feature_ranges=(-1,1)):
    #0～255　→　０～１に変換
    x = ((x - x.min())/ (255 - x.min()))
    #－１～１に再変換
    min, max = feature_ranges
    x = x * (max - min) + min
    return x


##Datasetクラス
class Dataset:
    def __init__(self, train, test, val_flac=0.5, shuffle=False, scale_func=None):
        #目次折半
        split_index = int(len(test['y']) * (1 - val_flac))

        self.train_x, self.train_y = train['X'], train['y']
        #testデータを折半
        self.test_x, self.valid_x = test['X'][:,:,:,:split_index], test["X"][:,:,:,split_index:]
        self.test_y, self.valid_y = test['y'][:split_index], test['y'][split_index:]
    
        #XのMATLABデータをTensorFlow順に変換
        self.train_x = np.rollaxis(self.train_x, 3)
        self.valid_x = np.rollaxis(self.valid_x, 3)
        self.test_x =  np.rollaxis(self.test_x, 3)
        
        #Scale関数の指定
        if scale_func is None:
            self.scaler = scale
        else:
            self.scaler = scale_func
        #defaultはNone
        self.shuffle = shuffle 

    ##ミニバッチ生成関数の定義
    def batches(self, batch_size):
        ＃シャッフル
        if self.shuffle:
            #個数分のリスト作成
            idx = np.arange(len(dataset.train_x))
            np.random.shuffle(idx)

            self.train_x = self.train_x[idx]
            self.train_y = self.train_y[idx]
        
        n_batches = len(self.train_y)//batch_size    
        
        for ii in range(0, len(self.train_y), batch_size):
            x = self.train_x[ii:ii+batch_size]
            y = self.train_y[ii:ii+batch_size]

            yield self.scaler(x), y


#変数（プレースホルダー）を初期化する関数
def model_inputs(real_dim, z_dim):
    inputs_real = tf.placeholder(tf.float32, (None, *real_dim), name='input_real')
    inputs_z = tf.placeholder(tf.float32, (None, z_dim), name='input_z')
    
    return inputs_real, inputs_z


##Generator
def generator(z, output_dim, reuse=False, alpha=0.2, training=True):
    with tf.variable_scope('generator', reuse=reuse):
        
        #第1層
        x1 = tf.layers.dense(z, 4*4*512)
        #2次元→1次元配列にreshape
        x1 = tf.reshape(x1, (-1, 4, 4, 512))
        #傾きが少なくなるよう正規化
        x1 = tf.layers.batch_normalization(x1, training=training)
        #Reaky ReLU
        x1 = tf.maximum(alpha * x1, x1)
        
        #第2層 CONV 1
        x2 = tf.layers.conv2d_transpose(x1, 256, 5, strides=2,  padding='same')
        x2 = tf.layers.batch_normalization(x2, training=training)
        x2 = tf. maximum(alpha * x2, x2)
        #→8x8x256
        
        #第3層 CONV 2
        x3 = tf.layers.conv2d_transpose(x2, 128, 5, strides=2,  padding='same')
        x3 = tf.layers.batch_normalization(x3, training=training)
        x3 = tf. maximum(alpha * x3, x3)
        #→16x16x128
        
        #Logits CONV 3
        logits = tf.layers.conv2d_transpose(x3, output_dim, 5, strides=2, padding='same')
        #→32x32x3
        
        #ハイパボリックタンジェント
        out = tf.tanh(logits)
        
        return out


##Discriminator
def discriminator(x, reuse=False, alpha=0.2):
    with tf.variable_scope('discriminator', reuse=reuse):
    
        #第1層
        x1 = tf.layers.conv2d(x, 64, 5, strides=2, padding='same')
        x1 = tf.maximum(alpha * x1, x1)
        #→16x16x64
        
        #第2層
        x2 = tf.layers.conv2d(x1, 128, 5, strides=2, padding='same')
        x2 = tf.layers.batch_normalization(x2, training=True)
        x2 = tf.maximum(alpha * x2, x2)
        #→8x8x128
        
        #第3層
        x3 = tf.layers.conv2d(x2, 256, 5, strides=2, padding='same')
        x3 = tf.layers.batch_normalization(x3, training=True)
        x3 = tf.maximum(alpha * x3, x3)
        #→4x4x256

        #一次元変換
        flat = tf.reshape(x3, (-1,  4*4*256))
        #全結合
        logits = tf.layers.dense(flat, 1)
        #０～１の配列化
        out=tf.sigmoid(logits)
        
        return out, logits


##Loss(損失関数)
def model_loss(input_real, input_z, output_dim, alpha=0.2):
    g_model = generator(input_z, output_dim, alpha=alpha)
    d_model_real, d_logits_real = discriminator(input_real, alpha=alpha)
    d_model_fake, d_logits_fake = discriminator(g_model, reuse=True, alpha=alpha)
    
    #本物を真と見なす際のloss
    d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_real, labels=tf.ones_like(d_model_real)))
    #偽物を偽と見なす際のloss
    d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake, labels=tf.zeros_like(d_model_fake)))
    #偽物を真と見なす際のloss
    g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake, labels=tf.ones_like(d_model_fake)))


    d_loss = d_loss_real + d_loss_fake
    
    return d_loss, g_loss


##Optimize
def model_opt(d_loss, g_loss, learning_rate, beta1):
    #データセット格納
    t_vars = tf.trainable_variables()
    d_vars = [var for var in t_vars if var.name.startswith('discriminator')]
    g_vars = [var for var in t_vars if var.name.startswith('generator')]
    
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        #discriminatorの最適化処理
        d_train_opt = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(d_loss, var_list=d_vars)
        #generatorの最適化処理
        g_train_opt = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(g_loss, var_list=g_vars)
        
        return d_train_opt, g_train_opt


##Modelクラス・・★結果が思しくない場合は、↓beta1の値を増減することで調整を図る
class GAN:
    def __init__(self, real_size, z_size, learning_rate, alpha=0.2, beta1=0.5):
        #Reset
        tf.reset_default_graph()
        
        self.input_real, self.input_z = model_inputs(real_size, z_size)
        #Loss
        self.d_loss, self.g_loss = model_loss(self.input_real, self.input_z, real_size[2], alpha=alpha)
        #Optmize
        self.d_opt, self.g_opt = model_opt(self.d_loss, self.g_loss, learning_rate, beta1)


##生成した画像を表示する関数の定義
def view_samples(epoch, samples, nrows, ncols, figsize=(5,5)):
    fig, axes = plt.subplots(figsize=figsize, nrows=nrows, ncols=ncols, sharey=True, sharex=True)

    for ax, img in zip(axes.flatten(), samples[epoch]):

        ax.axis('off')
        img= ((img - img.min())*255 / (img.max() - img.min())).astype(np.uint8)
        ax.set_adjustable('box')
        im = ax.imshow(img, aspect='equal')
    
    plt.subplots_adjust(wspace=0, hspace=0)
    
    return fig, axes


##Training    #print_every・・途中経過　/　show_every・・生成した画像　を何件ごとに表示するか
def train(net, dataset, epochs, batch_size, print_every=10, show_every=100, figsize=(5,5)):
    #途中経過を保存
    saver = tf.train.Saver()
    #一様分布からノイズ作成
    sample_z = np.random.uniform(-1, 1, size=(72, z_size))
    
    #値の初期化
    samples, losses = [], []
    steps = 0

    with tf.Session() as sess:
        #値の初期化
        sess.run(tf.global_variables_initializer())
        #１epoch中
        for e in range(epochs):
            
            for x, y in dataset.batches(batch_size):
                steps += 1
                
                #batchのなかで使うノイズ
                batch_z = np.random.uniform(-1, 1, size=(batch_size, z_size))
                
                #ネットワークのdのoptmizerを実行
                _ = sess.run(net.d_opt, feed_dict={net.input_real: x, net.input_z: batch_z})
                _ = sess.run(net.g_opt, feed_dict={net.input_z: batch_z, net.input_real: x})
                
                #途中経過の表示（10件おきに表示）
                if steps % print_every == 0:

                    train_loss_d = net.d_loss.eval({net.input_z: batch_z, net.input_real: x})
                    train_loss_g = net.g_loss.eval({net.input_z: batch_z})

                    print("Epoch{}/{}:".format(e+1, epochs),
                         "D Loss: {:.4f} ".format(train_loss_d),
                         "G Loss: {:.4f} ".format(train_loss_g))

                    losses.append((train_loss_d, train_loss_g))

                #指定した回数置きに画像も表示
                if steps % show_every == 0:

                    gen_samples = sess.run(generator(net.input_z, 3, reuse=True, training=False),
                                                            feed_dict={net.input_z: sample_z})
                    samples.append(gen_samples)
                    #－１で可変長
                    _ =  view_samples(-1, samples, 6, 12, figsize=figsize)
                    #表示
                    plt.show()
                    
        #全25エポック終了後、パスを指定(無ければ作成)して学習結果を保存
        saver.save(sess, './checkpoints/generator.ckpt')
        
    with open('samples.pkl', 'wb') as f:
            pkl.dump(samples, f)
    
    return losses, samples


##ハイパーパラメータの初期化
#R,G,Bの3ch
real_size = (32, 32, 3)
z_size = 100
#↓★結果を見て増減させる
learning_late = 0.0002
batch_size = 128
epochs = 25
alpha = 0.2
#AdamOptimizerの減衰値
beta1 = 0.5
#GANモデル生成
net = GAN(real_size, z_size, learning_late, alpha=alpha, beta1=beta1)


##Training  Run
dataset = Dataset(trainset, testset)
losses, samples = train(net, dataset, epochs, batch_size,  figsize=(10,5))


##グラフ描画
fig, ax = plt.subplots(figsize=(24,8))
losses = np.array(losses)
#dのloss（オレンジの線）.αは透明度
plt.plot(losses.T[0], label='D', alpha=0.5)
#gのloss（青線）
plt.plot(losses.T[1], label='G', alpha=0.5)
plt.title('Training Loss')
plt.legend()