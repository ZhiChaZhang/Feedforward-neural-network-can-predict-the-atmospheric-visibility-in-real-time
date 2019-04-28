# Import
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
# Import data
data = pd.read_csv('C:/Users/Administrator/Desktop/Testfile.csv')  

# Drop date variable
#data = data.drop(['DATE'], 1) #

# Dimensions of dataset
n = data.shape[0]
p = data.shape[1]

# Make data a np.array
data = data.values

# Training and test data
train_start = 0
train_end = int(np.floor(2/3*n))
test_start = train_end + 1
test_end = n
data_train = data[np.arange(train_start, train_end), :]
data_test = data[np.arange(test_start, test_end), :]
length=len(data_test)
# Scale data
scaler = MinMaxScaler(feature_range=(0, 1)) #feature_range=(0, 1) 表示归一化处理的范围是0，1
#归一的尺度确定，为scaler
scaler.fit(data_train)  #归一化必须按照列来,这里的max和min就是train 里的数据，因为scaler的目标是train
data_train = scaler.transform(data_train) 
data_test = scaler.transform(data_test)

#scaler.inverse_transform(max(pred))

# Build X and y
X_train = data_train[:, 1:]
y_train = data_train[:, 0]
X_test = data_test[:, 1:]
y_test = data_test[:, 0]

# Number of stocks in training data
n_stocks = X_train.shape[1]

# Neurons
n_neurons_1 = 1024*2
n_neurons_2 = 512*2
n_neurons_3 = 256*2
n_neurons_4 = 128*2
n_neurons_5 = 128*2
# Session
net = tf.InteractiveSession() #这个每net.run一次，才会执行一次带有tf.xxx的代码，因为
#tf.xxx是tensorflow给定的一个结构，只能在Session下运行

# Placeholder
X = tf.placeholder(dtype=tf.float32, shape=[None, n_stocks]) #None默认为1,n_stocks为第二维
#给占位符，类似于腾出空间
Y = tf.placeholder(dtype=tf.float32, shape=[None])

# Initializers
sigma = 1
weight_initializer = tf.variance_scaling_initializer(mode="fan_avg", distribution="uniform", scale=sigma)
#使用distribution="uniform",样本从[-limit, limit],其中limit = sqrt(3 * scale / n),内的均匀分布中抽取.
#如果mode =“fan_avg”,n为平均输入的数量和输出单元的数量. 
#scale：比例因子(正浮点数)
#表示了生成一个[-limit, limit],其中limit = sqrt(3 * 1 / n)的随机数。
bias_initializer = tf.zeros_initializer()  # 0矩阵的初始化，让偏置为0

# Hidden weights
W_hidden_1 = tf.Variable(weight_initializer([n_stocks, n_neurons_1])) #weight_里表示生成m*n的初始权重，这个分布是
bias_hidden_1 = tf.Variable(bias_initializer([n_neurons_1]))
W_hidden_2 = tf.Variable(weight_initializer([n_neurons_1, n_neurons_2]))
bias_hidden_2 = tf.Variable(bias_initializer([n_neurons_2]))
W_hidden_3 = tf.Variable(weight_initializer([n_neurons_2, n_neurons_3]))
bias_hidden_3 = tf.Variable(bias_initializer([n_neurons_3]))
W_hidden_4 = tf.Variable(weight_initializer([n_neurons_3, n_neurons_4]))
bias_hidden_4 = tf.Variable(bias_initializer([n_neurons_4]))
W_hidden_5 = tf.Variable(weight_initializer([n_neurons_4, n_neurons_5]))
bias_hidden_5 = tf.Variable(bias_initializer([n_neurons_5]))
#偏置加在权重之后，

# Output weights
W_out = tf.Variable(weight_initializer([n_neurons_5, 1]))
bias_out = tf.Variable(bias_initializer([1]))

# Hidden layer   这里使用非常常用的relu整流性单元
hidden_1 = tf.nn.relu(tf.add(tf.matmul(X, W_hidden_1), bias_hidden_1))
#这里就是隐藏层里运算过程，偏置加在x*w_hidden_1之后，然后放进relu这个激活函数里，计算隐藏层
hidden_2 = tf.nn.relu(tf.add(tf.matmul(hidden_1, W_hidden_2), bias_hidden_2))
hidden_3 = tf.nn.relu(tf.add(tf.matmul(hidden_2, W_hidden_3), bias_hidden_3))
hidden_4 = tf.nn.relu(tf.add(tf.matmul(hidden_3, W_hidden_4), bias_hidden_4))
hidden_5 = tf.nn.relu(tf.add(tf.matmul(hidden_4, W_hidden_5), bias_hidden_5))
# Output layer (transpose!)
out = tf.transpose(tf.add(tf.matmul(hidden_5, W_out), bias_out)) #进行转置

# Cost function
mse = tf.reduce_mean(tf.squared_difference(out, Y))

# Optimizer
opt = tf.train.AdamOptimizer().minimize(mse)

# Init
net.run(tf.global_variables_initializer())

# Setup plot
plt.ion()
fig = plt.figure()
ax1 = fig.add_subplot(111)
line1, = ax1.plot(y_test)
line2, = ax1.plot(y_test * 0.5)
plt.show()

# Fit neural net
batch_size = 30
mse_train = []
mse_test = []

# Run
epochs = 30
for e in range(epochs): #一次训练结束就完成一次epochs

    # Shuffle training data
    shuffle_indices = np.random.permutation(np.arange(len(y_train)))
    X_train = X_train[shuffle_indices]
    y_train = y_train[shuffle_indices]

    # Minibatch training,每一次训练结束都会自己进行调整，在这里对某一个批次的样本进行一次训练，然后再怼第二次进行训练
    #兼顾第一次和下一次训练的稳定性，因此分批次是达到最准确的可取手段之一。
    #进行一次Optimizer就会兼顾前后，调整方向和下降梯度。
    for i in range(0, len(y_train) // batch_size):
        start = i * batch_size
        batch_x = X_train[start:start + batch_size]
        batch_y = y_train[start:start + batch_size]
        # Run optimizer with batch
        net.run(opt, feed_dict={X: batch_x, Y: batch_y})  #feed_dict is given the value of placeholder，进行一次优化
        #每 batch_size的数据为一个批次，便进行一次优化,这里的优化肯定是不断传递给后面的训练
        # Show progress，用5个批次，输出一次数据
        #pred = net.run(out, feed_dict={X: X_test})    #如何下面的if不执行，那么这里输入一个批次，就会进行一次预测
        #1个i就进行一次预测，相当于i//1==0
        #print(pred,i)
        if np.mod(i, 100) == 0:    #mod四舍五入区
            #MSE train and test
            mse_train.append(net.run(mse, feed_dict={X: X_train, Y: y_train})) #mse_train和net.run的运行结果合在一起
            mse_test.append(net.run(mse, feed_dict={X: X_test, Y: y_test}))
            print('MSE Train: ', mse_train[-1])
            print('MSE Test: ', mse_test[-1])
            # Prediction
            pred = net.run(out, feed_dict={X: X_test})
            #line2.set_ydata(pred)
            #plt.title('Epoch ' + str(e) + ', Batch ' + str(i))
            #file_name = 'C:/Users/Administrator/Desktop' + str(e) + '_batch_' + str(i) + '.png'
            #plt.savefig('C:/Users/Administrator/Desktop/file_name')
            #plt.pause(0.01)
mse_final = net.run(mse, feed_dict={X: X_test, Y: y_test})
print(mse_final)
maxgy=max(data[np.arange(train_start, train_end), 0])
mingy=min(data[np.arange(train_start, train_end), 0])
predictioninverse=max(pred)*(maxgy-mingy)+mingy
actual=y_test.reshape(length,1)*(maxgy-mingy)+mingy
plt.plot(y_test.reshape(1,length)*(maxgy-mingy)+mingy,predictioninverse.reshape(1,length),'bo')
plt.savefig('C:/Users/Administrator/Desktop/file_name')