# Lab 6 Softmax Classifier
import tensorflow as tf
import numpy as np

'''
Sigmoid?
2.0, 1.0, 0.1이 y값으로 나왔을때
0.7, 0.2, 0.1로 만들어준다.

softmax함수는 2.0,1.0,0.1 > 0.7,0.2,0.1로 만들어준다.
합이 1로 된다. 모든 값은 0 ~ 1 사이
즉 확률로 된다 라고 보면 된다.

그중에 하나만 골라줘 > one-hot-encoding
argMax로 사용한다.

cross-entropy
예측한 값과 실제의 값이 얼마나 차이가 나는지 확인하는 cost Function을 만들자.
D(S,L) = -sum(Llog(S))
S = Yhat
L = Y

Sum(Y * Sum-log(YHati)) 여기서 YHati는 0~1사이이다. softmax를 통과했기 때문
답 : 0 1
예측1 : 0 1 > OK > [0,1] * -log[0,1] = [0,1] * [무한,0] > 최종 코스트 : [0,  0] => 0
예측2 : 1 0 > No > [0,1] * -log[1,0] = [0,1] * [0,무한] > 최종 코스트 : [0,무한] => 무한

Logistic Regression >
C(H(x),y) = ylog(H(x)) - (1-y)log(1-H(x))
Cost Function >
1/N(Sum(D(S(Wx+b),Li)))

'''

tf.set_random_seed(777)  # for reproducibility

# Predicting animal type based on various features
xy = np.loadtxt('data-04-zoo.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

print(x_data.shape, y_data.shape)

nb_classes = 7  # 0 ~ 6

X = tf.placeholder(tf.float32, [None, 16])
Y = tf.placeholder(tf.int32, [None, 1])  # 0 ~ 6
Y_one_hot = tf.one_hot(Y, nb_classes)  # one hot
print("one_hot", Y_one_hot)
Y_one_hot = tf.reshape(Y_one_hot, [-1, nb_classes])
print("reshape", Y_one_hot)

W = tf.Variable(tf.random_normal([16, nb_classes]), name='weight')
b = tf.Variable(tf.random_normal([nb_classes]), name='bias')

# tf.nn.softmax computes softmax activations
# softmax = exp(logits) / reduce_sum(exp(logits), dim)
logits = tf.matmul(X, W) + b
hypothesis = tf.nn.softmax(logits)

# Cross entropy cost/loss
cost_i = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,
                                                 labels=tf.stop_gradient([Y_one_hot]))
cost = tf.reduce_mean(cost_i)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

prediction = tf.argmax(hypothesis, 1)
correct_prediction = tf.equal(prediction, tf.argmax(Y_one_hot, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# Launch graph
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2000):
        sess.run(optimizer, feed_dict={X: x_data, Y: y_data})
        if step % 100 == 0:
            loss, acc = sess.run([cost, accuracy], feed_dict={
                                 X: x_data, Y: y_data})
            print("Step: {:5}\tLoss: {:.3f}\tAcc: {:.2%}".format(
                step, loss, acc))

    # Let's see if we can predict
    pred = sess.run(prediction, feed_dict={X: x_data})
    # y_data: (N,1) = flatten => (N, ) matches pred.shape
    for p, y in zip(pred, y_data.flatten()):
        print("[{}] Prediction: {} True Y: {}".format(p == int(y), p, int(y)))

'''
Step:     0 Loss: 5.106 Acc: 37.62%
Step:   100 Loss: 0.800 Acc: 79.21%
Step:   200 Loss: 0.486 Acc: 88.12%
Step:   300 Loss: 0.349 Acc: 90.10%
Step:   400 Loss: 0.272 Acc: 94.06%
Step:   500 Loss: 0.222 Acc: 95.05%
Step:   600 Loss: 0.187 Acc: 97.03%
Step:   700 Loss: 0.161 Acc: 97.03%
Step:   800 Loss: 0.140 Acc: 97.03%
Step:   900 Loss: 0.124 Acc: 97.03%
Step:  1000 Loss: 0.111 Acc: 97.03%
Step:  1100 Loss: 0.101 Acc: 99.01%
Step:  1200 Loss: 0.092 Acc: 100.00%
Step:  1300 Loss: 0.084 Acc: 100.00%
...
[True] Prediction: 0 True Y: 0
[True] Prediction: 0 True Y: 0
[True] Prediction: 3 True Y: 3
[True] Prediction: 0 True Y: 0
[True] Prediction: 0 True Y: 0
[True] Prediction: 0 True Y: 0
[True] Prediction: 0 True Y: 0
[True] Prediction: 3 True Y: 3
[True] Prediction: 3 True Y: 3
[True] Prediction: 0 True Y: 0
'''
