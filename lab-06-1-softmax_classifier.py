# Lab 6 Softmax Classifier

'''
WX의 문제 : 3개 이상의 대상을 분별하는것에 적합하지 않았다. (100,200,-10)
g(z) = 0이나 1사이의 값이 나오면 좋겠다.
연구시 나온 것이 시그모이드 (1/(1+e^-2)) = Logistic
이 시그모이드 함수를 이용해 1과 0으로 나누었다.
H(x) = WX
> H(x) = g(H(x))

X,W -> z > 시그모이드 -> Y [0,1]
Y : Real
YHat : Predict : H(x)

두개를 구분하는 선을 찾아내는 것이 Logistic Regression
Multinomial Classifcation : 여러개의 Class가 있고 (A,B,C가 점수,출석,시간별로 다르게 분류된다) 이를 분류해내는것
하나는 C이거나 C가 아니거나를 Check (binary classfiation)
또 하나는 B이거나 B가 아니거나를 Check (binary classfication)
또 하나는 A이거나 A가 아니거나를 Check (binary classfication)
3개의 binary classfication으로 분류가 가능한것인데,
X - AClassifier - YHat
  - BClassifier - YHat
  - CClassifier - YHat

w1x1 + w2x2 + w3x3
이걸 A, B, C 각각에서 독립적으로 계산한다.
그래서 이걸 하나로 합쳐,
WA1 WA2 WA3   X1
WB1 WB2 WB3 * X2 = [3,1] > YHatA, YHatB, YHatC
WC1 WC2 WC3   X3



'''


import tensorflow as tf
tf.set_random_seed(777)  # for reproducibility

x_data = [[1, 2, 1, 1],
          [2, 1, 3, 2],
          [3, 1, 3, 4],
          [4, 1, 5, 5],
          [1, 7, 5, 5],
          [1, 2, 5, 6],
          [1, 6, 6, 6],
          [1, 7, 7, 7]]
y_data = [[0, 0, 1],
          [0, 0, 1],
          [0, 0, 1],
          [0, 1, 0],
          [0, 1, 0],
          [0, 1, 0],
          [1, 0, 0],
          [1, 0, 0]]

X = tf.placeholder("float", [None, 4])
Y = tf.placeholder("float", [None, 3])
nb_classes = 3

W = tf.Variable(tf.random_normal([4, nb_classes]), name='weight')
b = tf.Variable(tf.random_normal([nb_classes]), name='bias')

# tf.nn.softmax computes softmax activations
# softmax = exp(logits) / reduce_sum(exp(logits), dim)
hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)

# Cross entropy cost/loss
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

# Launch graph
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2001):
        sess.run(optimizer, feed_dict={X: x_data, Y: y_data})
        if step % 200 == 0:
            print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}))

    print('--------------')

    # Testing & One-hot encoding
    a = sess.run(hypothesis, feed_dict={X: [[1, 11, 7, 9]]})
    print(a, sess.run(tf.argmax(a, 1)))

    print('--------------')

    b = sess.run(hypothesis, feed_dict={X: [[1, 3, 4, 3]]})
    print(b, sess.run(tf.argmax(b, 1)))

    print('--------------')

    c = sess.run(hypothesis, feed_dict={X: [[1, 1, 0, 1]]})
    print(c, sess.run(tf.argmax(c, 1)))

    print('--------------')

    all = sess.run(hypothesis, feed_dict={
                   X: [[1, 11, 7, 9], [1, 3, 4, 3], [1, 1, 0, 1]]})
    print(all, sess.run(tf.argmax(all, 1)))

'''
--------------
[[  1.38904958e-03   9.98601854e-01   9.06129117e-06]] [1]
--------------
[[ 0.93119204  0.06290206  0.0059059 ]] [0]
--------------
[[  1.27327668e-08   3.34112905e-04   9.99665856e-01]] [2]
--------------
[[  1.38904958e-03   9.98601854e-01   9.06129117e-06]
 [  9.31192040e-01   6.29020557e-02   5.90589503e-03]
 [  1.27327668e-08   3.34112905e-04   9.99665856e-01]] [1 0 2]
'''
