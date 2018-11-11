# Lab 3 Minimizing Cost
import tensorflow as tf
tf.set_random_seed(777)  # for reproducibility

# tf Graph Input
X = [1, 2, 3]
Y = [1, 2, 3]

# Set wrong model weights
W = tf.Variable(5.0)

# Linear model
hypothesis = X * W

# cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y)) #cost를 구함

# Minimize: Gradient Descent Magic
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)  #미분연산을 통해 점점 최소로
train = optimizer.minimize(cost) #이렇게 간다

# Launch the graph in a session.
sess = tf.Session()
# Initializes global variables in the graph.
sess.run(tf.global_variables_initializer())

for step in range(100):
    print(step, sess.run(W))
    sess.run(train)
#5.0에서 1.0까지

'''
0 5.0
1 1.26667
2 1.01778
3 1.00119
4 1.00008
...
96 1.0
97 1.0
98 1.0
99 1.0
'''
