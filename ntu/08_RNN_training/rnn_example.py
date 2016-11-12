# source from http://speech.ee.ntu.edu.tw/~tlkagk/courses/MLDS_2015_2/theano/rnn.example.py
# modify from https://github.com/Lasagne/Lasagne/blob/master/examples/recurrent.py
import theano
import theano.tensor as T
import numpy as np
import sys
from itertools import izip
import time

# Min/max sequence length
MIN_LENGTH = 50
MAX_LENGTH = 55
# Number of units in the hidden (recurrent) layer
N_HIDDEN = 100
# input
N_INPUT = 2
# output
N_OUTPUT = 1

# from https://github.com/Lasagne/Lasagne/blob/master/examples/recurrent.py
def gen_data(min_length=MIN_LENGTH, max_length=MAX_LENGTH):
    '''
    Generate a sequences for the "add" task, e.g. the target for the
    following
    ``| 0.5 | 0.7 | 0.3 | 0.1 | 0.2 | ... | 0.5 | 0.9 | ... | 0.8 | 0.2 |
      |  0  |  0  |  1  |  0  |  0  |     |  0  |  1  |     |  0  |  0  |``
    would be 0.3 + .9 = 1.2.  This task was proposed in [1]_ and explored in
    e.g. [2]_.
    Parameters
    ----------
    min_length : int
        Minimum sequence length.
    max_length : int
        Maximum sequence length.
    Returns
    -------
    X : np.ndarray
        Input to the network, of shape (length, 2), where the last
        dimension corresponds to the two sequences shown above.
    y : np.ndarray
        Correct output for each sample (a scalar).
    '''
    # Generate x_seq
    length = np.random.randint(min_length, max_length)
    x_seq = np.concatenate([np.random.uniform(size=(length, 1)),
                        np.zeros((length, 1))],
                       axis=-1)
    # Set the second dimension to 1 at the indices to add
    x_seq[np.random.randint(length/10), 1] = 1
    x_seq[np.random.randint(length/2, length), 1] = 1
    # Multiply and sum the dimensions of x_seq to get the target value
    y_hat = np.sum(x_seq[:, 0]*x_seq[:, 1])
    return x_seq, y_hat

# what can we get from gen_data()
#print gen_data()

x_seq = T.matrix('input')
y_hat = T.scalar('target')

Wi = theano.shared( np.random.randn(N_INPUT,N_HIDDEN) )
bh = theano.shared( np.zeros(N_HIDDEN) )
Wo = theano.shared( np.random.randn(N_HIDDEN,N_OUTPUT) )
bo = theano.shared( np.zeros(N_OUTPUT) )
Wh = theano.shared( np.random.randn(N_HIDDEN,N_HIDDEN) )
parameters = [Wi,bh,Wo,bo,Wh]

def sigmoid(z):
        return 1/(1+T.exp(-z))

def step(x_t,a_tm1,y_tm1):
        a_t = sigmoid( T.dot(x_t,Wi) \
                + T.dot(a_tm1,Wh) + bh )
        y_t = T.dot(a_t,Wo) + bo
        return a_t, y_t

a_0 = theano.shared(np.zeros(N_HIDDEN))
y_0 = theano.shared(np.zeros(N_OUTPUT))

[a_seq,y_seq],_ = theano.scan(
                        step,
                        sequences = x_seq,
                        outputs_info = [ a_0, y_0 ],
			truncate_gradient=-1
                )

y_seq_last = y_seq[-1][0] # we only care about the last output
cost = T.sum( ( y_seq_last - y_hat )**2 )

gradients = T.grad(cost,parameters)

def MyUpdate(parameters,gradients):
	mu =  np.float32(0.001)
	parameters_updates = [(p,p - mu * g) for p,g in izip(parameters,gradients) ]
	return parameters_updates

rnn_test = theano.function(
        inputs= [x_seq],
        outputs=y_seq_last
)

rnn_train = theano.function(
        inputs=[x_seq,y_hat],
        outputs=cost,
	updates=MyUpdate(parameters,gradients)
)

for i in range(10000000):
	x_seq, y_hat = gen_data()
	print "iteration:", i, "cost:",  rnn_train(x_seq,y_hat)

for i in range(10):
	x_seq, y_hat = gen_data()
	print "reference", y_hat, "RNN output:", rnn_test(x_seq)
