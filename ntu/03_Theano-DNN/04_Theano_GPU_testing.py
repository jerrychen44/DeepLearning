import os
# change to GPU
#os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu,floatX=float32"

import theano
import theano.tensor as T
import random
import numpy,time



def timer():

    print(theano.config.device)
    X = T.matrix()
    Y = T.matrix()
    Z = T.dot(X,Y)
    f = theano.function([X,Y],Z)



    x = numpy.random.randn(10000,10000)
    y = numpy.random.randn(10000,10000)

    tStart = time.time()
    z = f(x,y)
    tEnd =time.time()

    print("delta time =%s"%(tEnd-tStart))
    ''' delta time =131.8805868625641 '''
    return 0


def main():
    print(theano.config.device)
    timer()
    print("so far, because my mac doesn't have Nvidia GPU, so it got error")
    #ERROR (theano.sandbox.cuda): nvcc compiler not found on $PATH. Check your nvcc installation and try again.



    return 0
main()
