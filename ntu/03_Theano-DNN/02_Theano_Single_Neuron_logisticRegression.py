import theano
import theano.tensor as T
import random
import numpy



def single_neuron_example1():
    print("single neuron in intuitive way")
    #########################
    # Step1, Define input variable
    #########################
    x = T.vector() # feature vector
    w = T.vector() # weight vector
    b = T.scalar() # bias

    ###################
    # Step2: define output variables
    ###################
    z = T.dot(w,x) + b
    # chose activation function is sigmoid
    y = 1 / ( 1 + T.exp( -z ) )


    ###################
    # Step3: Declare Function
    ###################
    neuron = theano.function(inputs=[x,w,b], outputs=[y])


    # we assume that we have already get the weight vector and b already.
    w = [-1,1]
    b = 0
    # we fake 100 x data
    for i in range(100):
        x = [random.random(),random.random()]
        print(x)

        # apply the neuron to get y
        print(neuron(x,w,b))
        '''
        ........
        ........
        [0.9228112110662247, 0.8211311543967381]
        [array(0.474601864321836)]
        [0.8357217945717774, 0.17515431697410366]
        [array(0.34061214710120313)]
        [0.8067082563062566, 0.7214862284614993]
        [array(0.4787073784728632)]
        [0.7918145550126646, 0.19116994331059278]
        [array(0.3541962306262372)]
        [0.156087265840137, 0.03876595042427544]
        [array(0.47070326750236996)]
        [0.9769519701870543, 0.5259610487878122]
        [array(0.38912519145598684)]
        [0.19886855402474213, 0.33129417898043645]
        [array(0.5330581099304749)]
        '''
    return 0

def single_neuron_example2():
    print("single neuron in shared variables")
    #########################
    # Step1, Define input variable
    #########################
    x = T.vector() # feature vector

    w = theano.shared( numpy.array([1.,1.]) ) # weight vector with initial value [1,1]
    b = theano.shared(0.) # bias with initial value 0
    print("now w, b is a shared variables, not symbol. but x still is a symbol")
    ###################
    # Step2: define output variables
    ###################
    z = T.dot(w,x) + b
    # chose activation function is sigmoid to do nonlinear transform
    y = 1 / ( 1 + T.exp( -z ) )


    ###################
    # Step3: Declare Function
    ###################
    neuron = theano.function(inputs=[x], outputs=y)
    print("Although you don't set w, b to this function, but since they be set as shared variables above.")
    print("The function can access those shared variables.")

    # we assume that we have already get the weight vector and b already.
    # and use get_value, and set_value to access this shared variables
    #print(w.get_value())
    #w.set_value([0.,0.])

    # we fake 100 x data
    for i in range(100):
        #fake x data
        x = [random.random(),random.random()]
        print(x)

        # apply the neuron to get y
        print(neuron(x))
        '''
        ..........
        ..........
        [0.0005654614718300088, 0.9849420792112815]
        0.7281996648457243
        [0.09079115365570256, 0.08468515860521564]
        0.5437568557857444
        [0.0794266070793418, 0.8603689291061583]
        0.7190583548285042
        [0.18251658230123136, 0.5522709825853359]
        0.6758549902374171
        '''

    return 0

def single_neuron_training_example3():


    x = T.vector() # feature vector

    w = theano.shared( numpy.array([-1.,1.]) ) # weight vector with initial value [1,1]
    b = theano.shared(0.) # bias with initial value 0

    z = T.dot(w,x) + b
    # chose activation function is sigmoid to do nonlinear transform
    y = 1 / ( 1 + T.exp( -z ) )
    neuron = theano.function(inputs=[x], outputs=y)


    #Define a cost function
    y_hat = T.scalar() # y_hat is a real answear , y is the answear you predict by neuron above
    cost = T.sum((y-y_hat)**2) # cost function

    # coumputing the Grandients
    dw,db = T.grad(cost,[w,b]) # dw is the vaule after dcost/ dw


    #################
    # Do gradient Descent on Single Neuron
    #################
    ######## Method 1 : weak solution ##############
    '''
    # tedious way, weak solution
    # this function will be use for get the Partial differential later.
    gradient= theano.function(inputs=[x,y_hat],outputs=[dw,db])


    x =[ 1, -1]
    y_hat = 1
    for i in range(100):
        print(neuron(x))
        # compute gradient
        dw,db = gradient(x,y_hat)
        # because w, b is shared variables, we need to use
        # set_value, get_value to access them.
        # updated the variables w, b. learning rate here is 0.1
        w.set_value (w.get_value()- 0.1*dw)
        b.set_value (b.get_value()- 0.1*db)

        print(w.get_value(),b.get_value())

    '''
    ######## Method 2 : Effective Way ##############
    '''
    # Effective Way. Use updated in function. (accepte a list of pairs)
    # form is (shared-variable name, an updates expression)
    # everytime you call the gradient function, the left shared-variable will be replace
    # by the right result.
    gradient= theano.function(inputs=[x,y_hat], updates = [(w,w-0.1*dw),(b,b-0.1*db)])

    x =[ 1, -1]
    y_hat = 1
    for i in range(100):
        print(neuron(x))
        # compute gradient
        gradient(x,y_hat) #
        print(w.get_value(),b.get_value())

    '''

    ######## Method 3 : Sophisticated Way, append a function. ##############
    # you can redirect all output to the function you want.
    # because something the update paramters function is complex, you can not
    # just use one line like (w,w-0.1*dw) in updates = , so you need handle them
    # in a fun.
    gradient= theano.function(inputs=[x,y_hat], updates = MyUpdate([w,b],[dw,db]) )
    x =[ 1, -1]
    y_hat = 1
    for i in range(100):
        print(neuron(x))
        # compute gradient
        gradient(x,y_hat) #
        print(w.get_value(),b.get_value())


    return 0

def MyUpdate(paramters,gradients):
    #you can set anything you want here.
    #we still use original gradient descent for example
    mu=0.1 #learning rate
    parameters_updates = [ (p, p-mu*g) for p, g in zip(paramters, gradients)]
    return parameters_updates

def main():
    #single_neuron_example1()
    #single_neuron_example2()
    single_neuron_training_example3()


    return 0


main()
