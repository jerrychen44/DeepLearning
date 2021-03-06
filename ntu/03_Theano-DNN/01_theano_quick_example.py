import theano
import theano.tensor as T


def playaround():
    # f(x) = X^2, and compute f(-2)
    x = theano.tensor.scalar()
    y = x**2
    f = theano.function([x],y)

    print(f(-2))



    ###############
    # 2 way to define the symbol
    ################
    a = theano.tensor.scalar()
    b = theano.tensor.matrix()

    # 'ha ha ha ' is the name of martic c
    c = theano.tensor.matrix('ha ha ha')

    print(a,b,c)

    #much easier
    a = T.scalar()
    b = T.matrix()
    c = T.matrix("ha ha ha")
    print(a,b,c)





    return 0

def sop_example():


    #########################
    # Step1, Define input variable
    #########################
    x1 = T.scalar()
    x2 = T.scalar()
    x3 = T.matrix()
    x4 = T.matrix()
    ###################
    # Step2: define output variables
    ###################
    y1 = x1 + x2
    y2 = x1 * x2
    y3 = x3 * x4 # two martix use * means to do "elementwise" operation.
    y4 = T.dot(x3, x4) # do the real matrix mutiplication

    ###################
    # Step3: Declare Function
    ###################
    # functin input x should be a list in python. so you see the [ ]
    f = theano.function([x],y)
    #f = theano.function(inputs=[x],outputs=y)



    return 0


def sop_example2():

    #########################
    # Step1, Define input variable
    #########################
    x1=T.scalar()
    x2=T.scalar()

    ###################
    # Step2: define output variables
    ###################
    y1=x1*x2
    y2=x1**2+x2**0.5

    ###################
    # Step3: Declare Function
    ###################
    f= theano.function([x1,x2],[y1,y2])

    ###################
    # Step4: Use Function
    ###################
    z=f(2,4)
    print(z)
    '''[array(8.0), array(6.0)]'''
    ''' y1 = 8.0, y2 = 6'''


    return 0


def sop_example3_matrix():
    #########################
    # Step1, Define input variable
    #########################
    a = T.matrix()
    b = T.matrix()

    ###################
    # Step2: define output variables
    ###################
    c = a * b
    d= T.dot(a, b)

    ###################
    # Step3: Declare Function
    ###################
    F1 = theano.function([a,b],c)
    F2 = theano.function([a,b],d)

    print("So far, the matrix a,b,c,d are only symbol. No value in it.")
    ###################
    # Step4: Use Function
    ###################
    A = [[1,2],[3,4]] # A is a 2x2 matrix
    B = [[2,4],[6,8]] # B is a 2x2 matrix
    C = [[1,2],[3,4],[5,6]] # C is 3x2 matrix

    print(F1(A,B))
    '''
    [[  2.   8.]
     [ 18.  32.]] 2x2
     '''
    print(F2(C,B))
    '''
    [[ 14.  20.]
     [ 30.  44.]
     [ 46.  68.]] 3x2
     '''

    return 0

def gradient_example1():
    x = T.scalar('x')
    y = x**2
    g= T.grad(y,x) # do the dy/dx. y should be a scalar


    f = theano.function([x],y)
    f_prime = theano.function([x],g)

    print(f(-2))
    '''4.0'''
    print(f_prime(-2))
    '''-4.0'''

    return 0

def gradient_example2():
    #########################
    # Step1, Define input variable
    #########################
    x1 = T.scalar()
    x2 = T.scalar()

    ###################
    # Step2: define output variables
    ###################
    y =x1 * x2
    g = T.grad(y, [x1,x2]) # do the dy/dx. y should be a scalar
    # g will get a list, and first one is the result of dy/dx1, second one
    # is dy/dx2, in this case which means g = [x2,x1]


    f = theano.function([x1, x2], y)
    f_prime = theano.function([x1, x2], g)

    print(f(2,4))
    '''8.0'''
    print(f_prime(2,4))
    '''[array(4.0), array(2.0)]'''

    return 0




def gradient_example3():
    A = T.matrix()
    B = T.matrix()


    #       a1  a2           b1 b2
    #   A = a3  a4     ,   B=b3 b4
    #   2x2                 2x2
    #
    #   C=  a1b1 a2b2
    #       a3b3 a4b4   2x2
    C = A * B

    #
    #   D= a1b1+a2b2+a3b3+a4b4
    #
    D = T.sum(C)

    #note: the first parameter of grad(,), must be a scalar, so you can use D
    # but can not put C
    g = T.grad(D,A)
    # D, A are all matrix, so g is a matrix too.
    # use the element A to di D,
    #
    #   g= |dD/da1 dD/da2| =|b1 b2|
    #      |dD/da3 dD/da4|  |b3 b4|
    #

    y_prime = theano.function([A, B], g)



    A = [[1,2],[3,4]]
    B = [[2,4],[6,8]]
    print( y_prime(A, B))
    '''
    [[ 2.  4.]
     [ 6.  8.]]
     '''
    return 0

def main():

    playaround()

    sop_example1()

    sop_example2()

    sop_example3_matrix()

    gradient_example1()

    gradient_example2()

    gradient_example3()


    return 0

main()
