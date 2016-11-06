import theano



def playaround():
    # f(x) = X^2, and compute f(-2)
    x = theano.tensor.scalar()
    y = x**2
    f = theano.function([x],y)

    print(f(-2))
    return 0



def main():
    return 0

main()