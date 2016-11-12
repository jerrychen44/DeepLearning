"""Softmax."""
import numpy as np
#scores = [3.0, 1.0, 0.2]
scores = [1.0, 2.0, 3.0]
'''
scores = np.array([[1, 2, 3, 6],
                   [2, 4, 5, 6],
                   [3, 8, 7, 6]])
'''

def softmax(y):
    """Compute softmax values for each sets of scores in x."""
    ey=np.exp(y)
    res=ey/ey.sum(axis=0)
    #print(res)
    return res # TODO: Compute and return softmax(x)


def main():
    global scores
    scores=np.array(scores)
    print(softmax(scores))
    ''' [ 0.09003057  0.24472847  0.66524096] '''
    print(softmax(scores*10))
    # will let output more close to 1 or 0, classify will become very confident
    ''' [  2.06106005e-09   4.53978686e-05   9.99954600e-01] '''
    print(softmax(scores/10))
    # will very close to normal disturtion, classify will become unsure
    '''[ 0.30060961  0.33222499  0.3671654 ]'''

    return 0


main()
