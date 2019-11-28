from scipy import linalg as LA 
import numpy as np


def main():
    W = np.array([[1,3/4,1/3,0,1/3],[3/4,1,1/4,1/2,0],[1/3,1/4,1,4/5,1/5],[0,1/2,4/5,1,0],[1/3,0,1/5,0,1]])
    D = np.zeros((5,5))
    D[0,0] = 29/12.0
    D[1,1] = 5/2
    D[2,2] = 31/12
    D[3,3] = 23/10
    D[4,4] = 23/15
    print(W)
    print(D)
    L = D-W
    # print('L')
    print(L)
    e,v = LA.eigh(L, eigvals = (0,2))
    print('e:{}'.format(e))
    print('v:{}'.format(v))

if __name__ == '__main__':
    main()