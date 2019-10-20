"""TODO: docstring
"""

import  numpy as np
import cvxpy as cp


def min_loss(alpha_init,C_2,C_3,R,dist):

    T = len(alpha_init)


    alpha = cp.Variable(T)


    loss  = R * alpha.T  + C_2 * cp.norm(alpha) + C_3 * dist * alpha.T

    # , cp.sum(const2 * alpha.T) >= cp.sum(const1 * alpha.T)
    prob = cp.Problem(cp.Minimize(loss),[cp.sum(alpha) == 1, alpha >= 0])

    prob.solve()



    return  alpha.value


def min_alphacvx(alpha_m, C_2, C_3, R, dist):
    """

    Function for optimizing w.r.t alpha (matrix form), reusing the function min_loss T times

    :param alpha_m:   matrix form of alpha, matrix T*T, with each col alpha_t, alpha_m = [alpha_1,alpha_2,\dots,alpha_T]
    :param C_2:       vector 1*T, with each value C_2 (cst) or different values
    :param C_3:       vector 1*T, with each value C_3 (cst) or different values
    :param R:         matrix form of empirical error, T*T, with each col R_t=[R_1(h_t),\dots,R_T(h_t)], R = [R_1,\dots,R_T]
    :param dist:      matrix from of empirical similarity, T*T, with each col dist_t = [dist(D_t,D_1),\dots,dist(D_t,D_T)], dist = [dist_1,\dots,dist_T]
    :return:

    """
    T = alpha_m.shape[0]
    # alpha = np.zeros([T,T])

    for i in range(T):

        alpha_t = alpha_m[:,i]
        R_t     = R[:,i]
        dist_t  = dist[:,i]
        alpha_m[:,i] = np.around(min_loss(alpha_t,C_2[i],C_3[i],R_t,dist_t),decimals=3)
        #print ('optimum alpha for task  '+str(i)+' = '+str(alpha_m[:,i]))

    return alpha_m

