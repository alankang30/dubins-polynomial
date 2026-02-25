# reimplement optimized 2D planning 
# understand it - have the paper pulled up on the side
# think of ways to improve it
# maybe clean it

import sympy
import numpy as np

def find_Q(deriv, poly_deg, n_legs):
    '''
    Q is the cost matrix
    @param deriv: for cost J, 0=position, 1=velocity, 2=acceleration, 3=jerk
    @param poly_deg: degree of polynomial
    @param n_legs: number of legs in trajectory (num. waypoints - 1)

    @return Q matrix for cost J = p^T Q p
    '''
    k, l, m, n, n_c, n_l = sympy.symbols('k, l, m, n, n_c, n_l', integer=True)
    # k summation dummy variable
    # n deg of polynomial
    # 

    beta = sympy.symbols('beta')  # scaled time on leg, 0-1
    c = sympy.MatrixSymbol('c', n_c, 1)  # coefficient matrices, length is n+1, must be variable (n_c)
    sympy.display(c)
    T = sympy.symbols('T')  # time of leg
    P = sympy.summation(c[k, 0]*sympy.factorial(k)/sympy.factorial(k-m)*beta**(k-m)/T**m, (k, m, n))  # polynomial derivative
    P = P.subs({m: deriv, n: poly_deg}).doit()
    J = sympy.integrate(P**2, (beta, 0, 1)).doit()  # cost
    p = sympy.Matrix([c[i, 0] for i in range(poly_deg+1)])  # vector of terms
    Q = sympy.Matrix([J]).jacobian(p).jacobian(p)/2  # find Q using second derivative
    assert (p.T@Q@p)[0, 0].expand() == J  # assert hessian matches cost
    
    Ti = sympy.MatrixSymbol('T', n_l, 1)
    return sympy.diag(*[
        Q.subs(T, Ti[i]) for i in range(n_legs) ])



def find_A():
    pass

def find_A_cont(deriv, poly_deg, n_legs, leg):
    '''
    needed for continuity correction, so that we can have continuous roll angle.
    '''    
    k, m, n, n_c, n_l = sympy.symbols('k, m, n, n_c, n_l', integer=True)
    # k summation dummy variable
    # n deg of polynomial

    c = sympy.MatrixSymbol('c', n_c, n_l)  # coefficient matrices, length is n+1, must be variable (n_c)
    T = sympy.MatrixSymbol('T', n_l, 1)  # time of leg
    
    p = sympy.Matrix([c[i, l] for l in range(n_legs) for i in range(poly_deg+1) ])  # vector of terms

    beta0 = 1
    beta1 = 0
    P = sympy.summation(
        c[k, leg]*sympy.factorial(k)/sympy.factorial(k-m)*beta0**(k-m)/T[leg]**m
        - c[k, leg + 1]*sympy.factorial(k)/sympy.factorial(k-m)*beta1**(k-m)/T[leg+1]**m, (k, m, n))  # polynomial derivative
    P = P.subs({m: deriv, n: poly_deg}).doit()
    A_row = sympy.Matrix([P]).jacobian(p)
    b_row = sympy.Matrix([0])
    return A_row, b_row

def find_cost_function(poly_deg=5, min_deriv=3, rowsf_x=[0, 1, 2, 3, 6, 7, 8, 9], rowsf_y=[0, 1, 2, 3, 6, 7, 8, 9], n_legs=2):
    Q = find_Q(deriv=min_deriv, poly_deg=poly_deg, n_legs=n_legs)


def run_trajectory():
    '''
    actually optimizes our cost function to get a trajectory
    '''
    #scipy.optimize.minimize(something)
    pass