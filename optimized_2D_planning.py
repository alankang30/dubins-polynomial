import sympy
import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors

def find_Q(deriv, poly_deg, n_legs):
    """
    Finds the cost matrix Q
    @param deriv: for cost J, 0=position, 1=velocity, etc.
    @param poly_deg: degree of polynomial
    @n_legs: number of legs in trajectory (num. waypoints - 1)
    @return Q matrix for cost J = p^T Q p
    """
    k, l, m, n, n_c, n_l = sympy.symbols('k, l, m, n, n_c, n_l', integer=True)
    # k summation dummy variable
    # n deg of polynomial

    beta = sympy.symbols('beta')  # scaled time on leg, 0-1
    c = sympy.MatrixSymbol('c', n_c, 1)  # coefficient matrices, length is n+1, must be variable (n_c)
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

def find_A(deriv, poly_deg, beta, n_legs, leg, value):
    """
    Finds rows of constraint matrix for setting value of trajectory and its derivatives
    @param deriv: the derivative that you would like to set, 0=position, 1=vel etc.
    @param poly_deg: degree of polynomial
    @param beta: 0=start of leg, 1=end of leg
    @n_legs: number of legs in trajectory (num. waypoints - 1)
    @leg: current leg
    @value: value of deriv at that point
    @return A_row, b_row
    """
    k, m, n, n_c, n_l = sympy.symbols('k, m, n, n_c, n_l', integer=True)
    # k summation dummy variable
    # n deg of polynomial

    c = sympy.MatrixSymbol('c', n_c, n_l)  # coefficient matrices, length is n+1, must be variable (n_c)

    T = sympy.MatrixSymbol('T', n_l, 1)  # time of leg
    
    p = sympy.Matrix([c[i, l] for l in range(n_legs) for i in range(poly_deg+1) ])  # vector of terms

    P = sympy.summation(c[k, leg]*sympy.factorial(k)/sympy.factorial(k-m)*beta**(k-m)/T[leg]**m, (k, m, n))  # polynomial derivative
    P = P.subs({m: deriv, n: poly_deg}).doit()

    A_row = sympy.Matrix([P]).jacobian(p)
    b_row = sympy.Matrix([value])
    return A_row, b_row

def find_A_cont(deriv, poly_deg, n_legs, leg):
    """
    Finds rows of constraint matrix for continuity
    @param deriv: the derivative to enforce continuity for
    @param poly_deg: degree of polynomial
    @param beta: 0=start of leg, 1=end of leg
    @n_legs: number of legs in trajectory (num. waypoints - 1)
    @leg: current leg, enforce continuity between leg and leg + 1
    @return A_row, b_row
    """    
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

def compute_trajectory(p, T_opt):
    S = np.hstack([0, np.cumsum(T_opt)])
    t = []
    x = []
    for i in range(len(T_opt)):
        beta = np.linspace(0, 1)
        ti = T_opt[i]*beta + S[i]
        xi = np.polyval(np.flip(p[i*6:(i+1)*6]), beta)
        t.append(ti)
        x.append(xi)
    x = np.hstack(x)
    t = np.hstack(t)
    
    return {
        't': t,
        'x': x}

def find_cost_function(poly_deg=5, min_deriv=3, rowsf_x=[0, 1, 2, 3, 6, 7, 8, 9], rowsf_y=[0, 1, 2, 3, 6, 7, 8, 9], n_legs=2):
    """
    Find cost function for time allocation
    @param poly_deg: degree of polynomial
    @param min_deriv: J = integral( min_deriv(t)^2 dt ), 0=pos, 1=vel, etc.
    @param rowsf: fixed boundary conditions
        0 pos leg 0 start
        1 pos leg 0 end
        2 vel leg 0 start
        3 vel leg 0 end
        4 acc leg 0 start
        5 acc leg 0 end
        .. repeats for next leg
    @param n_legs: number of legs
    """
    A_rows_x = []
    b_rows_x = []
    A_rows_y = []
    b_rows_y = []

    Q = find_Q(deriv=min_deriv, poly_deg=poly_deg, n_legs=n_legs)

    # symbolic boundary conditions
    n_l, n_d = sympy. symbols('n_l, n_d', integer=True)  # number of legs and derivatives
    x = sympy.MatrixSymbol('x', n_d, n_l)
    y = sympy.MatrixSymbol('y', n_d, n_l)
    T = sympy.MatrixSymbol('T', n_l, 1)  # time of leg
    
    # continuity
    if False:  # enable to enforce continuity
        for m in range(3):
            for i in range(n_legs-1):
                A_row, b_row = find_A_cont(deriv=m, poly_deg=5, n_legs=2, leg0=i, leg1=i+1)
                A_rows.append(A_row)
                b_rows.append(b_row)

    # position, vel, accel, beginning and end of leg
    if True:
        for i in range(n_legs):
            for m in range(3):
                # start x
                A_row_x, b_row_x = find_A(deriv=m, poly_deg=poly_deg, beta=0, n_legs=n_legs, leg=i, value=x[m, i])
                A_rows_x.append(A_row_x)
                b_rows_x.append(b_row_x)
        
                # stop x
                A_row_x, b_row_x = find_A(deriv=m, poly_deg=poly_deg, beta=1, n_legs=n_legs, leg=i, value=x[m, i+1])
                A_rows_x.append(A_row_x)
                b_rows_x.append(b_row_x)

                # start y
                A_row_y, b_row_y = find_A(deriv=m, poly_deg=poly_deg, beta=0, n_legs=n_legs, leg=i, value=y[m, i])
                A_rows_y.append(A_row_y)
                b_rows_y.append(b_row_y)
        
                # stop y
                A_row_y, b_row_y = find_A(deriv=m, poly_deg=poly_deg, beta=1, n_legs=n_legs, leg=i, value=y[m, i+1])
                A_rows_y.append(A_row_y)
                b_rows_y.append(b_row_y)

    A_x = sympy.Matrix.vstack(*A_rows_x)
    A_y = sympy.Matrix.vstack(*A_rows_y)

    # Check square matrix for x
    if not A_x.shape[0] == A_x.shape[1]:
        raise ValueError('A_x must be square', A_x.shape)

    # Check square matrix for x
    if not A_y.shape[0] == A_y.shape[1]:
        raise ValueError('A_y must be square', A_y.shape)
    
    b_x = sympy.Matrix.vstack(*b_rows_x)
    b_y = sympy.Matrix.vstack(*b_rows_y)

    I_x = sympy.Matrix.eye(A_x.shape[0])
    I_y = sympy.Matrix.eye(A_y.shape[0])
    
    # free x constraints
    rowsp_x = list(range(A_x.shape[0]))
    for row in rowsf_x:
        rowsp_x.remove(row)

    # free y constraints
    rowsp_y = list(range(A_y.shape[0]))
    for row in rowsf_y:
        rowsp_y.remove(row)

    # compute permutation matrix for x
    rows_x = rowsf_x + rowsp_x
    C_x = sympy.Matrix.vstack(*[I_x[i, :] for i in rows_x])

    # compute permutation matrix for y
    rows_y = rowsf_y + rowsp_y
    C_y = sympy.Matrix.vstack(*[I_y[i, :] for i in rows_y])

    # find R_x for x
    A_I_x = A_x.inv()
    R_x = (C_x@A_I_x.T@Q@A_I_x@C_x.T)
    R_x.simplify()

    # find R_y for y
    A_I_y = A_y.inv()
    R_y = (C_y@A_I_y.T@Q@A_I_y@C_y.T)
    R_y.simplify()

    # split R_x
    n_f_x = len(rowsf_x) # number fixed
    n_p_x = len(rowsp_x)  # number free
    Rpp_x = R_x[n_f_x:, n_f_x:]
    Rfp_x = R_x[:n_f_x, n_f_x:]

    # split R_y
    n_f_y = len(rowsf_y) # number fixed
    n_p_y = len(rowsp_y)  # number free
    Rpp_y = R_y[n_f_y:, n_f_y:]
    Rfp_y = R_y[:n_f_y, n_f_y:]
    
    # find fixed parameters for x
    df_x = (C_x@b_x)[:n_f_x, 0]

    # find fixed parameters for y
    df_y = (C_y@b_y)[:n_f_y, 0]

    # find free parameters for x
    dp_x = -Rpp_x.inv()@Rfp_x.T@df_x
    
    # find free parameters for y
    dp_y = -Rpp_y.inv()@Rfp_y.T@df_y

    # complete parameters vector for x 
    d_x = sympy.Matrix.vstack(df_x, dp_x)

    # complete parameters vector for y 
    d_y = sympy.Matrix.vstack(df_y, dp_y)
    
    # find polynomial coefficients for x
    p_x = A_I_x@d_x

    # find polynomial coefficients for y
    p_y = A_I_y@d_y
    
    Ti = sympy.symbols('T_0:{:d}'.format(n_legs))

    # find optimized cost
    k = sympy.symbols('k')  # time weight
    J = ((p_x.T@Q@p_x)[0, 0]).simplify() + ((p_y.T@Q@p_y)[0, 0]).simplify() + k*sum(Ti)

    J = J.subs(T, sympy.Matrix(Ti))
    p_x = p_x.subs(T, sympy.Matrix(Ti))
    p_y = p_y.subs(T, sympy.Matrix(Ti))
    
    return {
        'f_J': sympy.lambdify([Ti, x, y, k], J),
        'f_p_x': sympy.lambdify([Ti, x, k], list(p_x)),
        'f_p_y': sympy.lambdify([Ti, y, k], list(p_y))
    }


def run_traj(x1,v_x,y1,v_y,ax,ay,k, plot = False): #input 4x1 matrix
    # calls other functions, optimize and minimize. 
    n_legs = len(x1)-1
    cost = find_cost_function(poly_deg=5, min_deriv=3,
    rowsf_x= list(range(6*n_legs)), 
    rowsf_y = list(range(6*n_legs)), 
    n_legs=n_legs)

    x = sympy.Matrix([  # boundary conditions for x
        x1,v_x,ax
        ])

    y = sympy.Matrix([  # boundary conditions for y
        y1,v_y,ay
        ])

    k_time = 10^20 #weight on time
    sol = sol = scipy.optimize.minimize(lambda T: cost['f_J'](T, x, y, k_time), [1]*n_legs, bounds=[[0.1, 100]]* n_legs)

    T_opt = sol['x']

    p_opt_x = cost['f_p_x'](T_opt, x, k_time)
    p_opt_y = cost['f_p_y'](T_opt, y, k_time)

    traj_x = compute_trajectory(p_opt_x, T_opt)
    traj_y = compute_trajectory(p_opt_y, T_opt)

    print("Leg Times: ", T_opt)
    print("Total Time: ", sum(T_opt))

    if plot:
        plt.figure()
        plt.plot(traj_x['t'], traj_x['x'])
        plt.xlabel('t')
        plt.ylabel('x')
        plt.grid(True)
        plt.title('x vs t')

        plt.figure()
        plt.plot(traj_y['t'], traj_y['x'])
        plt.xlabel('t')
        plt.ylabel('y')
        plt.grid(True)
        plt.title('y vs t')

        plt.figure()
        normalize = colors.Normalize(vmin=min(traj_x['t']), vmax=max(traj_x['t']))
        plt.scatter(traj_x['x'], traj_y['x'], c=traj_x['t'], s=8, cmap='viridis', norm=normalize, marker=(5,2))
        plt.xlabel('x')
        plt.ylabel('y')
        plt.grid(True) 
        plt.title('y vs x path')

    return {'x':traj_x['x'], 'y':traj_y['x'], 't_x':traj_x['t'],'t_y':traj_y['t']}

