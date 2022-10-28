

def mult_diag(d, mtx, left=True):
    """
        the code is adapted from https://github.com/HuthLab/speechmodeltutorial
        Multiply a full matrix by a diagonal matrix.
        This function should always be faster than dot.
        Input:
            d -- 1D (N,) array (contains the diagonal elements)
            mtx -- 2D (N,N) array

        Output:
            mult_diag(d, mts, left=True) == dot(diag(d), mtx)
            mult_diag(d, mts, left=False) == dot(mtx, diag(d))
        
        By Pietro Berkes
        From http://mail.scipy.org/pipermail/numpy-discussion/2007-March/026807.html
    """
    if left:
        # return (d*mtx.T).T
        return (d*mtx.transpose()).transpose()
    else:
        return d*mtx