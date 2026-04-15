import numpy as np

def expected_value_discrete(x, p):
    """
    Returns: float expected value
    """
    # Write code here
    x=np.asarray(x,dtype=float)
    p=np.asarray(p,dtype=float)

    # Check shapes
    if x.shape != p.shape:
        raise ValueError("x and p must have the same shape")
    
    # Check probability sum
    if not np.isclose(np.sum(p), 1.0, atol=1e-6):
        raise ValueError("Probabilities must sum to 1")
        
    ans=0.0
    for i in range(len(x)):
        ans+=x[i]*p[i]
    return ans
        
        
