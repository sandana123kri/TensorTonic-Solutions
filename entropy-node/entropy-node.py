import numpy as np

def entropy_node(y):
    """
    Compute entropy for a single node using stable logarithms.
    """
    # Write code here
    y=np.asarray(y)
    if(y.size==0):
        return 0.0
    val, count=np.unique(y,return_counts=True)
    p=count/count.sum()
    p=p[p>0]
    return float(-(p*np.log2(p)).sum()) if p.size else 0.0
    
    