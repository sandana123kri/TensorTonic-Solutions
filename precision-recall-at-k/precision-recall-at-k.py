def precision_recall_at_k(recommended, relevant, k):
    """
    Compute precision@k and recall@k for a recommendation list.
    """
    # Write code here
    top_k=recommended[:k]
    intersection = list(set(top_k) & set(relevant))
    n=len(intersection)
    precision=n/k
    recall=n/len(relevant)
    return [precision,recall]