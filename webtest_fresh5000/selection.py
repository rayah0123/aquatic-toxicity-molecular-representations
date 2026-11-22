"""Early-stopping improvement threshold: 1e-12; no TEST data are used."""
def update(best,wait,value):
    improved=value<best-1e-12
    return (value,0,True) if improved else (best,wait+1,False)

def reason(epoch,maximum,wait,patience):
    if wait>=patience:return 'early_stopping'
    if epoch>=maximum:return 'max_epochs'
    return 'running'
