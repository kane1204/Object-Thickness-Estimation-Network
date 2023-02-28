import torch
# Adapted from https://visualstudiomagazine.com/Articles/2022/11/14/pytorch-regression-2.aspx?Page=2


def accuracy_fast(pred, y, pct_close=0.10):
    """An accuracy function that works with all inputs and outputs at once, 
       so is faster than the item-by-item approach. This is useful when you just want an accuracy result quickly."""
    Y = y  # all targets
    n_items = Y.shape[0]*Y.shape[2]*Y.shape[3]  # number of items in the batch
    
    n_correct = torch.sum((torch.abs(pred - Y) < torch.abs(pct_close * Y)))
    result = (n_correct.item() / n_items)  # scalar
    return result

def accuracy_debug(model, ds, pct_close):
    """Simple accuracy function that works item-by-item. This approach is slow but 
       you to insert print statements to diagnose incorrect predictions."""
    n_correct = 0; n_wrong = 0
    for i in range(len(ds)):
        X = ds[i][0]   # 2-d inputs
        Y = ds[i][1]   # 2-d target
        with torch.no_grad():
            oupt = model(X)  # computed income

        if torch.abs(oupt - Y) < torch.abs(pct_close * Y):
            n_correct += 1
        else:
            n_wrong += 1
    acc = (n_correct * 1.0) / (n_correct + n_wrong)
    return acc

def mse_loss_with_nans(input, target, mask):

    # Missing data are nan's
    # mask = torch.isnan(target)

    # Missing data are 0's
    mask = target == 0
    print (mask)

    out = (input[~mask]-target[~mask])**2
    loss = out.mean()

    return loss