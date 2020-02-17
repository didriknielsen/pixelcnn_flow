import torch

def prep_int(arg, num, allow_none=False):
    if allow_none:
        if arg is None: arg = [arg] * num
    if isinstance(arg, int): arg = [arg] * num
    assert len(arg) == num
    return arg

def prep_float(arg, num, allow_none=False):
    if allow_none:
        if arg is None: arg = [arg] * num
    if isinstance(arg, float): arg = [arg] * num
    assert len(arg) == num
    return arg

def prep_str(arg, num):
    try:
        arg = eval(arg)
        assert isinstance(arg, list)
    except:
        arg = [arg] * num
    assert len(arg) == num
    return arg

def prep_bool(arg, num):
    try:
        arg = eval(arg)
        assert isinstance(arg, list)
    except:
        arg = [arg] * num
    assert len(arg) == num
    return arg
