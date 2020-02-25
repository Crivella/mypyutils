def dict_str(dct, tab=''):
    res = [tab + '{']
    tab += '  '
    for k,v in dct.items():
        if isinstance(v, dict):
            new = dict_str(v, tab).strip()
        else:
            new = v
        res.append(tab + '\'{}\': {},'.format(k, new))

    res.append(tab + '}')
    return '\n'.join(res)

def deep_copy(old):
    res = {}
    for k,v in old.items():
        if not isinstance(v, dict):
            res[k] = v
        else:
            res[k] = deep_copy(v)

    return res

def deep_update(old, new, overwrite=True):
    """Update nested dictionaries"""
    if not overwrite:
        res = deep_copy(old)
    else:
        res = old
    for k,v in new.items():
        if isinstance(v, dict) and k in res and isinstance(res[k], dict):
            deep_update(res[k], v)
        else:
            res[k] = v  
    return res