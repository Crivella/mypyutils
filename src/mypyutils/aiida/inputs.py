from ..dict import deep_update

def ListInputs_to_dict(inputs):
    app = {}
    for name in list(inputs):
        value = getattr(inputs, name)
        app[name] = value

    new = {}
    for k, v in app.items():
        if not '__' in k:
            new[k] = v
            continue
        nested = k.split('__')
        res = {}
        ptr = res
        num = len(nested)
        for i, name in enumerate(nested):
            if i < num -1:
                ptr[name] = {}
                ptr = ptr[name]
            else:
                ptr[name] = v

        deep_update(new, res)

    return new