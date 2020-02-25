def tail(content, n=20, tab=''):
    s = content.split('\n')

    filtered = s[-n:]

    return ('\n'+tab).join(filtered)