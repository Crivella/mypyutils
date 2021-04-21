from aiida import orm
from aiida.orm.utils import load_node

def report_exception(node, tab=''):
    print(tab + '{}'.format(('\n' + tab + '- ').join(str(node.exception).split('\n'))))

def report_failed(node, tab='', actions={}):
    ptr = failed_node = node
    while True:
        print(tab + '- {}<{}>: [{}] {}'.format(ptr.process_class.__name__, ptr.pk, ptr.exit_status, ptr.exit_message))
        if ptr.is_excepted:
            report_exception(ptr, tab=tab + '  ')
        if ptr.exit_status:
            failed_node = ptr
        try:
            ptr = ptr.called[0]
        except:
            break

    print(tab + 'Failed on {}'.format(failed_node.process_class.__name__))

    for typ, act in actions.items():
        if failed_node.process_class == typ:
            return act(failed_node)

def validate_node(node):
    if isinstance(node, int):
        res = load_node(node)
    elif isinstance(node, orm.Node):
        res = node
    else:
        raise ValueError('`pk` must be either a aiida.orm.Node or a pk to a node')

    return res