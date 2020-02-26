from .unsorted import tail
from .dict import deep_update

from aiida import load_profile, orm
from aiida.orm.utils import load_node
from aiida.plugins import WorkflowFactory, CalculationFactory
# from aiida.engine import submit

from aiida_z2pack.workchains.chern import Z2pack3DChernWorkChain

PwBandsWorkChain = WorkflowFactory('quantumespresso.pw.bands')
PwCalc = CalculationFactory('quantumespresso.pw')

load_profile()

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

def report_exception(node, tab=''):
    print(tab + '{}'.format(('\n' + tab + '- ').join(str(node.exception).split('\n'))))

def report_failed(node, tab='', actions={}):
    ptr = failed_node = node
    while True:
        print(tab + '- {}<{}>: [{}] {}'.format(ptr.process_class.__name__, ptr.pk, ptr.exit_status, ptr.exit_message))
        if ptr.is_excepted:
            report_exception(ptr, tab=tab + '  ')
        try:
            ptr = ptr.called[0]
        except:
            break

    for typ, act in actions.items():
        if failed_node.process_class == typ:
            return act(failed_node)

def analyze_workchain(
    node,
    report_actions={},
    # **kwargs
    on_failed=None,
    on_excepted=None
    ):
    if isinstance(node, int):
        wc = load_node(node)
    elif isinstance(node, orm.Node):
        wc = node
    else:
        raise ValueError('`pk` must be either a aiida.orm.Node or a pk to a node')

    print(wc.pk)
    # relaunch = True
    if not wc.is_finished and not wc.is_excepted:
        print('  still running')
    if wc.is_finished_ok:
        print('  finished_ok')
    if wc.is_failed:
        print('  Failed:')
        report_failed(wc, tab='    ', actions=report_actions)

        if not on_failed is None:
            on_failed(wc)

    if wc.is_excepted:
        print('  Excepted!!')
        report_exception(wc, tab='    ')

        if not on_excepted is None:
            on_excepted(wc)

    print('-'*100)