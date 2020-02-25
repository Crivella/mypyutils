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

def report_failed(node, tab='', actions={}):
    ptr = node
    while True:
        print(tab + '- {}<{}>: [{}] {}'.format(ptr.process_class.__name__, ptr.pk, ptr.exit_status, ptr.exit_message))
        if ptr.is_excepted:
            print(tab + '  ' + '- {}'.format(('\n'+ tab + '  ' + '- ').join(str(ptr.exception).split('\n'))))
        try:
            ptr = ptr.called[0]
        except:
            break

    for typ, act in actions.items():
        if isinstance(ptr, typ):
            act(ptr)
            break

def analyze_workchain(
    node,
    report_actions={},
    on_failed=None
    ):
    if isinstance(node, int):
        wc = load_node(node)
    elif isinstance(node, orm.Node):
        wc = node
    else:
        raise ValueError('`pk` must be either a aiida.orm.Node or a pk to a node')

    print(wc.pk)
    relaunch = True
    if not wc.is_finished and not wc.is_excepted:
        print('  still running')
    if wc.is_finished_ok:
        print('  finished_ok')
    if wc.is_failed:
        print('  Failed:')
        # def action_on_PwCalc_term(ptr):
        #     retrieved = ptr.outputs.retrieved
        #     content = retrieved.get_object_content('aiida.out')
        #     if 'S matrix not positive definite' in content:
        #         # relaunch = False
        #         print('        S matrix not positive definite')
        #     else:
        #         print('    ' + '#'*40)
        #         print(tail(content, tab='    '))
        #         print('    ' + '#'*40)

        report_failed(wc, tab='    ', actions=report_actions)
        # ptr = wc
        # while True:
        #     print('    - {}<{}>: [{}] {}'.format(ptr.process_class.__name__, ptr.pk, ptr.exit_status, ptr.exit_message))
        #     if ptr.is_excepted:
        #         print('      - {}'.format('\n      - '.join(str(ptr.exception).split('\n'))))
        #     try:
        #         ptr = ptr.called[0]
        #     except:
        #         break

        if not on_failed is None:
            on_failed(wc)
        # try:
        #     scf = wc.called[0].outputs.scf_remote_folder
        # except:
        #     scf = None

        # if relaunch:
        #     print('      ---- Relaunching ...')
        #     inputs = ListInputs_to_dict(wc.inputs)
        #     inputs['z2pack_base']['z2pack']['metadata'] = mdata_z2pack
        #     inputs['find']['bands']['pw']['metadata'] = mdata_nscf
        #     inputs['find']['scf']['pw']['metadata'] = mdata_scf
        #     # print(dict_str(inputs))
        #     # exit()
        #     if not scf is None:
        #         print('         with previous scf')
        #         inputs['find'].pop('scf', {})
        #         inputs['find']['parent_folder'] = scf

        #     if not debug:
        #         res = submit(Z2pack3DChernWorkChain, **inputs)
        #         print('        Launched <{}>'.format(res.pk))
        #         g.remove_nodes(wc)
        #         g_fail.add_nodes(wc)
        #         g.add_nodes(res)


    if wc.is_excepted:
        print('  Excepted!!')
        print('    {}'.format('\n    - '.join(str(wc.exception).split('\n'))))

    print('-'*100)