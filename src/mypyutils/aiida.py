from .dict import deep_update

import numpy as np
from scipy.spatial import KDTree

from aiida import orm
from aiida.orm.utils import load_node
from aiida.engine import calcfunction
from qepppy.utils import recipr_base

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

def analyze_workchain(
    node,
    report_actions={},
    # **kwargs
    on_killed=None,
    on_running=None,
    on_finished_ok=None,
    on_failed=None,
    on_excepted=None,
    tab='',
    ):
    wc = validate_node(node)

    if wc.is_killed:
        print(tab + 'Killed!!!')
        if not on_killed is None:
            return on_killed(wc)
        return

    if wc.is_finished_ok:
        print(tab + 'finished_ok')
        if not on_finished_ok is None:
            return on_finished_ok(wc)
        return

    if wc.is_failed:
        print(tab + 'Failed:')
        report_failed(wc, tab=tab+'  ', actions=report_actions)

        if not on_failed is None:
            return on_failed(wc)
        return

    if wc.is_excepted:
        print(tab + 'Excepted!!')
        report_exception(wc, tab=tab+'  ')

        if not on_excepted is None:
            return on_excepted(wc)
        return

    print(tab + 'still running')
    if not on_running is None:
        on_running(wc)

def void(*args, **kwargs):
    return

def analyze_FindCrossingsWorkChain(node, gap_thr=0.0025, noprint=False):
    wc = validate_node(node)
    if noprint:
        log = void
    else:
        log = print
    log('Analyzing {}<{}>'.format(wc.process_class.__name__, wc.pk))

    l_find = [called for called in wc.called if 'bands_data' in list(called.inputs)]

    calc   = l_find[0].inputs.bands_data.creator
    param  = calc.outputs.output_parameters.get_dict()
    n_el   = param['number_of_electrons']
    spin   = param['spin_orbit_calculation']
    cb     = round(n_el) // (int(not spin) + 1)
    vb     = cb - 1

    lbands  = [c.inputs.bands_data for c in l_find]
    lbands.sort(key=lambda x: x.pk, reverse=True)    

    res = {
        'min_gap':[],
        'pinned':[],
        'pgaps':[],
        'found':[],
        'fgaps':[],
        'distance':[]
        }
    for nn, bands in enumerate(lbands):
        b = bands.get_bands()
        kpt_c = bands.get_kpoints(cartesian=True)
        g = b[:,cb] - b[:,vb]

        cell   = bands.cell
        recipr = recipr_base(cell)
        irecipr = np.linalg.inv(recipr)


        log('bands<{}>'.format(bands.pk))
        try:
            kki      = bands.creator.inputs.kpoints.creator.inputs
            pinned   = kki.centers.get_array('pinned')
            distance = kki.distance.value
        except:
            distance = 200
            pinned = np.array([[0.,0.,0.]])
        log('  prev distance: {}, gap_thr: {}'.format(distance, gap_thr))
        
        centers  = KDTree(pinned)
        kpt_tree = KDTree(kpt_c)
        query    = centers.query_ball_tree(kpt_tree, r=distance*1.74/2)

        pinned_thr = distance * 4.00

        lim = max(-5 // np.log10(distance), 1) if distance < 1 else 200
        if distance < 0.01:
            lim = 1
        log('    LIM: {}'.format(lim))
        where_found  = []
        where_pinned = []
        for n,q in enumerate(query):
            q = np.array(q, dtype=np.int)
            if len(q) == 0:
                log('    skipping {}, no neighbours'.format(pinned[n]))
                continue

            mi =  g[q].argmin()
            min_gap = g[q[mi]]

            # _, i = kpt_tree.query(pinned[n])
            # prev_min_gap = g[i]

            # log('    {:2d}.  min_gap: {:.6f}  kpt: {}  pmg: {:.6f}  kpt: {} {}'.format(n, min_gap, kpt_c[q][mi], prev_min_gap, kpt_c[i], pinned[n]))
            # if min_gap / prev_min_gap > 0.95 and distance < 0.005:
            #     log('         skipping mg/pmg: {}'.format(min_gap / prev_min_gap))
            #     continue
            log('    {:2d}.  min_gap: {:.6f}  kpt: {}  pinned: {}'.format(n, min_gap, kpt_c[q][mi], pinned[n]))
            scale = 2.5 if lim > 1 else 1.0001
            if distance == 200:
                scale = 0.25 / min_gap
            app = None
            while app is None or len(app) > lim:
                app = np.where(g[q] < min_gap * scale)[0]
                scale *= 0.98
                if scale < 1.0001:
                    app = np.where(g[q] < min_gap * 1.0001)[0]
                    break
            where_found.extend([q[i] for i in app if g[q[i]] <= gap_thr])
            aaa = [q[i] for i in app if gap_thr < g[q[i]]]
            where_pinned.extend([q[i] for i in app if gap_thr < g[q[i]] < pinned_thr])

            where_app = [q[i] for i in app if g[q[i]] >= pinned_thr]
            if where_app:
                log('         Skipping {} out of {} for fermi velocity'.format(where_app, aaa))

        wp = np.unique(np.array(where_pinned, dtype=np.int))
        wf = np.unique(np.array(where_found, dtype=np.int))

        log('    --------------')

        for n, w in enumerate(wp):
            min_gap = g[w]
            min_kpt = kpt_c[w]
            log('   {:4d}.   GAP: {:.6f}  kpt_cart: {},  kpt_cryst: {}'.format(n, min_gap, min_kpt, np.dot(min_kpt, irecipr)))

        log("    Min gap: {},  kpt: {}".format(g.min(), kpt_c[g.argmin()]))

        log('NEW pinned count: {}'.format(len(wp)))
        log('FOUND:', len(wf))
        for f in wf:
            log('  kpt: {}   gap: {:.6f}'.format(kpt_c[f], g[f]))
        log()

        res['min_gap'].append(g.min())
        res['pinned'].append(kpt_c[wp])
        res['pgaps'].append(g[wp])
        res['found'].append(kpt_c[wf])
        res['fgaps'].append(g[wf])
        res['distance'].append(distance)

    return res

save = {
    # 12523: {(2,1,1): load_node(36758)},
    # 12155: {(2,1,1): load_node(36764)}
    }
def make_supercell(structure, supercell):
    if not isinstance(supercell, orm.ArrayData):
        app = orm.ArrayData()
        app.set_array('data', np.array(supercell))
        supercell = app
    pk = structure.pk
    sc = tuple(supercell.get_array('data'))
    if pk in save:
        if sc in save[pk]:
            return save[pk][sc]

    res = _make_supercell(structure, supercell)

    if not pk in save:
        save[pk] = {}
    save[pk][sc] = res

    return res

@calcfunction
def _make_supercell(structure, supercell):
    from itertools import product
    mag_atoms = supercell.get_array('mag_atoms')
    sc = tuple(supercell.get_array('data'))
    x,y,z = sc
    grid = np.array(list(product(range(x), range(y), range(z))))

    cell = structure.cell

    new = orm.StructureData()
    new.set_cell((np.array(cell).T * sc).T)

    kinds = structure.kinds
    sites = structure.sites

    counter = {k.name:0 for k in kinds}

    for R_cryst in grid:
        for n,site in enumerate(sites):
            pos = np.array(site.position) + np.dot(R_cryst, cell)
            name = site.kind_name

            counter[name] += 1

            new.append_atom(
                position=pos,
                symbols=name,
                name=f'{name}{counter[name] if name in mag_atoms else ""}',
                )

    return new

def plot_bandstructure(bs_node, dy=None, savedir='.'):
    import os
    import matplotlib.pyplot as plt

    try:
        data   = bs_node.outputs.band_structure
    except:
        print('WARNING! No output from', bs_node)
        return
    else:
        print('Doing bs_node: {}, bands: {}'.format(bs_node.pk, data.pk))
    # continue
    struct = bs_node.inputs.structure
    param  = bs_node.outputs.scf_parameters.get_dict()
    ef     = param['fermi_energy']

    plot_info = data._get_bandplot_data(cartesian=True, prettify_format='gnuplot_seekpath', join_symbol='|', y_origin=ef)

    x = np.array(plot_info['x'])
    y = np.array(plot_info['y'])

    if dy:
        ymin = -dy
        ymax = dy
    else:
        ymin = y.min()
        ymax = y.max()

    labels = plot_info['labels']

    tpos = [_[0] for _ in labels]
    tlab = [_[1] for _ in labels]
    tlab = [fr'${_}$' if '$' not in _ else _ for _ in tlab if _]

    formula = struct.get_formula()
    fname = os.path.join(savedir, '{}-{}.pdf'.format(bs_node.pk, formula))

    for i in '0123456789':
        formula = formula.replace(i, fr'$_{i}$')

    fig, ax = plt.subplots()

    xmin, xmax = x.min(), x.max()
    ax.plot(x, y, color='k', linewidth=.5)
    ax.hlines([0],xmin, xmax, linestyles='dashed', color='cyan')
    ax.vlines(tpos, ymin, ymax, linestyles='dashed', color='gray')

    ax.set_title('{}  -  {}'.format(formula, data.pk))
    # ax.set_title('{}'.format(formula))
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    ax.set_xticks(tpos)
    if tlab:
        ax.set_xticklabels(tlab)
    ax.set_ylabel('Energy (eV)')
    
    # pdf.savefig(fig)
    plt.savefig(fname)
    plt.close(fig)
    # break
