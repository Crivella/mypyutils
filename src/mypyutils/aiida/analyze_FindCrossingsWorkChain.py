import numpy as np
from scipy.spatial import KDTree

from .utils import validate_node
from qepppy.utils import recipr_base

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