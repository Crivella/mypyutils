import os
import numpy as np
import matplotlib.pyplot as plt

from aiida import orm

def plot_bandstructure(
    node, 
    ax=None,
    dy=None,
    skip_done=False,
    save=True, savedir='.', ext='pdf', formula='', save_dat=False,
    plot_kwargs={},
    # line_kwargs={}
    ):
    struct = None
    param = None
    if isinstance(node, orm.BandsData):
        data = node
    elif isinstance(node, orm.WorkChainNode):
        pname = node.process_class.__name__
        if pname == 'PwBandsWorkChain':
            data = node.outputs.band_structure
            struct = node.inputs.structure
            param  = node.outputs.scf_parameters.get_dict()
        elif pname == 'Wannier90BandsWorkChain':
            data = node.outputs.wannier90_interpolated_bands
            struct = node.inputs.structure
            param  = node.outputs.scf_parameters.get_dict()
        else:
            raise NotImplemented
    else:
        raise NotImplemented

    print('Doing node: {}, bands: {}'.format(node.pk, data.pk))

    ef = 0
    if param:
        ef = param['fermi_energy']

    if not formula and struct:
        formula = struct.get_formula()
    os.makedirs(savedir, exist_ok=True)
    fname = os.path.join(savedir, '{}-{}.{}'.format(node.pk, formula, ext))

    plot_info = data._get_bandplot_data(cartesian=True, prettify_format='gnuplot_seekpath', join_symbol='|', y_origin=ef)

    x = np.array(plot_info['x'])
    y = np.array(plot_info['y'])

    if save_dat:
        res = np.hstack((x.reshape(-1,1), y))
        dat_fname = fname.replace(f'.{ext}', '.dat')
        np.savetxt(dat_fname, res)

    if skip_done and os.path.exists(fname):
        return x,y

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


    for i in '0123456789':
        formula = formula.replace(i, fr'$_{i}$')

    newax = False
    if ax is None:
        fig, ax = plt.subplots()
        newax = True

    plot_kwargs.setdefault('color', 'k')
    plot_kwargs.setdefault('linewidth', .5)

    xmin, xmax = x.min(), x.max()
    line = ax.plot(x, y, **plot_kwargs)
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
    if save:
        plt.savefig(fname)
    if newax:
        plt.close(fig)
    
    return x, y, line
