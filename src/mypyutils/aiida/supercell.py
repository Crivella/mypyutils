import numpy as np

from aiida import orm
from aiida.engine import calcfunction

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