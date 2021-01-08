import numpy as np
import scipy
# from scipy.spatial import KDTree
from aiida import orm
# from aiida.common import AttributeDict
# from aiida.plugins import WorkflowFactory, CalculationFactory
from aiida.engine import calcfunction

# from aiida_quantumespresso.utils.mapping import prepare_process_inputs

def recipr_base(base):
	return np.linalg.inv(base).T * 2 * np.pi

def _kpt_crop(
	kpt_coord, recipr=np.diag([1,1,1]), kpt_weight=None,
	centers=[], radii=np.inf,
	anticrop=False,
	verbose=False
	):
	"""
	Crop the k-points in a sphere of radius 'radius' with center 'center':
	Params:
	 - kpt_coord: (#nkpt,3) shaped array of kpoints in cart/cryst coordinates.
	 - recipr: transformation matrix for the kpt coordinates to cartesian.
	 		   (kpt_cart = recipr.dot(kpt_cord))
	           (default = np.diag([1,1,1]) for kpt_coord = kpt_cart)
	           if kpt_coord is in crystal, recipr should be a matrix of the 3 reciprocal
	           space basis vectors as rows. 
	 - kpt_weight: (#nkpt) shaped array of kpoints weights default to list of ones
	 - centers: list of tuples of 3 floats containing the coordinate of the center
	           of the crop sphere. default = []
	 - radius: Radius of the crop sphere. Defaulr = np.inf
	 - verbose: Print information about the cropping. Default = True
	"""
	nc = len(centers)
	if isinstance(radii, float):
		radii = [radii] * nc
	else:
		if len(radii) != nc:
			raise ValueError("Must pass same number of centers and radii or only one radius.")
	for radius in radii:
		if radius < 0:
			raise ValueError("Radius must be greather than 0.")

	kpt_coord  = np.array(kpt_coord)
	kpt_cart   = kpt_coord.dot(recipr)

	n_kpt = kpt_cart.shape[0]
	if kpt_weight is None:
		kpt_weight = np.ones((n_kpt,))
	else:
		kpt_weight = np.array(kpt_weight)
	kpt_weight /= kpt_weight.sum()

	list_center = np.array(centers).reshape(-1,3)
	index = set()
	for center,radius in zip(list_center, radii):
		norms  = np.linalg.norm(kpt_cart - center, axis=1)

		if not anticrop:
			w = np.where(norms <= radius)[0]
			index = index.union(set(w))
		else:
			w = np.where(norms > radius)[0]
			if not index:
				index = index.union(set(w))
			else:
				index = index.intersection(set(w))

	index = list(index)

	res_kpt    = kpt_cart[index]
	res_weight = kpt_weight[index]

	crop_weight = res_weight.sum()
	tot_weight  = kpt_weight.sum()
	# norm = crop_weight / tot_weight

	if verbose:
		print(f"# Cropping k-points around {centers} with radius {radius}")
		print(f"# Cropped {len(index)} k-points out of {n_kpt}")
		print(f"# The weight of the selected points is {crop_weight} vs the total weight {tot_weight}")
		print(f"# Re-normaliing by a factor {tot_weight/crop_weight}")

	# res_weight /= res_weight.sum()
	# res_weight = self._normalize_weight(res_weight)

	return res_kpt, res_weight
	
@calcfunction
def kpt_crop(kpoints: orm.KpointsData, centers: orm.ArrayData, radii: orm.ArrayData, anticrop=orm.Bool(False)) -> orm.KpointsData:
	kpt_cryst = kpoints.get_kpoints_mesh(print_list=True)
	cell = kpoints.cell
	recipr = recipr_base(cell)

	centers = centers.get_array('centers')
	radii   = radii.get_array('radii')

	kpt, wgt = _kpt_crop(kpt_cryst, recipr, centers=centers, radii=radii, anticrop=anticrop.value)

	res = orm.KpointsData()
	res.set_cell(cell)
	res.set_kpoints(kpt, cartesian=True, weights=wgt)

	return res

@calcfunction
def pw2gwArray_to_XyData(array: orm.ArrayData) -> orm.XyData:
	res = orm.XyData()
	return res

@calcfunction
def mergeXyData(data1, data2, weight1, weight2) -> orm.XyData:
	if isinstance(weight1, orm.Float):
		W1 = weight1.value
	elif isinstance(weight1, orm.KpointsData):
		W1 = weight1.get_array('weights')
	else:
		raise TypeError('`weight1` is of unsupported type {}'.format(type(weight1)))
	if isinstance(weight2, orm.Float):
		W2 = weight2.value
	elif isinstance(weight2, orm.KpointsData):
		W2 = weight2.get_array('weights')
	else:
		raise TypeError('`weight2` is of unsupported type {}'.format(type(weight1)))

	x1 = data1.get_x()
	x2 = data2.get_x()

	if x1[0] != x2[0] or x1[2] != x2[2]:
		raise ValueError('Mismatch in X axis')
	X1 = x1[1]
	X2 = x2[1]

	y1 = data1.get_y()
	y2 = data2.get_y()

	x_min = min(X1.min(), X2.min())
	x_max = max(X1.max(), X2.max())
	dx = min(x1[1] - x1[0], x2[1] - x2[0])

	names = []
	units = []
	ly1 = []
	ly2 = []
	for name, data, units in y1:
		for n2, d2, u2 in y2:
			if n2 == name:
				if units != u2:
					raise ValueError('Mismatch in Y Axis. Units `{}` != `{}` for array `{}`'.format(
						units, u2, name))
				names.append(name)
				units.append(units)
				ly1.append(data)
				ly2.append(d2)
				break
		else:
			raise ValueError('Mismatch in Y Axis. `{}` array missing from second set'.format(name))

	resX = np.arange(x_min, x_max, dx)
	resYdata = []
	resYnames = []
	resYunits = []
	for Y1, Y2, name, unit in zip(ly1, ly2, names, units):
		newY1 = scipy.interpolate.interp1d(X1, Y1, fill_value=0)(resX)
		newY2 = scipy.interpolate.interp1d(X2, Y2, fill_value=0)(resX)

		newY = W1 * newY1 + W2 * newY2

		resYdata.append(newY)
		resYnames.append(name)
		resYunits.append(units)

	res = orm.XyData()

	res.set_x(resX, x1[0], x1[2])
	res.set_y(resYdata, resYnames, resYunits)

	return res

# multipliers = [1,3,5,7,9,15,21,27,35,45,63,75,81]

# def generate_congruent_grids(mesh, max_i):
# 	from scipy.spatial import KDTree
# 	mesh = np.array(mesh)
# 	done = None
# 	for m in multipliers:
# 		print()
# 		print(m)
# 		new = m*mesh
# 		if any(i>max_i for i in new):
# 			break

# 		grid = generate_monkhorst_pack_grid(new)
# 		if done is None:
# 			done = grid
# 			res = grid
# 		else:
# 			t = KDTree(done)
# 			q = t.query_ball_point(grid, r=1E-5)
# 			w = [i for i,l in enumerate(q) if len(l) == 0]
# 			res = grid[w]
# 			done = np.vstack((done, res))
# 			# print('-----')
# 			# print(q)
# 			# print(w)
# 			# print('-----')
# 			# break

# 		print(f'{len(grid)} -> {len(res)}')
# 		yield res
# 		# break

