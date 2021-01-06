import numpy as np
from scipy.spatial import KDTree
# from aiida import orm
# from aiida.common import AttributeDict
# from aiida.plugins import WorkflowFactory, CalculationFactory
# from aiida.engine import WorkChain, ToContext, if_, calcfunction

# from aiida_quantumespresso.utils.mapping import prepare_process_inputs

def recipr_base(base):
	return np.linalg.inv(base).T * 2 * np.pi

def cart_to_cryst(recipr, coord):
	return coord.dot(np.linalg.inv(recipr))

def cryst_to_cart(recipr, coord):
	return coord.dot(recipr)

def u(r, q):
	return (2.*r - q - 1.), (2. * q)

def factorize(x):
	res = [1]
	i = 2
	while i <= x:
		if x%i == 0:
			res.append(i)
			x /= i
		else:
			i += 1

	return res

def generate_monkhorst_pack_grid(mesh, shift=(0,0,0)):
	"""
	Generate a Monkhorst-Pack grid of k-point.
	Params:
	 -mesh:  tuple of 3 ints > 0
	 -shift: tuple of 3 ints that can be either 0 or 1
	"""
	from itertools import product

	s1,s2,s3   = shift

	l1,l2,l3 = [
		list(
			(n+shift[i])/d for n,d in 
				(
					u(r+1,q) for r in range(q)
				)
		) for i,q in enumerate(mesh)
		]

	kpts = np.array(list(product(l1,l2,l3)))

	return kpts

def kpt_crop(
	kpt_cart, kpt_weight=None,
	centers=[], radius=np.inf, 
	verbose=True
	):
	"""
	Crop the k-points in a sphere of radius 'radius' with center 'center':
	Params:
	 - kpt_cart: (#nkpt,3) shaped array of kpoint in cartesian coordinates
	 - kpt_weight: (#nkpt) shaped array of kpoints weights default to list of ones
	 - centers: list of tuples of 3 floats containing the coordinate of the center
	           of the crop sphere. default = []
	 - radius: Radius of the crop sphere. Defaulr = np.inf
	 - verbose: Print information about the cropping. Default = True
	"""
	if radius < 0:
		raise ValueError("Radius must be greather than 0.")

	n_kpt = len(kpt_cart)
	if kpt_weight is None:
		kpt_weight = np.ones((n_kpt))

	list_center = np.array(centers).reshape(-1,3)
	index = set()
	for center in list_center:
		norms  = np.linalg.norm(kpt_cart - center, axis=1)
		w      = np.where(norms <= radius)[0]

		index = index.union(set(w))

	index = list(index)

	crop_weight = kpt_weight[index].sum()
	tot_weight  = kpt_weight.sum()
	if verbose:
		print(f"# Cropping k-points around {centers} with radius {radius}")
		print(f"# Cropped {len(index)} k-points out of {n_kpt}")
		print(f"# The weight of the selected points is {crop_weight} vs the total weight {tot_weight}")
		print(f"# Re-normaliing by a factor {tot_weight/crop_weight}")

	res_kpt    = kpt_cart[index]
	res_weight = kpt_weight[index]

	res_weight /= res_weight.sum()
	# res_weight = self._normalize_weight(res_weight)

	return res_kpt, res_weight

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


# cell = [
# 	[0.0, 3.133592071, 3.133592071],
# 	[3.133592071, 0.0, 3.133592071],
# 	[3.133592071, 3.133592071, 0.0]]

# recipr = recipr_base(cell)

# v1,v2,v3 = recipr
# vol = np.dot(np.cross(v1,v2), v3)

# print(vol)
# d = 1E-2
# print('q =', vol**(1/3) /d)

# print(np.linalg.norm(cell, axis=1))

# exit()

# recipr = recipr_base(cell)

# for g in generate_congruent_grids([2,2,2], 30):
# 	pass
# 	# print(g)
# exit()

# # print(recipr)
# # kpt_cryst = generate_monkhorst_pack_grid([1,1,1])
# # print(kpt_cryst)
# # print()
# kpt_cryst0 = generate_monkhorst_pack_grid([2,2,1])
# print(kpt_cryst0)
# print()
# t0 = KDTree(kpt_cryst0)
# # kpt_cryst = generate_monkhorst_pack_grid([3,3,1])
# # print(kpt_cryst)
# # print()
# # kpt_cryst = generate_monkhorst_pack_grid([4,4,1])
# # print(kpt_cryst)
# # print()
# # kpt_cryst = generate_monkhorst_pack_grid([5,5,1])
# # print(kpt_cryst)
# # print()
# kpt_cryst1 = generate_monkhorst_pack_grid([6,6,1])
# t1 = KDTree(kpt_cryst1)
# c = t0.query_ball_tree(t1, r = 1E-5)
# print(all(len(_) for _ in c))
# print()

# kpt_cryst2 = generate_monkhorst_pack_grid([18,18,1])
# t2 = KDTree(kpt_cryst2)
# c = t1.query_ball_tree(t2, r = 1E-5)
# print(all(len(_) for _ in c))
# print()

# kpt_cryst5 = generate_monkhorst_pack_grid([10,10,1])
# t5 = KDTree(kpt_cryst5)
# c = t0.query_ball_tree(t5, r = 1E-5)
# print(all(len(_) for _ in c))
# c = t1.query_ball_tree(t5, r = 1E-5)
# print(all(len(_) for _ in c))
# print()

# print(factorize(1*2*2*2*3*4*5*7*11))

# # kpt_cryst = generate_monkhorst_pack_grid([7,7,1])
# # print(kpt_cryst)
# # print()
# # kpt_cart = cryst_to_cart(recipr, kpt_cryst)
# # print(kpt_cart)

# # print(kpt_crop(kpt_cart, centers=[(0,0,0), (0.1,0,0)], radius=0.4))