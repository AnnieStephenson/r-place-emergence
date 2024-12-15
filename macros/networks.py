
import numpy as np
import os, sys
import rplacem.canvas_part as cp
import rplacem.utilities as util
import rplacem.plot_utilities as plot
from rplacem import var as var
import pickle
import graph_tool.all as gt
import matplotlib.pyplot as plt

file_path = os.path.join(var.DATA_PATH, 'canvas_compositions_0-100.pickle') #_all
with open(file_path, 'rb') as f:
    canvas_parts = pickle.load(f)
    f.close()

'''
with open(os.path.join(var.DATA_PATH, 'canvas_compositions_0-100.pickle') , 'wb') as handle:
    pickle.dump(canvas_parts,
                handle,
                protocol=pickle.HIGHEST_PROTOCOL)
'''

cpart = canvas_parts[1]

sec = cpart.pixel_changes['seconds']
user = cpart.pixel_changes['user'][sec < var.TIME_GREYOUT]
x, y = cpart.pixchanges_coords()
x = x[sec < var.TIME_GREYOUT]
y = y[sec < var.TIME_GREYOUT]
sec = sec[sec < var.TIME_GREYOUT]

# separate the pixel changes into time bins of width tgrain
tgrain = 300
tsep = np.arange(min(sec), max(sec) +10-1e-4, tgrain)
inds_sep = np.searchsorted(sec, tsep)

# indices for each unique user
user_uni, user_id, changes_peruser = np.unique(user, return_inverse=True, return_counts=True)  # user_id is the index of each user in "user". And user_uni[user_id] == user

twind_sec = 1800
twind = int(twind_sec/tgrain) # window in which active users are considered co-active



usernet = gt.Graph(directed=False)


print(len(tsep), 'bins')

edges = []
invweight = []
num_users = np.zeros(len(tsep) - twind)
for i in range(0, len(tsep) - twind):
    # get the active users and coords in the time window
    users_in_tw = np.unique(user_id[inds_sep[i]:inds_sep[i+twind]]) # or do this as a dynamic network in bins of tgrain?
    x_in_tw = x[inds_sep[i]:inds_sep[i+twind]]
    y_in_tw = y[inds_sep[i]:inds_sep[i+twind]]
    num_users[i] = len(users_in_tw)

    # add the edges between the active users
    comb_inds = np.triu_indices(num_users[i], 1) # upper triangular indices without the diagonal
    edges += np.array([users_in_tw[comb_inds[0]], users_in_tw[comb_inds[1]]]).T.tolist()
    invweight += [num_users[i]]*len(comb_inds[0])

# unique edges and weights
print('get unique edges')
weight = 1/np.array(invweight)
# remove duplicate edges
#edges, weights = np.unique(np.array(edges), axis=0, return_counts=True) 
# remove duplicates and sum weights
edges, inv_unique = np.unique(np.array(edges), axis=0, return_inverse=True) 
weights = np.bincount(inv_unique, weights=weight)

print(len(edges), len(weights))

print(weights[:100])
print('adding edges')
usernet.add_edge_list(edges)
weight_property = usernet.new_ep("float")
for e, weight in zip(usernet.edges(), weights):
    weight_property[e] = weight
usernet.ep.weight = weight_property
print(np.max(usernet.ep.weight.a))

# add vertex property with changes per user
npixelchanges = usernet.new_vp("int")
for v, nchanges in zip(usernet.vertices(), changes_peruser):
    npixelchanges[v] = np.log(nchanges)
usernet.vp.npixelchanges = npixelchanges

print("n_users = ", len(np.unique(user)))
for prop_name, prop_map in usernet.edge_properties.items():
    print(f"Property name: {prop_name}, Property type: {prop_map.value_type()}")

# plot the network
# change the force of the graph layout
pos = gt.sfdp_layout(usernet, eweight=weight_property)
gt.graph_draw(usernet, 
              pos=pos,
              vertex_size=gt.prop_to_size(npixelchanges, mi=4, ma=50, power=1),
              edge_pen_width=gt.prop_to_size(weight_property, mi=0, ma=10, power=1),
              output=os.path.join(var.FIGS_PATH, 'usernetwork', 'testnet.pdf'),
              output_size=(2000,2000),
              edge_color=[0.179, 0.203, 0.210, 0.1], # default color but with more transparency
              bg_color='white')

# get the number of pixel changes per user
n_pixchanges = np.bincount(user_id)

# plot the degree distribution
plt.figure()
deg = usernet.get_out_degrees(usernet.get_vertices(), eweight=weight_property)
plt.hist(deg, bins=100)
plt.xlabel('Degree')
plt.ylabel('Count')
plt.xscale('log')
plt.yscale('log')
plt.savefig(os.path.join(var.FIGS_PATH, 'usernetwork', 'testnet_degdist.png'))

# plot degree distribution vs number of pixel changes
print(len(n_pixchanges), len(deg))
plt.figure()
plt.scatter(n_pixchanges, deg, alpha=0.4, s=0.5)
plt.xlabel('Number of pixel changes')
plt.ylabel('Degree')
plt.xscale('log')
plt.yscale('log')
plt.savefig(os.path.join(var.FIGS_PATH, 'usernetwork', 'testnet_deg_vs_pixchanges.png'))

# plot number of pixel changes vs time
plt.figure()
plt.hist(sec, bins=100)
plt.xlabel('Time (s)')
plt.ylabel('Number of pixel changes')
plt.savefig(os.path.join(var.FIGS_PATH, 'usernetwork', 'testnet_pixchanges_vs_time.png'))