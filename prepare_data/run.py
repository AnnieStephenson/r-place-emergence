import ray
import CondenseData as cond


'''
#8 minutes for all ray jobs on Guillaume's machine
ray.init()
print('new ray job')
ray.get([cond.condense_part.remote(i*4,(i+1)*4) for i in range(0,6)])
print('new ray job')
ray.get([cond.condense_part.remote(24 + i*4, 24 + (i+1)*4) for i in range(0,6)])
print('new ray job')
ray.get([cond.condense_part.remote(48 + i*4, 48 + (i+1)*4) for i in range(0,6)])
print('new ray job')
ray.get([cond.condense_part.remote(72 + i*4, 72 + (i+1)*4) for i in range(0,2)])
'''

#cond.Merging()

#cond.Sorting()

#cond.remove_duplicates()

cond.misc_checks()
#cond.clean_data_dir()
