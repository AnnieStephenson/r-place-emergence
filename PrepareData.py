import ray
import rplacem.io as rp

ray.init()
print('new ray job')
ray.get([rp.condense_data_part.remote(i*4,(i+1)*4) for i in range(0,6)])
print('new ray job')
ray.get([rp.condense_data_part.remote(24 + i*4, 24 + (i+1)*4) for i in range(0,6)])
print('new ray job')
ray.get([rp.condense_data_part.remote(48 + i*4, 48 + (i+1)*4) for i in range(0,6)])
print('new ray job')
ray.get([rp.condense_data_part.remote(72 + i*4, 72 + (i+1)*4) for i in range(0,2)])

#rp.condense_data_merge()
