import ray
import CondenseData as cond

year=2023
maxfilenum = 52 if (year == 2023) else 78

condenser = cond.DataCondenser(year) #.remote(year)

'''
ray.init(ignore_reinit_error=True)
print('new ray job')
ray.get([condenser.condense_part.remote(i*4,(i+1)*4) for i in range(0,6)])
print('new ray job')
ray.get([condenser.condense_part.remote(24 + i*4, 24 + (i+1)*4) for i in range(0,6)])
print('new ray job')
if maxfilenum == 52:
    ray.get([condenser.condense_part.remote(48, 51), condenser.condense_part.remote(51, 53)])
else:
    ray.get([condenser.condense_part.remote(48 + i*4, 48 + (i+1)*4) for i in range(0,6)])
if maxfilenum > 72:
    print('new ray job')
    ray.get([condenser.condense_part.remote(72 + i*4, 72 + (i+1)*4) for i in range(0,2)])
'''

#print('merging')
#condenser.Merging()

#print('sorting')
#condenser.Sorting()

#print('remove duplicates')
#condenser.remove_duplicates()
#condenser.tag_hidden_mod_changes()

#condenser.misc_checks()
#condenser.clean_data_dir()
