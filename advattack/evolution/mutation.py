import numpy as np

header_len = 44
data_max = 32767
data_min = -32768
mutation_p = 0.0005


# #### return the mutation of the parent with sigma = 1(in order eps = 16), N(0,1)
def mutate(parent, eps_limit, byteorder_type):
     ba = bytearray(parent)
     step = 2
     sigma = eps_limit / 16
     for i in range(header_len, len(parent), step):
         int_parent = int.from_bytes(ba[i:i+2], byteorder=byteorder_type, signed=True)
         new_int_parent = min(data_max, max(data_min, int_parent + round(np.random.normal(0,1) * sigma)))
         new_bytes = int(new_int_parent).to_bytes(2, byteorder=byteorder_type, signed=True)
         ba[i] = new_bytes[0]
         ba[i+1] = new_bytes[1]
     return bytes(ba)

#### mutation defined by Alzantot:
# def mutate(parent, eps_limit, byteorder_type):
#    ba = bytearray(parent)
#    step = 2
#    for i in range(header_len, len(parent), step):
#       if np.random.random() < mutation_p:
#           int_parent = int.from_bytes(ba[i:i+2], byteorder=byteorder_type, signed=True)
#           new_int_parent = min(data_max, max(data_min, int_parent + np.random.choice(range(-eps_limit, eps_limit))))
#           new_bytes = int(new_int_parent).to_bytes(2, byteorder=byteorder_type, signed=True)
#           ba[i] = new_bytes[0]
#           ba[i+1] = new_bytes[1]
#    return bytes(ba)