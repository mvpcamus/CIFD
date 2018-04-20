# image.py scale configuration

# for old testbench
CVM_MINMAX_SET1 = {
  'gmin': {'c':-1.5, 'v':-4.0, 'm':-130},
  'gmax': {'c': 1.5, 'v': 4.0, 'm': 130}
}

CVM_MINMAX_SET2 = {
  'gmin': {'c':-1.5, 'v':-0.01, 'm':-0.01},
  'gmax': {'c': 1.5, 'v': 0.01, 'm': 0.01}
}

CVM_MINMAX_SET3 = {
  'gmin': {'c':-0.01, 'v':-0.01, 'm':-130},
  'gmax': {'c': 0.01, 'v': 0.01, 'm': 130}
}

# for new testbench
CVM_MINMAX_SET4 = {
  'gmin': {'c':-15.0, 'v':-0.45, 'm':-40},
  'gmax': {'c': 15.0, 'v': 0.45, 'm': 40}
}

CVM_MINMAX_SET5 = {
  'gmin': {'c':-15.0, 'v':-0.01, 'm':-0.01},
  'gmax': {'c': 15.0, 'v': 0.01, 'm': 0.01}
}

CVM_MINMAX_SET6 = {
  'gmin': {'c':-0.01, 'v':-0.01, 'm':-40},
  'gmax': {'c': 0.01, 'v': 0.01, 'm': 40}
}

CVM_MINMAX = [
  CVM_MINMAX_SET1, CVM_MINMAX_SET2, CVM_MINMAX_SET3,
  CVM_MINMAX_SET4, CVM_MINMAX_SET5, CVM_MINMAX_SET6
]
