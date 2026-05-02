import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt_ref

code = """import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
"""
ns = {}
exec(code, ns)
print('plt in ns:', 'plt' in ns)
print('pd in ns:', 'pd' in ns)
print('type plt:', type(ns.get('plt')))

# Now test with sys.stdout redirect
import sys, io
buf = io.StringIO()
sys.stdout = buf
exec("print('hello from exec'); import matplotlib.pyplot as plt2; print(type(plt2))", ns)
sys.stdout = sys.__stdout__
print('captured:', buf.getvalue())
print('plt2 in ns:', 'plt2' in ns)
