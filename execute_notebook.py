"""Execute notebook cells — simplified & reliable"""
import sys, io, os, traceback, base64
import nbformat
from nbformat.v4 import new_output

import matplotlib
matplotlib.use('Agg')

NB_PATH = 'notebook.ipynb'
with open(NB_PATH, 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

# Pre-populate namespace with all needed imports
SETUP = (
    "import pandas as pd\n"
    "import numpy as np\n"
    "import matplotlib\n"
    "matplotlib.use('Agg')\n"
    "import matplotlib.pyplot as plt\n"
    "import seaborn as sns\n"
    "import warnings, joblib, os\n"
    "from sklearn.model_selection import train_test_split\n"
    "from sklearn.preprocessing import LabelEncoder, StandardScaler\n"
    "from sklearn.ensemble import RandomForestClassifier\n"
    "from sklearn.linear_model import LogisticRegression\n"
    "from sklearn.metrics import (classification_report, confusion_matrix,\n"
    "    accuracy_score, f1_score)\n"
    "from imblearn.over_sampling import SMOTE\n"
    "from xgboost import XGBClassifier\n"
    "warnings.filterwarnings('ignore')\n"
    "sns.set_theme(style='whitegrid', palette='muted')\n"
    "plt.rcParams['figure.dpi'] = 100\n"
)
ns = {}
exec(SETUP, ns)
print("Namespace keys:", [k for k in ns if not k.startswith('_')])

exec_count = 1

for i, cell in enumerate(nb.cells):
    if cell['cell_type'] != 'code':
        continue
    src = ''.join(cell['source']).strip()
    if not src:
        continue

    cell['outputs'] = []
    cell['execution_count'] = exec_count
    exec_count += 1

    buf = io.StringIO()
    old_out = sys.stdout
    sys.stdout = buf
    err_text = None

    try:
        exec(src, ns)
    except Exception:
        err_text = traceback.format_exc()
    finally:
        sys.stdout = old_out

    stdout_val = buf.getvalue()
    if stdout_val:
        cell['outputs'].append(new_output(
            output_type='stream', name='stdout', text=stdout_val))
    if err_text:
        cell['outputs'].append(new_output(
            output_type='stream', name='stderr', text=err_text))
        print(f'  [ERR] cell {i:2d} | {err_text.splitlines()[-1][:70]}')
    else:
        # Capture any matplotlib figures
        import matplotlib.pyplot as _plt
        fignums = list(_plt.get_fignums())
        for fn in fignums:
            fig = _plt.figure(fn)
            fbuf = io.BytesIO()
            fig.savefig(fbuf, format='png', bbox_inches='tight', dpi=90)
            fbuf.seek(0)
            b64 = base64.b64encode(fbuf.read()).decode()
            cell['outputs'].append(new_output(
                output_type='display_data',
                data={'image/png': b64, 'text/plain': ['<Figure>']},
                metadata={}
            ))
        _plt.close('all')
        print(f'  [OK]  cell {i:2d} | {src[:60].replace(chr(10)," ")}')

with open(NB_PATH, 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)

sz = os.path.getsize(NB_PATH)
print(f'\nSaved: {sz:,} bytes — {exec_count-1} code cells')
