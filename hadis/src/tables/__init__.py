"""Pareto-optimal cascade configuration tables (offline profiling output).

Each table is the lookup table HADIS's Resource Manager plans over: rows of
``(model_pair, router_thres, conf_thres, route_ratio)`` with the measured FID of
that operating point.  Producing them is the paper's offline profiling stage
(~78 GPU-hours); they are shipped as inputs and are not regenerated here.

Select one with ``--cascade_table`` on the controller.  ``hybrid`` is HADIS as
evaluated in the paper and is the default.

    hybrid       full hybrid routing over adjacent model pairs  (paper Figure 7)
    fixed13      ablation: fixed pair (sd35turbo, sd35large)
    disc_only    ablation: discriminator only
    router_only  ablation: router only
"""
import importlib

AVAILABLE = ['hybrid', 'fixed13', 'disc_only', 'router_only']
DEFAULT = 'hybrid'


def load_cascade_table(name=DEFAULT):
    """Return (comb_thres_config, fid_config) for the named table."""
    if name not in AVAILABLE:
        raise ValueError(f'Unknown cascade table {name!r}; available: {AVAILABLE}')
    module = importlib.import_module(f'tables.{name}')
    comb, fid = module.COMB_THRES_CONFIG, module.FID_CONFIG
    if len(comb) != len(fid):
        raise ValueError(f'Table {name!r} is inconsistent: '
                         f'{len(comb)} configurations vs {len(fid)} FID values')
    return comb, fid
