"""
ViMedAgent — Framework registry
Exports all run_* functions for use by run_experiments.py
"""

from frameworks.tier1.zero_shot    import run_zero_shot
from frameworks.tier1.cot          import run_cot
from frameworks.tier1.cot_sc       import run_cot_sc
from frameworks.tier1.self_refine  import run_self_refine

from frameworks.tier2.voting_ensemble import run_voting_ensemble

from frameworks.tier3.medagents    import run_medagents
from frameworks.tier3.medagents_vi import run_medagents_vi
from frameworks.tier3.mdagents     import run_mdagents
from frameworks.tier3.mdagents_vi  import run_mdagents_vi
from frameworks.tier3.mad_debate   import run_mad_debate

__all__ = [
    "run_zero_shot", "run_cot", "run_cot_sc", "run_self_refine",
    "run_voting_ensemble",
    "run_medagents", "run_medagents_vi",
    "run_mdagents",  "run_mdagents_vi",
    "run_mad_debate",
]
