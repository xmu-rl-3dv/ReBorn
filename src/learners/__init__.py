from .q_learner import QLearner
from .dmaq_qatten_learner import DMAQ_qattenLearner
from .rmix_learner import RMIXLearner
from .iqn_learner import IQNLearner


REGISTRY = {}

REGISTRY["q_learner"] = QLearner
REGISTRY["dmaq_qatten_learner"] = DMAQ_qattenLearner
REGISTRY["rmix_learner"] = RMIXLearner
REGISTRY["iqn_learner"] = IQNLearner
