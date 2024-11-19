REGISTRY = {}

from .central_rnn_agent import CentralRNNAgent
from .rnn_agent import RNNAgent
from .rnn_agent_rmix import RmixAgent
from .iqn_rnn_agent import IQNRNNAgent

REGISTRY["rnn"] = RNNAgent
REGISTRY["iqn_rnn"] = IQNRNNAgent
REGISTRY["rnn_agent_rmix"] = RmixAgent
REGISTRY["central_rnn"] = CentralRNNAgent
