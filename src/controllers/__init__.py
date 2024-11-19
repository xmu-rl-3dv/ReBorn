REGISTRY = {}

from .basic_controller import BasicMAC
from .basic_central_controller import CentralBasicMAC
from .multi_controller import MultiMAC
from .risk_controller import RiskMAC #added for riskq
from .rmix_controller import RmixMAC
from .dmix_controller import DmixMAC
REGISTRY['risk_mac'] = RiskMAC #added for riskq
REGISTRY["basic_central_mac"] = CentralBasicMAC
REGISTRY["basic_mac"] = BasicMAC
REGISTRY["multi_mac"] = MultiMAC
REGISTRY["rmix_mac"] = RmixMAC
REGISTRY["dmix_mac"] = DmixMAC

