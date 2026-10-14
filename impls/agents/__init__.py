from agents.crl import CRLAgent
from agents.gcbc import GCBCAgent
from agents.gciql import GCIQLAgent
from agents.gcivl import GCIVLAgent
from agents.hiql import HIQLAgent
from agents.qrl import QRLAgent
from agents.sac import SACAgent
from agents.lavl import LAVLAgent
from agents.offline_metra import OfflineMETRAAgent
from agents.online_metra import OnlineMETRAAgent

agents = dict(
    crl=CRLAgent,
    gcbc=GCBCAgent,
    gciql=GCIQLAgent,
    gcivl=GCIVLAgent,
    hiql=HIQLAgent,
    qrl=QRLAgent,
    sac=SACAgent,
    lavl=LAVLAgent,
    offline_metra=OfflineMETRAAgent,
    online_metra=OnlineMETRAAgent,
)
