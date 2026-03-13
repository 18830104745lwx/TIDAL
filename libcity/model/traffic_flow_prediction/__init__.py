try:
    from libcity.model.traffic_flow_prediction.TIDAL import TIDAL
except ImportError:
    pass

__all__ = [
    "TIDAL",
]
