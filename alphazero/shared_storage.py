import copy

from .network import Network, make_uniform_network

class SharedStorage(object):

  def __init__(self):
    self._networks = {}
    self._latest_live = None

  def latest_network(self) -> Network:
    if self._latest_live is not None:
      return self._latest_live
    if self._networks:
      # Materialize a fresh Network so callers can't mutate the stored best.
      step = max(self._networks.keys())
      net = make_uniform_network()
      net.load_state_dict(copy.deepcopy(self._networks[step]))
      return net
    else:
      return make_uniform_network()  # policy -> uniform, value -> 0.5

  def use_live_network(self, network: Network):
    """Keep a direct reference to the latest network (no copies)."""
    self._latest_live = network

  def save_network(self, step: int, network: Network):
    if self._latest_live is not None:
      # In live mode, just keep the reference updated.
      self._latest_live = network
      return
    # Store a detached copy of the weights to avoid in-place mutation later.
    state_copy = {
        k: v.detach().cpu().clone()
        for k, v in network.state_dict().items()
    }
    self._networks[step] = state_copy
