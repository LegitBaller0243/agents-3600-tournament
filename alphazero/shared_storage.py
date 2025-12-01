import copy

from .network import Network, make_uniform_network

class SharedStorage(object):

  def __init__(self):
    self._networks = {}

  def latest_network(self) -> Network:
    if self._networks:
      # Materialize a fresh Network so callers can't mutate the stored best.
      step = max(self._networks.keys())
      net = make_uniform_network()
      net.load_state_dict(copy.deepcopy(self._networks[step]))
      return net
    else:
      return make_uniform_network()  # policy -> uniform, value -> 0.5

  def save_network(self, step: int, network: Network):
    # Store a detached copy of the weights to avoid in-place mutation later.
    state_copy = {
        k: v.detach().cpu().clone()
        for k, v in network.state_dict().items()
    }
    self._networks[step] = state_copy
