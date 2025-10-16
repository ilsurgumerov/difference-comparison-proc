# tasks/state.py
class PartyState:
    """
    Контейнер состояния участника протокола.
    Хранит локальные shares и промежуточные результаты.
    """
    def __init__(self, rank: int, world_size: int, ell: int):
        self.rank = rank
        self.world_size = world_size
        self.ell = ell
        self.mod = 1 << ell
        self.local = {}

    def set(self, name, value):
        self.local[name] = value

    def get(self, name, default=None):
        return self.local.get(name, default)
