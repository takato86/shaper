from shaner.value import TableValue
from shaner.aggregater.core import Discretizer, NDiscretizer
from shaner.aggregater.subgoal_based import DTA

id2aggr = {
    "disc": Discretizer,
    "ndisc": NDiscretizer,
    "dta": DTA
}

id2value = {
    "table": TableValue
}


class ValueFactory:
    @staticmethod
    def create(vid, n_states, values=None):
        return id2value[vid](n_states, values)


class AggregaterFactory:
    @staticmethod
    def create(_id, abstractor):
        # abstractorはsplitterとachieverの抽象エンティティ
        # TODO クラスとして実装
        return id2aggr[_id](abstractor)
