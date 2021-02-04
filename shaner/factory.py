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
    def create(self, vid, **params):
        return id2value[vid](**params)


class AggregaterFactory:
    def create(self, _id, params):
        return id2aggr[_id](**params)
