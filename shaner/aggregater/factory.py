from shaner.aggregater.core import Discretizer
from shaner.aggregater.dta import DTA

id2aggr = {
    "disc": Discretizer,
    "dta": DTA
}

class AggregaterFactory:
    def create(self, _id, params):
        return id2aggr[_id](**params)
