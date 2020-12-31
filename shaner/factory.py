from shaner.value import TableValue

id2value = {
    "table": TableValue
}


class ValueFactory:
    def create(self, vid, **params):
        return id2value[vid](**params)
