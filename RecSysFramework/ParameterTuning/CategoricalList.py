from skopt.space import Categorical

class HashableListAsDict(dict):
    
    def __init__(self, arr):
        self.update({i:val for i, val in enumerate(arr)})

    def __hash__(self):
        return hash(tuple(sorted(self.items())))

    def __repr__(self):
        return str(list(self.values()))

    def __getitem__(self, key):
        return self.tolist()[key]

    def tolist(self):
        return list(self.values())


class CategoricalList(Categorical):

    def __init__(self, categories, **categorical_kwargs):
        super().__init__(self._convert_hashable(categories), **categorical_kwargs)

    def _convert_hashable(self, list_of_lists):
        return [HashableListAsDict(list_)
                for list_ in list_of_lists]

    def __contains__(self, point):
        return any(list(v for k, v in sorted(point.items())) == x.tolist() for x in self.categories)

