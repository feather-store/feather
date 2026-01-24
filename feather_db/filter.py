import time
from .core import SearchFilter, ContextType

class FilterBuilder:
    def __init__(self):
        self._filter = SearchFilter()
    
    def types(self, types_list):
        if not isinstance(types_list, list):
            types_list = [types_list]
        self._filter.types = types_list
        return self
    
    def source(self, s):
        self._filter.source = s
        return self

    def source_prefix(self, p):
        self._filter.source_prefix = p
        return self
    
    def after(self, ts):
        self._filter.timestamp_after = int(ts)
        return self

    def before(self, ts):
        self._filter.timestamp_before = int(ts)
        return self
    
    def min_importance(self, v):
        self._filter.importance_gte = float(v)
        return self
    
    def contains_tags(self, tags):
        if not isinstance(tags, list):
            tags = [tags]
        self._filter.tags_contains = tags
        return self

    def build(self):
        return self._filter
