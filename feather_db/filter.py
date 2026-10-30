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

    def valid_between(self, after=None, before=None):
        """By valid time — when it happened, not when it was recorded
        (Metadata.valid_at, else the timestamp). Inclusive; either bound
        may be omitted."""
        if after is not None:
            self._filter.valid_after = int(after)
        if before is not None:
            self._filter.valid_before = int(before)
        return self
    
    def min_importance(self, v):
        self._filter.importance_gte = float(v)
        return self

    def min_confidence(self, v):
        """Only records at least this certain.

        `importance` is how much a record matters; `confidence` is how sure we
        are it is true. Keeping them separate is deliberate — a critical rule
        believed with low certainty and a trivial fact known for sure are not
        the same thing, and collapsing them loses the distinction that makes
        either useful.
        """
        self._filter.confidence_gte = float(v)
        return self
    
    def contains_tags(self, tags):
        if not isinstance(tags, list):
            tags = [tags]
        self._filter.tags_contains = tags
        return self

    def namespace(self, ns: str) -> "FilterBuilder":
        self._filter.namespace_id = ns
        return self

    def namespace_prefix(self, prefix: str) -> "FilterBuilder":
        """`prefix` and every namespace below it, '.' separating levels:
        "acme.u1" matches "acme.u1" and "acme.u1.prefs", not "acme.u10".
        Indexed: the search ranks within those namespaces, not over a global
        top-k filtered afterwards."""
        self._filter.namespace_prefix = prefix
        return self

    def entity(self, eid: str) -> "FilterBuilder":
        self._filter.entity_id = eid
        return self

    def attribute(self, key: str, value: str) -> "FilterBuilder":
        current = self._filter.attributes_match
        if current is None:
            self._filter.attributes_match = {key: value}
        else:
            current[key] = value
            self._filter.attributes_match = current
        return self

    def build(self):
        return self._filter
