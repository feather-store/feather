"""
Domain-specific profile adapters for Feather DB.

The C++ core stores generic namespace_id / entity_id / attributes fields.
DomainProfile subclasses map domain vocabulary onto those generic fields.
"""

from .core import Metadata


class DomainProfile:
    """
    Base class. Wraps a Metadata object and provides typed attribute helpers.
    Subclass this to create domain-specific adapters (marketing, healthcare, etc.).
    """

    def __init__(self, meta: Metadata = None):
        self._meta = meta if meta is not None else Metadata()

    # --- Generic accessors ---

    def set_namespace(self, value: str) -> "DomainProfile":
        """Set the partition / ownership key (brand, org, tenant)."""
        self._meta.namespace_id = value
        return self

    def set_entity(self, value: str) -> "DomainProfile":
        """Set the subject key (user, customer, product, patient)."""
        self._meta.entity_id = value
        return self

    def set_attr(self, key: str, value) -> "DomainProfile":
        """Store a domain-specific key-value pair. Value is coerced to str."""
        attrs = self._meta.attributes
        attrs[key] = str(value)
        self._meta.attributes = attrs
        return self

    def get_attr(self, key: str, default=None):
        """Retrieve a stored attribute value, or default if absent."""
        return self._meta.attributes.get(key, default)

    def to_metadata(self) -> Metadata:
        """Return the underlying Metadata object."""
        return self._meta


class MarketingProfile(DomainProfile):
    """
    Digital Marketing domain adapter.

    Mapping:
        namespace_id  → brand_id
        entity_id     → user_id
        attributes    → channel, campaign_id, ctr, roas, platform, …
    """

    def __init__(self, meta: Metadata = None):
        super().__init__(meta)

    # --- Domain-level setters ---

    def set_brand(self, brand_id: str) -> "MarketingProfile":
        """Set brand_id (maps to namespace_id)."""
        return self.set_namespace(brand_id)

    def set_user(self, user_id: str) -> "MarketingProfile":
        """Set user_id (maps to entity_id)."""
        return self.set_entity(user_id)

    def set_channel(self, channel: str) -> "MarketingProfile":
        """Set acquisition / engagement channel (e.g. 'instagram', 'email')."""
        return self.set_attr("channel", channel)

    def set_campaign(self, campaign_id: str) -> "MarketingProfile":
        """Set campaign identifier."""
        return self.set_attr("campaign_id", campaign_id)

    def set_ctr(self, ctr: float) -> "MarketingProfile":
        """Set click-through rate (0.0–1.0)."""
        return self.set_attr("ctr", ctr)

    def set_roas(self, roas: float) -> "MarketingProfile":
        """Set return on ad spend."""
        return self.set_attr("roas", roas)

    def set_platform(self, platform: str) -> "MarketingProfile":
        """Set ad platform (e.g. 'meta', 'google', 'tiktok')."""
        return self.set_attr("platform", platform)

    # --- Domain-level readers ---

    @property
    def brand_id(self) -> str:
        return self._meta.namespace_id

    @property
    def user_id(self) -> str:
        return self._meta.entity_id

    @property
    def channel(self) -> str:
        return self.get_attr("channel", "")

    @property
    def campaign_id(self) -> str:
        return self.get_attr("campaign_id", "")

    @property
    def ctr(self) -> float:
        v = self.get_attr("ctr")
        return float(v) if v is not None else 0.0

    @property
    def roas(self) -> float:
        v = self.get_attr("roas")
        return float(v) if v is not None else 0.0

    @property
    def platform(self) -> str:
        return self.get_attr("platform", "")
