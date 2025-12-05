# app/schemas/input.py
from pydantic import BaseModel, Field


class AddressData(BaseModel):
    street_tokens: list[str] = Field(default_factory=list)
    city: str = ""
    state: str = ""
    zipcode: str = ""
    address1: str = ""
    address2: str = ""
    address: str = ""
    latitude: float | None = None
    longitude: float | None = None

    # Make the instance callable -> returns the full address string
    def __call__(self) -> str:
        return self.address

    # Make str(instance) also return the full address
    def __str__(self) -> str:
        return self.address

    # Nice repr for logs
    def __repr__(self) -> str:
        return f"AddressData({self.address!r})"


class QueryInput(BaseModel):
    url: str
    beds: int
    baths: float
    accommodates: int
