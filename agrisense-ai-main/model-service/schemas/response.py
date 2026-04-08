"""
schemas/response.py
Pydantic request / response models for POST /predict.
"""

from typing import Literal, Optional
from pydantic import BaseModel, Field


class PredictionResponse(BaseModel):
    """Successful prediction — returned for fruit, leaf, and flower_cluster images."""

    image_type: Literal["fruit", "leaf", "flower_cluster"] = Field(
        ..., description="Category identified by the router transformer"
    )
    disease: str = Field(
        ..., description="Full disease name or cluster health label"
    )
    severity: str = Field(
        ...,
        description=(
            "Human-readable severity, e.g. 'Moderate — 35% area affected' "
            "or 'None — healthy tissue'"
        ),
    )
    confidence: str = Field(
        ..., description="Classifier confidence, e.g. '94.2%'"
    )
    recommendation: str = Field(
        ..., description="Actionable treatment or management advice"
    )
    details: str = Field(
        ..., description="Brief scientific explanation of the diagnosis"
    )
    flower_count: Optional[int] = Field(
        default=None,
        description="Number of individual flowers detected (flower_cluster images only)"
    )


class ErrorResponse(BaseModel):
    """Returned (HTTP 422) when the image cannot be classified."""

    error: str = Field(
        default=(
            "Could not classify image. "
            "Please upload a clear image of an apple leaf, fruit, or flower cluster."
        )
    )
