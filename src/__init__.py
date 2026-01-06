"""
Vietnam Housing Price Prediction Package

A complete machine learning package for predicting housing prices in Hanoi, Vietnam.
"""

__version__ = "1.0.0"
__author__ = "Vietnam Housing Team"

from .preprocessing import HousingDataPreprocessor
from .model import HousingPriceModel
from . import utils

__all__ = [
    "HousingDataPreprocessor",
    "HousingPriceModel",
    "utils",
]
