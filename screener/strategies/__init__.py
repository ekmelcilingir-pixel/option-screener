from .base import Candidate, Strategy
from .csp import CashSecuredPut
from .covered_call import CoveredCall
from .credit_spread import CreditSpread
from .earnings import EarningsPlay

ALL_STRATEGIES: list[Strategy] = [CashSecuredPut(), CoveredCall(), CreditSpread(), EarningsPlay()]

__all__ = ["Candidate", "Strategy", "ALL_STRATEGIES"]
