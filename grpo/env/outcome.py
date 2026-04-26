"""
谈判结果枚举

Outcome 表示一场谈判的终止状态。
"""
from enum import Enum


class Outcome(Enum):
    ONGOING = "ongoing"                    # 尚未终止
    DEAL = "deal"                          # 合法成交
    VIOLATION_BUYER = "violation_buyer"    # buyer 报价超预算的违规成交
    VIOLATION_SELLER = "violation_seller"  # seller 报价低于成本的违规成交
    WALKAWAY = "walkaway"                  # 一方主动放弃
    TIMEOUT = "timeout"                    # 超过 max_rounds 未成交
    FORMAT_ERROR = "format_error"          # 格式违规超过预算

    @property
    def is_deal(self) -> bool:
        """是否发生了成交（含违规成交）"""
        return self in (Outcome.DEAL, Outcome.VIOLATION_BUYER, Outcome.VIOLATION_SELLER)

    @property
    def is_terminal(self) -> bool:
        """是否已终止"""
        return self != Outcome.ONGOING
