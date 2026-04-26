"""
谈判场景数据结构定义
"""
from dataclasses import dataclass, asdict, field, fields
import json
from typing import Any, Dict


@dataclass
class NegotiationScenario:
    scenario_id: str
    item_name: str           # 商品名称，如"二手 iPhone 15"
    item_description: str    # 商品描述
    buyer_budget: float      # 买家最高出价（隐藏底线）
    seller_cost: float       # 卖家最低售价（隐藏底线）
    market_ref_price: float  # 市场参考价（双方可见的公共信息）
    max_rounds: int = 10     # 最大谈判轮数
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, d: dict) -> "NegotiationScenario":
        valid_keys = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**filtered)

    @classmethod
    def from_json(cls, s: str) -> "NegotiationScenario":
        return cls.from_dict(json.loads(s))

    @property
    def bargaining_zone(self) -> float:
        """正的谈判空间 = buyer_budget - seller_cost"""
        return self.buyer_budget - self.seller_cost

    @property
    def midpoint_price(self) -> float:
        """双方底线的中间价"""
        return (self.buyer_budget + self.seller_cost) / 2

    @property
    def gap_ratio(self) -> float:
        """谈判空间占卖家成本的比例"""
        if self.seller_cost <= 0:
            return 0.0
        return (self.buyer_budget - self.seller_cost) / self.seller_cost
