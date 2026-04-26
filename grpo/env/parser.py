"""
DialogueParser：从模型生成文本中提取报价 / deal / walkaway 动作

解析优先级：WALKAWAY > DEAL > OFFER > INVALID
"""
import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class ParseResult:
    action_type: str              # "offer" | "deal" | "walkaway" | "invalid"
    price: Optional[float]        # offer / deal 的数字；walkaway / invalid 为 None
    raw_text: str
    is_format_valid: bool         # 是否符合格式规范（有 offer/deal/walkaway 之一）


class DialogueParser:
    """
    解析规则：
    - [报价：X] 或 [报价:X]  → offer, price=X
    - <deal>X</deal>          → deal, price=X
    - <walkaway> 或 <walkaway/>  → walkaway
    - 以上都没有              → invalid
    """

    PRICE_RE = re.compile(r'\[报价[：:]\s*(\d+(?:\.\d+)?)\s*\]')
    DEAL_RE  = re.compile(r'<deal>\s*(\d+(?:\.\d+)?)\s*</deal>')
    WALK_RE  = re.compile(r'<walkaway\s*/?>')

    def parse(self, utterance: str) -> ParseResult:
        text = utterance or ""

        # 优先级 1：walkaway
        if self.WALK_RE.search(text):
            return ParseResult(
                action_type="walkaway",
                price=None,
                raw_text=text,
                is_format_valid=True,
            )

        # 优先级 2：deal
        m = self.DEAL_RE.search(text)
        if m:
            return ParseResult(
                action_type="deal",
                price=float(m.group(1)),
                raw_text=text,
                is_format_valid=True,
            )

        # 优先级 3：offer
        m = self.PRICE_RE.search(text)
        if m:
            return ParseResult(
                action_type="offer",
                price=float(m.group(1)),
                raw_text=text,
                is_format_valid=True,
            )

        # 无法解析
        return ParseResult(
            action_type="invalid",
            price=None,
            raw_text=text,
            is_format_valid=False,
        )
