"""
pricing.py

Single source of truth for the cited, dated external prices used by
run_external_api.py and cost_comparison.py, so the same $/unit number
isn't duplicated (and can't quietly drift) between the two files.

Sources, checked 2026-09-01:
  - OpenAI text-embedding-3-small: https://platform.openai.com/docs/models/text-embedding-3-small
  - AWS EC2 c7g.xlarge (4 vCPU, 8 GiB, Graviton3) on-demand, us-east-1:
    https://instances.vantage.sh/aws/ec2/c7g.xlarge
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class CitedPrice:
    usd: float
    unit: str
    source: str
    checked_date: str


OPENAI_TEXT_EMBEDDING_3_SMALL = CitedPrice(
    usd=0.02,
    unit="per 1M tokens",
    source="https://platform.openai.com/docs/models/text-embedding-3-small",
    checked_date="2026-09-01",
)

AWS_C7G_XLARGE = CitedPrice(
    usd=0.145,
    unit="per hour",
    source="https://instances.vantage.sh/aws/ec2/c7g.xlarge",
    checked_date="2026-09-01",
)
