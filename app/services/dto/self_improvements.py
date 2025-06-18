from dataclasses import dataclass
from typing import List

@dataclass
class ReflectionResult:
    """Self-reflection 결과"""
    score: float  # 0-10 점수
    issues: List[str]  # 발견된 문제점
    suggestions: List[str]  # 개선 제안
    overall_assessment: str  # 전체 평가


@dataclass
class ImprovementIteration:
    """개선 반복 기록"""
    iteration: int
    original_response: str
    reflection_result: ReflectionResult
    improved_response: str
    improvement_reason: str
    timestamp: float