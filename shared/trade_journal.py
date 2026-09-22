"""
Human Trade Journal (Steenbarger / Douglas)
=============================================

Provides a structured human-facing trade journal for recording
subjective observations, emotions, and daily reviews — distinct
from the ML-centric TradeMemory.

Inspired by:
- The Daily Trading Coach (Brett Steenbarger): self-coaching
- Trading in the Zone (Mark Douglas): mindset + discipline tracking

Features:
- Pre-trade checklist with mood/confidence scoring
- Post-trade reflection with lesson extraction
- End-of-day review with daily grade
- CSV/JSON persistence for portability
- Performance patterns by mood and discipline scores

Usage:
    from shared.trade_journal import TradeJournal
    journal = TradeJournal()
    journal.pre_trade_check(symbol="AAPL", mood=7, confidence=8, setup="CAN SLIM breakout")
    journal.log_trade(symbol="AAPL", direction="LONG", pnl=250.0, lesson="Patience paid off")
    journal.daily_review(grade="B", notes="Followed rules on 3/4 trades")
    journal.save()
"""

from __future__ import annotations

import csv
import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from shared.utils.paths import stocks_plugin_subdir

logger = logging.getLogger(__name__)

_DEFAULT_JOURNAL_DIR = stocks_plugin_subdir("journal", create=False)


@dataclass
class PreTradeEntry:
    """Pre-trade checklist entry (Douglas mindset check)."""
    timestamp: str
    symbol: str
    mood_score: int  # 1-10
    confidence_score: int  # 1-10
    setup_type: str
    market_condition: str = ""
    notes: str = ""
    checklist_passed: bool = True


@dataclass
class TradeLogEntry:
    """Post-trade reflection entry."""
    timestamp: str
    symbol: str
    direction: str
    entry_price: float = 0.0
    exit_price: float = 0.0
    pnl: float = 0.0
    mood_score: int = 5
    followed_plan: bool = True
    lesson: str = ""
    mistake: str = ""
    tags: List[str] = field(default_factory=list)


@dataclass
class DailyReviewEntry:
    """End-of-day review entry (Steenbarger self-coaching)."""
    date: str
    grade: str  # A, B, C, D, F
    total_trades: int = 0
    total_pnl: float = 0.0
    rules_followed_pct: float = 100.0
    biggest_win: float = 0.0
    biggest_loss: float = 0.0
    emotional_state: str = ""
    what_went_well: str = ""
    what_to_improve: str = ""
    tomorrow_focus: str = ""


class TradeJournal:
    """Human-facing trade journal for psychology and discipline tracking.

    Args:
        journal_dir: Directory for storing journal files.
    """

    def __init__(self, journal_dir: Optional[str] = None) -> None:
        self._dir = journal_dir or _DEFAULT_JOURNAL_DIR
        Path(self._dir).mkdir(parents=True, exist_ok=True)

        self._pre_trades: List[PreTradeEntry] = []
        self._trade_logs: List[TradeLogEntry] = []
        self._daily_reviews: List[DailyReviewEntry] = []

        self._load()

    def pre_trade_check(
        self,
        symbol: str,
        mood: int = 5,
        confidence: int = 5,
        setup: str = "",
        market_condition: str = "",
        notes: str = "",
    ) -> bool:
        """Record a pre-trade checklist entry.

        Returns True if the checklist passes (mood >= 4 and confidence >= 4).
        Returns False if trader should skip this trade.
        """
        passed = mood >= 4 and confidence >= 4
        entry = PreTradeEntry(
            timestamp=datetime.now().isoformat(),
            symbol=symbol,
            mood_score=max(1, min(10, mood)),
            confidence_score=max(1, min(10, confidence)),
            setup_type=setup,
            market_condition=market_condition,
            notes=notes,
            checklist_passed=passed,
        )
        self._pre_trades.append(entry)

        if not passed:
            logger.warning(
                "PRE-TRADE CHECK FAILED for %s: mood=%d, confidence=%d — consider skipping",
                symbol, mood, confidence,
            )
        else:
            logger.info(
                "Pre-trade check passed for %s: mood=%d, confidence=%d, setup=%s",
                symbol, mood, confidence, setup,
            )
        return passed

    def log_trade(
        self,
        symbol: str,
        direction: str = "LONG",
        entry_price: float = 0.0,
        exit_price: float = 0.0,
        pnl: float = 0.0,
        mood: int = 5,
        followed_plan: bool = True,
        lesson: str = "",
        mistake: str = "",
        tags: Optional[List[str]] = None,
    ) -> None:
        """Log a completed trade with subjective reflection."""
        entry = TradeLogEntry(
            timestamp=datetime.now().isoformat(),
            symbol=symbol,
            direction=direction.upper(),
            entry_price=entry_price,
            exit_price=exit_price,
            pnl=pnl,
            mood_score=max(1, min(10, mood)),
            followed_plan=followed_plan,
            lesson=lesson,
            mistake=mistake,
            tags=tags or [],
        )
        self._trade_logs.append(entry)
        logger.info(
            "Trade logged: %s %s pnl=$%.2f lesson='%s'",
            direction, symbol, pnl, lesson,
        )

    def daily_review(
        self,
        grade: str = "C",
        emotional_state: str = "",
        what_went_well: str = "",
        what_to_improve: str = "",
        tomorrow_focus: str = "",
    ) -> DailyReviewEntry:
        """Record an end-of-day review (Steenbarger self-coaching).

        Auto-calculates today's trade stats from logged trades.
        """
        today = date.today().isoformat()
        today_trades = [t for t in self._trade_logs if t.timestamp.startswith(today)]

        total_pnl = sum(t.pnl for t in today_trades)
        rules_pct = (
            sum(1 for t in today_trades if t.followed_plan) / len(today_trades) * 100
            if today_trades else 100.0
        )
        wins = [t.pnl for t in today_trades if t.pnl > 0]
        losses = [t.pnl for t in today_trades if t.pnl < 0]

        entry = DailyReviewEntry(
            date=today,
            grade=grade.upper(),
            total_trades=len(today_trades),
            total_pnl=total_pnl,
            rules_followed_pct=rules_pct,
            biggest_win=max(wins) if wins else 0.0,
            biggest_loss=min(losses) if losses else 0.0,
            emotional_state=emotional_state,
            what_went_well=what_went_well,
            what_to_improve=what_to_improve,
            tomorrow_focus=tomorrow_focus,
        )
        self._daily_reviews.append(entry)
        logger.info(
            "Daily review: grade=%s, trades=%d, pnl=$%.2f, rules=%.0f%%",
            grade, len(today_trades), total_pnl, rules_pct,
        )
        return entry

    def get_performance_by_mood(self) -> Dict[int, Dict[str, float]]:
        """Analyse trade performance grouped by mood score."""
        from collections import defaultdict
        mood_groups: Dict[int, list] = defaultdict(list)
        for t in self._trade_logs:
            mood_groups[t.mood_score].append(t.pnl)

        result = {}
        for mood, pnls in sorted(mood_groups.items()):
            result[mood] = {
                "avg_pnl": sum(pnls) / len(pnls),
                "total_trades": len(pnls),
                "win_rate": sum(1 for p in pnls if p > 0) / len(pnls) * 100,
            }
        return result

    def get_discipline_stats(self) -> Dict[str, Any]:
        """Get discipline tracking statistics."""
        if not self._trade_logs:
            return {"total_trades": 0, "plan_adherence_pct": 0}

        total = len(self._trade_logs)
        followed = sum(1 for t in self._trade_logs if t.followed_plan)
        avg_grade_map = {"A": 4, "B": 3, "C": 2, "D": 1, "F": 0}
        grades = [avg_grade_map.get(r.grade, 2) for r in self._daily_reviews]

        return {
            "total_trades": total,
            "plan_adherence_pct": followed / total * 100,
            "avg_daily_grade": sum(grades) / len(grades) if grades else 0,
            "total_reviews": len(self._daily_reviews),
        }

    # ─── Persistence ───

    def save(self) -> None:
        """Save all journal data to disk."""
        data = {
            "pre_trades": [asdict(e) for e in self._pre_trades],
            "trade_logs": [asdict(e) for e in self._trade_logs],
            "daily_reviews": [asdict(e) for e in self._daily_reviews],
            "saved_at": datetime.now().isoformat(),
        }
        path = os.path.join(self._dir, "journal.json")
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            logger.info("Trade journal saved to %s", path)
        except Exception as e:
            logger.error("Failed to save journal: %s", e)

    def _load(self) -> None:
        """Load journal data from disk."""
        path = os.path.join(self._dir, "journal.json")
        if not os.path.exists(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._pre_trades = [PreTradeEntry(**e) for e in data.get("pre_trades", [])]
            self._trade_logs = [TradeLogEntry(**e) for e in data.get("trade_logs", [])]
            self._daily_reviews = [DailyReviewEntry(**e) for e in data.get("daily_reviews", [])]
            logger.info(
                "Journal loaded: %d pre-trades, %d trades, %d reviews",
                len(self._pre_trades), len(self._trade_logs), len(self._daily_reviews),
            )
        except Exception as e:
            logger.debug("Failed to load journal: %s", e)

    def export_csv(self, path: Optional[str] = None) -> str:
        """Export trade log to CSV file."""
        csv_path = path or os.path.join(self._dir, "trades.csv")
        try:
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                if self._trade_logs:
                    writer = csv.DictWriter(f, fieldnames=asdict(self._trade_logs[0]).keys())
                    writer.writeheader()
                    for entry in self._trade_logs:
                        row = asdict(entry)
                        row["tags"] = ",".join(row["tags"])
                        writer.writerow(row)
            logger.info("Trades exported to %s", csv_path)
            return csv_path
        except Exception as e:
            logger.error("Failed to export CSV: %s", e)
            return ""
