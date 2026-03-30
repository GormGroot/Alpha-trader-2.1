"""
LLM Client — wrapper til Claude API (Anthropic).

Bruges af morning_briefing, evening_analysis og ad-hoc analyser.
Holder styr på token-forbrug og omkostninger.

Budget: ~$30-50/måned (claude-3-haiku primært, sonnet til vigtige analyser).
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from loguru import logger


# ── Token Tracker ────────────────────────────────────────────

@dataclass
class TokenUsage:
    """Token-forbrug for ét API-kald."""
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def estimated_cost_usd(self) -> float:
        """Estimeret pris i USD."""
        # Claude 3.5 Haiku priser (per 1M tokens)
        if "haiku" in self.model.lower():
            return (self.input_tokens * 0.25 + self.output_tokens * 1.25) / 1_000_000
        # Claude 3.5 Sonnet
        elif "sonnet" in self.model.lower():
            return (self.input_tokens * 3.0 + self.output_tokens * 15.0) / 1_000_000
        # Claude 3 Opus
        elif "opus" in self.model.lower():
            return (self.input_tokens * 15.0 + self.output_tokens * 75.0) / 1_000_000
        # Default (sonnet)
        return (self.input_tokens * 3.0 + self.output_tokens * 15.0) / 1_000_000


class TokenTracker:
    """Tracker for API token-forbrug med SQLite persistence."""

    def __init__(self, db_path: str = "data_cache/token_usage.db") -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS token_usage (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model TEXT NOT NULL,
                    input_tokens INTEGER NOT NULL,
                    output_tokens INTEGER NOT NULL,
                    cost_usd REAL NOT NULL,
                    purpose TEXT,
                    timestamp TEXT NOT NULL
                )
            """)

    def record(self, usage: TokenUsage, purpose: str = "") -> None:
        """Registrér token-forbrug."""
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                """INSERT INTO token_usage
                   (model, input_tokens, output_tokens, cost_usd, purpose, timestamp)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (usage.model, usage.input_tokens, usage.output_tokens,
                 usage.estimated_cost_usd, purpose, usage.timestamp.isoformat()),
            )

    def get_daily_cost(self, date: str | None = None) -> float:
        """Total omkostning for en dag (default: i dag)."""
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        with sqlite3.connect(self._db_path) as conn:
            row = conn.execute(
                "SELECT SUM(cost_usd) FROM token_usage WHERE timestamp LIKE ?",
                (f"{date}%",),
            ).fetchone()
            return row[0] or 0.0

    def get_monthly_cost(self, year: int | None = None, month: int | None = None) -> float:
        """Total omkostning for en måned."""
        now = datetime.now()
        year = year or now.year
        month = month or now.month
        prefix = f"{year}-{month:02d}"
        with sqlite3.connect(self._db_path) as conn:
            row = conn.execute(
                "SELECT SUM(cost_usd) FROM token_usage WHERE timestamp LIKE ?",
                (f"{prefix}%",),
            ).fetchone()
            return row[0] or 0.0

    def get_summary(self) -> dict[str, Any]:
        """Oversigt over forbrug."""
        return {
            "today_usd": round(self.get_daily_cost(), 4),
            "this_month_usd": round(self.get_monthly_cost(), 2),
        }


# ── LLM Client ──────────────────────────────────────────────

class LLMClient:
    """
    Claude API client med token tracking og cost management.

    Brug:
        client = LLMClient()
        response = client.analyze(
            prompt="Analysér denne portefølje...",
            data={"positions": [...]},
            purpose="morning_briefing",
        )
        print(response)
        print(client.tracker.get_summary())
    """

    # Budget-grænser
    DAILY_BUDGET_USD = 3.0
    MONTHLY_BUDGET_USD = 50.0

    def __init__(
        self,
        api_key: str = "",
        default_model: str = "claude-3-5-haiku-20241022",
        important_model: str = "claude-sonnet-4-5-20250514",
        db_path: str = "data_cache/token_usage.db",
    ) -> None:
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self._default_model = default_model
        self._important_model = important_model
        self.tracker = TokenTracker(db_path)
        self._client = None

        if self._api_key:
            try:
                import anthropic
                self._client = anthropic.Anthropic(api_key=self._api_key)
            except ImportError:
                logger.warning("[llm] anthropic package ikke installeret")
        else:
            logger.warning("[llm] ANTHROPIC_API_KEY ikke sat — LLM funktioner disabled")

    @property
    def is_available(self) -> bool:
        return self._client is not None

    def _check_budget(self) -> bool:
        """Tjek om vi er inden for budget."""
        daily = self.tracker.get_daily_cost()
        monthly = self.tracker.get_monthly_cost()

        if daily >= self.DAILY_BUDGET_USD:
            logger.warning(f"[llm] Dagligt budget overskredet: ${daily:.2f}")
            return False
        if monthly >= self.MONTHLY_BUDGET_USD:
            logger.warning(f"[llm] Månedligt budget overskredet: ${monthly:.2f}")
            return False
        return True

    def analyze(
        self,
        prompt: str,
        data: dict | str = "",
        purpose: str = "general",
        important: bool = False,
        max_tokens: int = 2000,
        temperature: float = 0.3,
        system_prompt: str = "",
    ) -> str:
        """
        Send analyse-request til Claude API.

        Args:
            prompt: Instruktionsprompt.
            data: Data der skal analyseres (konverteres til JSON).
            purpose: Formål (til tracking).
            important: Brug stærkere model (Sonnet i stedet for Haiku).
            max_tokens: Max output tokens.
            temperature: Kreativitet (0=deterministisk, 1=kreativ).
            system_prompt: Valgfrit system prompt.

        Returns:
            Claude's svar som tekst.
        """
        if not self.is_available:
            return "[LLM ikke tilgængelig — sæt ANTHROPIC_API_KEY]"

        if not self._check_budget():
            return "[Budget overskredet — LLM analyse deaktiveret for i dag]"

        model = self._important_model if important else self._default_model

        # Byg besked
        if isinstance(data, dict):
            data_str = json.dumps(data, indent=2, default=str, ensure_ascii=False)
        else:
            data_str = str(data)

        user_content = prompt
        if data_str:
            user_content += f"\n\nDATA:\n```json\n{data_str}\n```"

        # Retry med exponential backoff
        max_retries = 3
        for attempt in range(max_retries):
            try:
                kwargs: dict[str, Any] = {
                    "model": model,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "messages": [{"role": "user", "content": user_content}],
                }
                if system_prompt:
                    kwargs["system"] = system_prompt

                response = self._client.messages.create(**kwargs)

                # Registrér forbrug
                usage = TokenUsage(
                    input_tokens=response.usage.input_tokens,
                    output_tokens=response.usage.output_tokens,
                    model=model,
                )
                self.tracker.record(usage, purpose)

                logger.info(
                    f"[llm] {purpose}: {usage.total_tokens} tokens "
                    f"(${usage.estimated_cost_usd:.4f}) [{model}]"
                )

                return response.content[0].text

            except Exception as exc:
                wait_time = 2 ** attempt  # 1s, 2s, 4s
                if attempt < max_retries - 1:
                    logger.warning(
                        f"[llm] API fejl (forsøg {attempt + 1}/{max_retries}): "
                        f"{exc}. Venter {wait_time}s..."
                    )
                    time.sleep(wait_time)
                else:
                    logger.error(f"[llm] API fejl efter {max_retries} forsøg: {exc}")
                    return f"[LLM fejl: {exc}]"

        return "[LLM fejl: Ukendt]"

    def quick_analyze(self, prompt: str, data: dict | str = "") -> str:
        """Hurtig analyse med Haiku (billigst)."""
        return self.analyze(prompt, data, purpose="quick", max_tokens=1000)

    def deep_analyze(self, prompt: str, data: dict | str = "") -> str:
        """Dybdegående analyse med Sonnet."""
        return self.analyze(prompt, data, purpose="deep",
                            important=True, max_tokens=4000)
