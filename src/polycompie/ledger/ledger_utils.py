import csv
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone
from .schema import ArticleLedger


class LedgerManager:
    """Load/save/find/update pipeline ledger CSV."""

    def __init__(self, ledger_path: str = "data/ledger.csv"):
        self.ledger_path = Path(ledger_path)

    def load(self) -> List[ArticleLedger]:
        """Load all ledger entries."""
        if not self.ledger_path.exists():
            return []

        entries = []
        with open(self.ledger_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                for key, value in row.items():
                    if value == "":
                        row[key] = None

                if row.get("created_at"):
                    row["created_at"] = datetime.fromisoformat(row["created_at"])
                if row.get("updated_at"):
                    row["updated_at"] = datetime.fromisoformat(row["updated_at"])

                if row.get("year") and row["year"] != "":
                    row["year"] = int(row["year"])

                entries.append(ArticleLedger(**row))

        return entries

    def save(self, entries: List[ArticleLedger]) -> None:
        """Persist entries to CSV."""
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)

        if not entries:
            with open(self.ledger_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self._get_fieldnames())
                writer.writeheader()
            return

        rows = []
        for entry in entries:
            row = entry.model_dump()
            row["created_at"] = row["created_at"].isoformat()
            row["updated_at"] = row["updated_at"].isoformat()
            for key, value in row.items():
                if value is None:
                    row[key] = ""
            rows.append(row)

        with open(self.ledger_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._get_fieldnames())
            writer.writeheader()
            writer.writerows(rows)

    def find_by_article_id(self, article_id: str) -> Optional[ArticleLedger]:
        """Return entry for article_id or None."""
        entries = self.load()
        for entry in entries:
            if entry.article_id == article_id:
                return entry
        return None

    def update_entry(self, article_id: str, **updates) -> bool:
        """Update entry by article_id; returns True if found."""
        entries = self.load()
        found = False

        for i, entry in enumerate(entries):
            if entry.article_id == article_id:
                for key, value in updates.items():
                    if hasattr(entry, key):
                        setattr(entry, key, value)

                entry.updated_at = datetime.now(timezone.utc)
                entries[i] = entry
                found = True
                break

        if found:
            self.save(entries)

        return found

    def add_entry(self, entry: ArticleLedger) -> None:
        """Add a new entry to the ledger."""
        entries = self.load()

        if any(e.article_id == entry.article_id for e in entries):
            raise ValueError(f"Entry with article_id {entry.article_id} already exists")

        entries.append(entry)
        self.save(entries)

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the ledger (dynamic across all collections)."""
        entries = self.load()

        # Count per-collection dynamically
        collections: Dict[str, int] = {}
        for e in entries:
            name = e.collection or "unknown"
            collections[name] = collections.get(name, 0) + 1

        stats: Dict[str, Any] = {
            "total": len(entries),
            "collections": collections,
        }

        # Stage counts
        for stage in ("download", "convert", "clean", "extract"):
            field = f"{stage}_status"
            counts: Dict[str, int] = {}
            for e in entries:
                val = getattr(e, field, None) or "unknown"
                counts[val] = counts.get(val, 0) + 1
            stats[stage] = counts

        return stats

    def _get_fieldnames(self) -> List[str]:
        return list(ArticleLedger.model_fields.keys())


def print_ledger_stats(ledger_path: str = "data/ledger.csv"):
    """Print statistics about the ledger."""
    manager = LedgerManager(ledger_path)
    stats = manager.get_stats()

    print("=" * 70)
    print("Ledger Statistics")
    print("=" * 70)
    print(f"Total articles: {stats['total']}")

    print("Collections:")
    for name, count in stats["collections"].items():
        print(f"  {name}: {count}")

    for stage in ("download", "convert", "clean", "extract"):
        print(f"{stage.title()}:")
        for status, count in stats[stage].items():
            print(f"  {status}: {count}")

    print("=" * 70)