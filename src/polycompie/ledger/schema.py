from typing import Optional, Literal
from pydantic import BaseModel, Field
from datetime import datetime, timezone
from pathlib import Path


class ArticleLedger(BaseModel):
    """
    Single article entry in the pipeline ledger (CSV).
    Tracks progress through all pipeline stages.

    Path convention:
        - convert_path / clean_path are stored RELATIVE to the article folder,
          e.g. "outputs/docling/article_id.md"
        - get_*_path() methods resolve to full path:
          {base_dir}/{collection}/{article_id}/{relative_path}
        - base_dir default is "data" (relative to working dir pipeline/polm_ie/)
    """

    article_id: str  # Folder name (PDF stem)
    collection: Optional[str] = None

    # Article metadata
    doi: Optional[str] = None
    title: Optional[str] = None
    year: Optional[int] = None
    source_title: Optional[str] = None
    scopus_query: Optional[str] = None

    # Download stage
    oa_link: Optional[str] = None
    pdf_url: Optional[str] = None
    download_status: Literal["manual", "pending", "success", "failed"] = "manual"
    download_note: Optional[str] = None

    # Conversion stage
    convert_status: Literal["pending", "ok", "failed"] = "pending"
    convert_path: Optional[str] = None  # Relative to article dir: outputs/docling/{id}.md

    # Cleaning stage
    clean_status: Literal["pending", "ok", "failed"] = "pending"
    clean_path: Optional[str] = None  # Relative to article dir: outputs/cleaned/paper.md

    # Extraction stage
    extract_status: Literal["pending", "ok", "failed"] = "pending"

    # Timestamps
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def _article_dir(self, base_dir: str = "data") -> Path:
        """Return the article's root folder: {base_dir}/{collection}/{article_id}."""
        return Path(base_dir) / self.collection / self.article_id

    def get_pdf_path(self, base_dir: str = "data") -> str:
        """Get PDF path: {base_dir}/{collection}/{article_id}/{article_id}.pdf"""
        return str(self._article_dir(base_dir) / f"{self.article_id}.pdf")

    def get_convert_path(self, base_dir: str = "data") -> str:
        """Get convert output path.

        If convert_path is stored, resolve it under the article dir.
        Otherwise fall back to the default location.
        """
        article = self._article_dir(base_dir)
        if self.convert_path:
            return str(article / self.convert_path)
        return str(article / "outputs" / "docling" / f"{self.article_id}.md")

    def get_clean_path(self, base_dir: str = "data") -> str:
        """Get cleaned output path.

        If clean_path is stored, resolve it under the article dir.
        Otherwise fall back to the default location.
        """
        article = self._article_dir(base_dir)
        if self.clean_path:
            return str(article / self.clean_path)
        return str(article / "outputs" / "cleaned" / f"{self.article_id}.md")