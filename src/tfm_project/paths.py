from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    """Canonical repository paths.

    Active code must resolve files through this object rather than depending on the
    shell working directory. Frozen evidence is read-only by convention; reproduced
    experiments are written to dedicated ``reproduced`` locations.
    """

    root: Path

    @classmethod
    def discover(cls, start: str | Path | None = None) -> "ProjectPaths":
        env = os.getenv("TFM_PROJECT_ROOT")
        if env:
            root = Path(env).expanduser().resolve()
            if cls._looks_like_project(root):
                return cls(root)
            raise FileNotFoundError(f"TFM_PROJECT_ROOT is not a thesis repository: {root}")

        candidates: list[Path] = []
        if start is not None:
            p = Path(start).expanduser().resolve()
            candidates.extend([p, *p.parents])

        here = Path(__file__).resolve()
        candidates.extend([here.parent, *here.parents])
        cwd = Path.cwd().resolve()
        candidates.extend([cwd, *cwd.parents])

        seen: set[Path] = set()
        for p in candidates:
            if p in seen:
                continue
            seen.add(p)
            if cls._looks_like_project(p):
                return cls(p)

        raise FileNotFoundError(
            "Could not locate the thesis repository root. Run from the repository "
            "or set TFM_PROJECT_ROOT."
        )

    @staticmethod
    def _looks_like_project(path: Path) -> bool:
        return (
            (path / "protocols").is_dir()
            and (path / "src" / "compute").is_dir()
            and (path / "data").is_dir()
        )

    def resolve(self, value: str | Path) -> Path:
        p = Path(value).expanduser()
        return p.resolve() if p.is_absolute() else (self.root / p).resolve()

    @property
    def data(self) -> Path:
        return self.root / "data"

    @property
    def data_raw(self) -> Path:
        return self.data / "raw"

    @property
    def data_processed(self) -> Path:
        return self.data / "processed"

    @property
    def data_frozen(self) -> Path:
        return self.data / "frozen"

    @property
    def pricing_dataset(self) -> Path:
        return self.data_frozen / "pricing_dataset"

    @property
    def ml_prepared(self) -> Path:
        return self.data_frozen / "ml_prepared"

    @property
    def protocols(self) -> Path:
        return self.root / "protocols"

    @property
    def models_final(self) -> Path:
        return self.root / "models" / "final"

    @property
    def results(self) -> Path:
        return self.root / "results"

    @property
    def reproduced_results(self) -> Path:
        return self.results / "reproduced"

    @property
    def reproduced_data(self) -> Path:
        return self.data / "reproduced"

    @property
    def reproduced_models(self) -> Path:
        return self.root / "models" / "reproduced"

    @property
    def figures(self) -> Path:
        return self.root / "figures"

    @property
    def tables(self) -> Path:
        return self.root / "tables"

    @property
    def frozen_write_roots(self) -> tuple[Path, ...]:
        return (
            self.data_frozen.resolve(),
            self.models_final.resolve(),
            (self.results / "ml" / "final_evaluation").resolve(),
            (self.results / "hedging" / "local").resolve(),
            (self.results / "hedging" / "dynamic_forward").resolve(),
            self.protocols.resolve(),
        )

    def assert_not_frozen_write(self, path: str | Path) -> Path:
        target = self.resolve(path)
        for frozen in self.frozen_write_roots:
            try:
                target.relative_to(frozen)
            except ValueError:
                continue
            raise PermissionError(
                f"Refusing to write into frozen thesis evidence: {target}\n"
                f"Use data/reproduced, models/reproduced or results/reproduced instead."
            )
        return target
