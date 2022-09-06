from dataclasses import dataclass


@dataclass
class Metadata:
    best_score: float = 0
    trained_epochs: int = 0

    @classmethod
    def from_dict(cls, d: dict) -> 'Metadata':
        return Metadata(
            best_score=d['best_score'],
            trained_epochs=d['trained_epochs']
        )

    def to_dict(self) -> dict:
        return {
            "best_score": self.best_score,
            "trained_epochs": self.trained_epochs
        }
