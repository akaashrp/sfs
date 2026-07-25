"""Router-owned requests not yet visible in an engine snapshot."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PendingDispatch:
    engine_request_id: str
    prompt_tokens: int
    predicted_output_tokens: float
    completion_cap: int

    def as_native_tuple(self) -> tuple[str, int, float, int]:
        return (
            self.engine_request_id,
            self.prompt_tokens,
            self.predicted_output_tokens,
            self.completion_cap,
        )


class PendingDispatchLedger:
    """Insertion-ordered pending dispatches, partitioned by instance."""

    def __init__(self, instance_ids: tuple[str, ...]) -> None:
        self._by_instance: dict[str, dict[str, PendingDispatch]] = {
            instance_id: {} for instance_id in instance_ids
        }

    def reserve(self, instance_id: str, reservation: PendingDispatch) -> None:
        reservations = self._by_instance[instance_id]
        if reservation.engine_request_id in reservations:
            raise RuntimeError(
                "Duplicate pending dispatch reservation: "
                f"{reservation.engine_request_id}"
            )
        reservations[reservation.engine_request_id] = reservation

    def unobserved_for_instance(
        self,
        instance_id: str,
    ) -> tuple[PendingDispatch, ...]:
        return tuple(self._by_instance[instance_id].values())

    def mark_observed(
        self,
        instance_id: str,
        engine_request_ids: tuple[str, ...],
    ) -> None:
        reservations = self._by_instance[instance_id]
        for request_id in engine_request_ids:
            reservations.pop(request_id, None)

    def release(self, instance_id: str, engine_request_id: str) -> None:
        self._by_instance[instance_id].pop(engine_request_id, None)
