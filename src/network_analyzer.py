import json
from collections import deque
from datetime import timedelta
from pathlib import Path
from threading import RLock

import networkx as nx

from .schemas import (
    NetworkConfig,
    NetworkEvidence,
    NetworkPrediction,
    TransactionInput,
)

NETWORK_CONFIG_PATH = (
    Path(__file__).parent / "artifacts" / "network_config.json"
)


class NetworkAnalyzer:
    """Maintain a bounded transaction graph and score network behavior."""

    def __init__(
        self,
        config: NetworkConfig | None = None,
        config_path: Path | str = NETWORK_CONFIG_PATH,
    ):
        self.config_path = Path(config_path)
        self.config = config or self._load_config()
        self.graph = nx.MultiDiGraph()
        self._edges: deque[tuple] = deque()
        self._results: dict[str, NetworkPrediction] = {}
        self._latest_timestamp = None
        self._lock = RLock()

    def _load_config(self) -> NetworkConfig:
        if not self.config_path.exists():
            return NetworkConfig()
        artifact = json.loads(self.config_path.read_text(encoding="utf-8"))
        if not isinstance(artifact, dict) or "config" not in artifact:
            raise ValueError(
                "Network configuration must contain a config object."
            )
        return NetworkConfig.model_validate(artifact["config"])

    def analyze_and_record(
        self,
        transaction: TransactionInput,
    ) -> NetworkPrediction:
        with self._lock:
            cached = self._results.get(transaction.transaction_id)
            if cached is not None:
                return cached
            self._validate_order(transaction)
            self._prune(transaction)
            prediction = self._analyze(transaction)
            self._record(transaction)
            self._results[transaction.transaction_id] = prediction
            return prediction

    def backfill(self, transactions: list[TransactionInput]) -> None:
        with self._lock:
            for transaction in sorted(
                transactions,
                key=lambda item: (item.timestamp, item.transaction_id),
            ):
                if transaction.transaction_id in self._results:
                    continue
                self._validate_order(transaction)
                self._prune(transaction)
                self._record(transaction)
                self._results[transaction.transaction_id] = NetworkPrediction(
                    network_score=0.0,
                    model_version=self.config.version,
                )

    def _validate_order(self, transaction: TransactionInput) -> None:
        if (
            self._latest_timestamp is not None
            and transaction.timestamp < self._latest_timestamp
        ):
            raise ValueError(
                "Network transactions must be processed chronologically."
            )

    def _prune(self, transaction: TransactionInput) -> None:
        cutoff = transaction.timestamp - timedelta(
            hours=self.config.lookback_hours
        )
        while self._edges and (
            self._edges[0][0] < cutoff
            or len(self._edges) >= self.config.max_edges
        ):
            _, sender, receiver, key, old_transaction_id = (
                self._edges.popleft()
            )
            if self.graph.has_edge(sender, receiver, key):
                self.graph.remove_edge(sender, receiver, key)
            self._results.pop(old_transaction_id, None)
            for node in (sender, receiver):
                if node in self.graph and self.graph.degree(node) == 0:
                    self.graph.remove_node(node)

    def _record(self, transaction: TransactionInput) -> None:
        self.graph.add_edge(
            transaction.sender_id,
            transaction.receiver_id,
            key=transaction.transaction_id,
            transaction_id=transaction.transaction_id,
            amount=transaction.amount,
            type=transaction.type,
            timestamp=transaction.timestamp,
            step=transaction.step,
        )
        self._edges.append(
            (
                transaction.timestamp,
                transaction.sender_id,
                transaction.receiver_id,
                transaction.transaction_id,
                transaction.transaction_id,
            )
        )
        self._latest_timestamp = transaction.timestamp

    def _analyze(
        self,
        transaction: TransactionInput,
    ) -> NetworkPrediction:
        sender = transaction.sender_id
        receiver = transaction.receiver_id
        sender_successors = (
            set(self.graph.successors(sender))
            if sender in self.graph
            else set()
        )
        receiver_predecessors = (
            set(self.graph.predecessors(receiver))
            if receiver in self.graph
            else set()
        )
        projected_fan_out = len(sender_successors | {receiver})
        projected_fan_in = len(receiver_predecessors | {sender})
        repeated_pair = (
            self.graph.number_of_edges(sender, receiver) + 1
            if sender in self.graph and receiver in self.graph
            else 1
        )
        incoming_volume = self._incoming_volume(sender)
        pass_through_ratio = (
            transaction.amount / incoming_volume
            if incoming_volume > 0
            else 0.0
        )
        sender_volume = (
            self._outgoing_volume(sender) + transaction.amount
        )
        component_size = self._projected_component_size(sender, receiver)
        closes_cycle = self._closes_short_cycle(sender, receiver)

        indicators = {
            "sender_fan_out": float(projected_fan_out),
            "receiver_fan_in": float(projected_fan_in),
            "repeated_pair_count": float(repeated_pair),
            "pass_through_ratio": float(pass_through_ratio),
            "component_size": float(component_size),
            "sender_outgoing_volume": float(sender_volume),
            "closes_short_cycle": float(closes_cycle),
        }
        evidence: list[NetworkEvidence] = []

        self._add_evidence(
            evidence,
            projected_fan_out >= self.config.fan_out_threshold,
            "HIGH_SENDER_FAN_OUT",
            self.config.fan_out_points,
            projected_fan_out,
            f"Sender connects to {projected_fan_out} receivers.",
        )
        self._add_evidence(
            evidence,
            projected_fan_in >= self.config.fan_in_threshold,
            "HIGH_RECEIVER_FAN_IN",
            self.config.fan_in_points,
            projected_fan_in,
            f"Receiver is funded by {projected_fan_in} senders.",
        )
        self._add_evidence(
            evidence,
            repeated_pair >= self.config.repeated_pair_threshold,
            "REPEATED_PAIR_ACTIVITY",
            self.config.repeated_pair_points,
            repeated_pair,
            f"Sender and receiver have {repeated_pair} transactions.",
        )
        self._add_evidence(
            evidence,
            incoming_volume > 0
            and pass_through_ratio >= self.config.pass_through_ratio,
            "RAPID_PASS_THROUGH",
            self.config.pass_through_points,
            pass_through_ratio,
            "Sender is forwarding a large share of recently received funds.",
        )
        self._add_evidence(
            evidence,
            closes_cycle,
            "SHORT_CYCLE_CLOSURE",
            self.config.cycle_points,
            1.0,
            "Transaction closes a short directed payment cycle.",
        )
        self._add_evidence(
            evidence,
            component_size >= self.config.component_size_threshold,
            "LARGE_CONNECTED_COMPONENT",
            self.config.component_points,
            component_size,
            f"Transaction belongs to a component of {component_size} accounts.",
        )
        self._add_evidence(
            evidence,
            sender_volume >= self.config.sender_volume_threshold,
            "CONCENTRATED_SENDER_VOLUME",
            self.config.volume_points,
            sender_volume,
            f"Sender volume in the active window is {sender_volume:,.2f}.",
        )
        return NetworkPrediction(
            network_score=min(100.0, sum(item.score for item in evidence)),
            model_version=self.config.version,
            indicators=indicators,
            evidence=evidence,
        )

    @staticmethod
    def _add_evidence(
        evidence: list[NetworkEvidence],
        triggered: bool,
        code: str,
        score: float,
        value: float,
        description: str,
    ) -> None:
        if triggered:
            evidence.append(
                NetworkEvidence(
                    code=code,
                    score=score,
                    value=float(value),
                    description=description,
                )
            )

    def _incoming_volume(self, node: str) -> float:
        if node not in self.graph:
            return 0.0
        return sum(
            float(data["amount"])
            for _, _, data in self.graph.in_edges(node, data=True)
        )

    def _outgoing_volume(self, node: str) -> float:
        if node not in self.graph:
            return 0.0
        return sum(
            float(data["amount"])
            for _, _, data in self.graph.out_edges(node, data=True)
        )

    def _projected_component_size(self, sender: str, receiver: str) -> int:
        nodes = {sender, receiver}
        pending = [sender, receiver]
        while pending and len(nodes) < self.config.component_size_threshold:
            node = pending.pop()
            if node not in self.graph:
                continue
            neighbors = set(self.graph.predecessors(node))
            neighbors.update(self.graph.successors(node))
            for neighbor in neighbors - nodes:
                nodes.add(neighbor)
                pending.append(neighbor)
                if len(nodes) >= self.config.component_size_threshold:
                    break
        return len(nodes)

    def _closes_short_cycle(self, sender: str, receiver: str) -> bool:
        if sender not in self.graph or receiver not in self.graph:
            return False
        try:
            return nx.shortest_path_length(
                self.graph,
                receiver,
                sender,
            ) <= 3
        except nx.NetworkXNoPath:
            return False
