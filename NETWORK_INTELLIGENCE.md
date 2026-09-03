# Network intelligence prototype

The prototype resolves entities by exact PaySim account ID (`nameOrig` and
`nameDest`). PaySim does not contain person or company identity attributes, so
fuzzy entity resolution is intentionally unsupported and remains future work.

`NetworkAnalyzer` keeps a bounded, process-local graph. The backend must create
one long-lived `FraudDetector` and chronologically backfill recent transactions
after every restart. Multiple workers do not share graph state; production
multi-process deployment requires a shared graph store.

Network indicators are supporting risk evidence. A link, cycle, or other graph
pattern is not proof of criminal activity and cannot independently declare a
transaction fraudulent.
