"""The platform service layer: database, artifact store, queue worker, API.

Everything here is a client of the benchmark core (executor, scoring,
leaderboard). Run logs remain ground truth; every row this layer writes is
derived and recomputable from them.
"""
