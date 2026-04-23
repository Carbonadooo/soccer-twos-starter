"""Local compatibility fixes for this old Ray training environment."""

try:
    import prometheus_client.exposition as prometheus_exposition
except Exception:
    prometheus_exposition = None

if prometheus_exposition is not None:
    _original_get_best_family = prometheus_exposition._get_best_family

    def _get_best_family(address, port):
        if address == "":
            address = "127.0.0.1"
        return _original_get_best_family(address, port)

    prometheus_exposition._get_best_family = _get_best_family
