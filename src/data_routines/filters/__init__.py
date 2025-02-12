from .base import FilterConfig  # noqa
from .filter_mono import FilterMonoConfig, FilterMono  # noqa

FILTERS = {
    "mono": {
        "config": FilterMonoConfig,
        "filter": FilterMono,
    }
}
