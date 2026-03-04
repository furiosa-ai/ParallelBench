from parallelbench.lm_eval_models.metadata_store import MetadataStore

# Import wrappers to trigger @register_model decorators for lm-eval
from parallelbench.lm_eval_models import (  # noqa: F401
    llada_wrapper,
    dream_wrapper,
    sedd_wrapper,
    trado_wrapper,
    ar_wrapper,
    api_wrapper,
)

__all__ = ["MetadataStore"]
