"""FEPP-CD: Federated, Explainable, Privacy-Preserving Cyberbullying Detection.

Lightweight modules (``metrics``, ``partition``, ``secagg``, ``attacks``,
``comm``) depend only on NumPy/pandas/scikit-learn. Modules that train or
explain models (``model``, ``client``, ``federated``, ``explain``) import
PyTorch and Hugging Face Transformers lazily.
"""

__version__ = "1.0.0"
