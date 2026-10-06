"""End-to-end CPU smoke test (requires torch + transformers; skipped otherwise)."""
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")


def test_federated_secagg_equals_fedavg(tmp_path):
    from fepp_cd.federated import _float_state, fedavg_aggregate, secagg_aggregate
    from fepp_cd.model import build_model
    from fepp_cd.secagg import setup_pairwise_seeds

    torch.manual_seed(0)
    g, _ = build_model("tiny-random")
    gs = _float_state(g)
    clients = [{k: v + 0.01 * torch.randn_like(v) for k, v in gs.items()} for _ in range(3)]
    w = [0.5, 0.3, 0.2]
    a = fedavg_aggregate(gs, clients, w)
    b = secagg_aggregate(gs, clients, w, setup_pairwise_seeds(3, seed=0), 1, 24)
    for k in a:
        assert torch.max(torch.abs(a[k] - b[k])) < 1e-6


def test_smoke_run_writes_outputs(tmp_path):
    from fepp_cd.federated import load_config, run

    cfg = load_config("configs/smoke.yaml", {"output_dir": str(tmp_path), "rounds": 2, "save_model": True})
    m = run(cfg)
    for f in ["predictions.csv", "round_log.csv", "client_distribution.csv", "metrics.json", "model.pt"]:
        assert (tmp_path / f).exists()
    pred = pd.read_csv(tmp_path / "predictions.csv")
    assert set(pred.columns) == {"y_true", "y_pred", "y_prob"} and len(pred) == m["n"]


def test_integrated_gradients_runs(tmp_path):
    pytest.importorskip("captum")
    from fepp_cd.explain import IGExplainer, rationale_alignment
    from fepp_cd.model import build_model

    model, tok = build_model("tiny-random")
    ex = IGExplainer(model, tok, max_len=16, n_steps=8)
    s = ex.word_scores(["you", "are", "stupid"])
    assert s.shape == (3,) and s.max() <= 3.0
    df = pd.DataFrame({"tokens": [["you", "are", "stupid"], ["i", "hate", "you"]],
                       "rationale_pos": [[2], [1]], "post_id": ["a", "b"]})
    ra = rationale_alignment(ex, df, n=2)
    assert len(ra) == 2 and ra.f1.between(0, 1).all()
