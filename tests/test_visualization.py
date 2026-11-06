from passos_wearable.evaluation import experiments as X
from passos_wearable.visualization import plots as P


def test_all_plots_write_files(rec, cfg, tmp_path):
    df = X.attach_reference(X.run_estimation(rec, cfg, 0, 200), rec, cfg)
    by = X.summarize_by(df, "activity")
    outs = [P.plot_overview(rec, tmp_path / "a.png"), P.plot_segment(rec, 100, 20, cfg, tmp_path / "b.png"),
            P.plot_hr_comparison(df, rec, out_path=tmp_path / "c.png"), P.plot_scatter_bland(df, rec, out_path=tmp_path / "d.png"),
            P.plot_error_by_activity(by, rec, out_path=tmp_path / "e.png"), P.plot_activity_comparison(df, rec, out_path=tmp_path / "f.png"),
            P.plot_sqi_vs_error(df, rec, out_path=tmp_path / "g.png"), P.plot_motion_vs_error(df, rec, out_path=tmp_path / "h.png")]
    assert all((tmp_path / f"{c}.png").stat().st_size > 5000 for c in "abcdefgh") and len(outs) == 8
