from handnet import sentiment as S


def test_param_counts():
    assert S.n_nonzero < 2000 and S.n_params > S.n_nonzero


def test_plain_polarity():
    assert S.predict("a very good movie") == 1
    assert S.predict("a complete waste of time") == 0


def test_negation_flips():
    assert S.score("this is not good") < 0.5
    assert S.score("not a bad film at all") > 0.5
    assert S.score("it is never boring") > 0.5


def test_contrast_favours_last_clause():
    assert S.score("boring at first but ultimately moving") > 0.5
    assert S.score("moving at first but ultimately boring") < 0.5


def test_not_just_keeps_polarity():
    assert S.score("the movie is n't just hilarious it 's witty and inventive") > 0.5


def test_explain_has_trace():
    ex = S.explain("not a bad film")
    assert "score" in ex and [t for t, _, _ in ex["trace"]] == ["not", "a", "bad", "film"]
