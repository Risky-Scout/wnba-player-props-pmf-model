from wnba_props_model.models.market import american_to_prob, no_vig_two_way


def test_american_to_prob():
    assert round(american_to_prob(-110), 6) == round(110/210, 6)
    assert round(american_to_prob(150), 6) == round(100/250, 6)


def test_no_vig_two_way():
    po, pu = no_vig_two_way(-110, -110)
    assert abs(po - 0.5) < 1e-12
    assert abs(pu - 0.5) < 1e-12
