from ..groundtruth import sgd

def test_intervention_score():
    pass


def test_sgd():
    mapping = sgd.mapping

    # test a few mappings
    assert mapping['VPT11'] == 'YMR231W'
    assert mapping['YER021W'] == 'YER021W'
    assert mapping['CYK3'] == 'YDL117W'