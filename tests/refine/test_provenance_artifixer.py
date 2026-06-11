from spag4d.refine import provenance


def test_artifixer_tag_is_distinct():
    vals = {provenance.PROVENANCE_ORIGINAL, provenance.PROVENANCE_DENSIFIED,
            provenance.PROVENANCE_OMNIROAM, provenance.PROVENANCE_GAP_SEED,
            provenance.PROVENANCE_ARTIFIXER3D}
    assert len(vals) == 5
    assert provenance.PROVENANCE_ARTIFIXER3D == 4
