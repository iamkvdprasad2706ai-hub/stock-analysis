from stock_analysis.ipo_research import ISSUER_RESEARCH, get_issuer_research


def test_all_covered_issuers_have_primary_source_links_and_investment_views():
    assert set(ISSUER_RESEARCH) == {
        "TONBO",
        "SMT",
        "BONFIG",
        "JPL",
        "PARASHLTH",
        "ZETWERK",
    }

    for symbol, research in ISSUER_RESEARCH.items():
        assert research.symbol == symbol
        assert research.researched_on
        assert research.sector
        assert research.business
        assert research.view
        assert research.rhp_url.startswith("https://nsearchives.nseindia.com/")
        assert research.issuer_url.startswith("https://")
        assert research.investor_relations_url.startswith("https://")


def test_lookup_is_case_insensitive_and_unknown_symbols_are_not_invented():
    assert get_issuer_research("tonbo") is ISSUER_RESEARCH["TONBO"]
    assert get_issuer_research("UNKNOWN") is None


def test_financials_use_explicitly_verified_values_and_periods():
    tonbo = get_issuer_research("TONBO")
    assert tonbo is not None
    assert tonbo.financials[2]["Revenue (₹m)"] == 3626.48
    assert tonbo.financials[2]["CFO (₹m)"] == -632.16
    assert "negative" in tonbo.risks[0]

    zetwerk = get_issuer_research("ZETWERK")
    assert zetwerk is not None
    assert zetwerk.financials[2]["Revenue (₹m)"] == 159133.23
    assert zetwerk.financials[2]["PAT (₹m)"] == -16061.71
    assert "omitted" in zetwerk.coverage_note
