"""Issuer facts verified from NSE-hosted IPO Red Herring Prospectuses."""

from __future__ import annotations

from dataclasses import dataclass

NSE_RHP_BASE_URL = "https://nsearchives.nseindia.com/content/ipo"


@dataclass(frozen=True)
class IPOIssuerResearch:
    """A dated, source-linked issuer research snapshot."""

    symbol: str
    researched_on: str
    sector: str
    business: str
    financials: tuple[dict[str, str | int | float], ...]
    peer_context: str
    expansion_and_investment: str
    order_book: str
    risks: tuple[str, ...]
    view: str
    issuer_url: str
    investor_relations_url: str
    rhp_url: str
    coverage_note: str


ISSUER_RESEARCH: dict[str, IPOIssuerResearch] = {
    "TONBO": IPOIssuerResearch(
        symbol="TONBO",
        researched_on="RHP dated 10 Oct 2026",
        sector="Defence and electro-optical imaging technology",
        business=(
            "Designs imaging and sensor products for defence, law-enforcement, "
            "homeland-security customers and defence OEMs; the RHP says products "
            "had been supplied to 24 countries as of 30 Jun 2026."
        ),
        financials=(
            {"Period": "FY2024", "Revenue (₹m)": 4281.89, "PAT (₹m)": 685.43, "CFO (₹m)": 783.33},
            {"Period": "FY2025", "Revenue (₹m)": 4690.80, "PAT (₹m)": 727.57, "CFO (₹m)": -1361.19},
            {"Period": "FY2026", "Revenue (₹m)": 3626.48, "PAT (₹m)": 508.80, "CFO (₹m)": -632.16},
            {"Period": "Q1 FY2027", "Revenue (₹m)": 360.08, "PAT (₹m)": -177.37, "CFO (₹m)": "Not provided"},
        ),
        peer_context=(
            "The RHP's industry and valuation sections should be reviewed alongside "
            "listed defence/electronics businesses; a directly comparable peer "
            "multiple was not verified in this snapshot."
        ),
        expansion_and_investment=(
            "The RHP says a ₹514.41m customer order received in FY2026 was deferred "
            "to FY2027 amid West Asia conflict-related execution delays."
        ),
        order_book=(
            "₹3,142.40m as of 30 Jun 2026, as disclosed in the RHP. A ₹59.03m "
            "international order book at 31 Mar 2025 was executed in FY2026."
        ),
        risks=(
            "FY2026 revenue and PAT declined year over year; operating cash flow "
            "was negative in FY2025 and FY2026.",
            "The RHP highlights order execution delays, customer concentration "
            "and geopolitical exposure.",
            "The IPO is an offer for sale; proceeds do not fund the company.",
        ),
        view=(
            "Cautious / valuation-dependent. The defence niche and disclosed "
            "backlog are positives, but recent revenue contraction, negative "
            "operating cash flow and customer/order timing concentration require "
            "a conservative valuation. Do not treat backlog as guaranteed revenue."
        ),
        issuer_url="https://www.tonboimaging.com/",
        investor_relations_url="https://tonboimaging.com/main/industry-report/",
        rhp_url=f"{NSE_RHP_BASE_URL}/RHP_TONBO.zip",
        coverage_note=(
            "Financial values are RHP restated figures in ₹ million; operating "
            "cash flow is reported, not a forecast."
        ),
    ),
    "SMT": IPOIssuerResearch(
        symbol="SMT",
        researched_on="RHP dated 2026",
        sector="Medical devices — vascular intervention and structural heart",
        business=(
            "Develops and sells vascular-intervention and structural-heart "
            "medical devices. The RHP reports sales across 65 countries and "
            "material investment in research and development."
        ),
        financials=(
            {"Period": "FY2024", "Revenue (₹m)": 9016.04, "PBT (₹m)": "Not verified"},
            {"Period": "FY2025", "Revenue (₹m)": 10248.79, "PBT (₹m)": 408.11},
            {"Period": "FY2026", "Revenue (₹m)": 12105.04, "PBT (₹m)": 802.92},
            {"Period": "Q1 FY2027", "Revenue (₹m)": 3516.69, "PBT (₹m)": 242.80},
        ),
        peer_context=(
            "The RHP compares the company with Poly Medicure and Laxmi Dental. "
            "Its disclosed Q1 FY2027 KPI table reports SMT EBITDA margin of "
            "13.81% and PAT margin of 4.29%, versus 24.10% and 15.30% for "
            "Poly Medicure; product mixes differ, so this is directional context."
        ),
        expansion_and_investment=(
            "FY2026 R&D spend was ₹1,041.64m (8.61% of revenue), and the RHP "
            "reports 156 R&D staff as of 30 Jun 2026. No quantified expansion "
            "capex commitment is included in this snapshot."
        ),
        order_book="No quantified order book was verified in the reviewed RHP disclosures.",
        risks=(
            "Profit before tax remains much smaller than revenue; reported FY2025 "
            "PBT was ₹408.11m against ₹10,248.79m revenue.",
            "The RHP notes currency, distributor, tender, product-approval and "
            "reimbursement/pricing risks.",
            "The offer is an offer for sale; proceeds do not fund the company.",
        ),
        view=(
            "Cautious / valuation-dependent. Revenue growth, export reach and "
            "R&D spending are constructive, but disclosed margins trail the "
            "named large peer and the entire offer is secondary. Review offer "
            "valuation and cash conversion before applying."
        ),
        issuer_url="https://www.smtpl.com/",
        investor_relations_url="https://www.smtpl.com/investor",
        rhp_url=f"{NSE_RHP_BASE_URL}/RHP_SMT.zip",
        coverage_note=(
            "Revenue and PBT are RHP restated figures in ₹ million. PAT, balance "
            "sheet and cash-flow comparisons are omitted here because they were "
            "not verified in the extracted summary."
        ),
    ),
    "BONFIG": IPOIssuerResearch(
        symbol="BONFIG",
        researched_on="RHP dated 2026",
        sector="Industrial machinery and power-transmission systems",
        business=(
            "Provides technology-led power transmission and industrial drive "
            "solutions across industry and automation, off-highway mobility "
            "and wind end markets."
        ),
        financials=(
            {"Period": "FY2023", "Revenue (₹m)": 16873.30, "EBITDA margin": "14.45%", "PAT (₹m)": 1466.65},
            {"Period": "FY2024", "Revenue (₹m)": 18541.83, "EBITDA margin": "15.91%", "PAT (₹m)": 1915.98},
            {"Period": "FY2025", "Revenue (₹m)": 20135.26, "EBITDA margin": "18.41%", "PAT (₹m)": 2378.74},
            {"Period": "H1 2026", "Revenue (₹m)": 11318.69, "EBITDA margin": "19.78%", "PAT (₹m)": 1485.81},
        ),
        peer_context=(
            "The RHP names Elecon Engineering, Schaeffler India, Shanthi Gears "
            "and Timken India in peer KPI comparisons. Its own EBITDA margin "
            "improved from 14.45% in FY2023 to 18.41% in FY2025; compare peers "
            "on a consistent reporting period and business mix."
        ),
        expansion_and_investment=(
            "The RHP describes investments in manufacturing capacity and a "
            "strategy focused on higher-value industries; no quantified order "
            "backlog figure was verified."
        ),
        order_book="No quantified order book was verified in the reviewed RHP disclosures.",
        risks=(
            "The offer is entirely an offer for sale; proceeds go to the selling "
            "shareholder, not the issuer.",
            "Exposure to industrial, wind and off-highway demand cycles and "
            "customer/end-market concentration can affect growth.",
        ),
        view=(
            "Constructive operating trend, but valuation-dependent. Revenue, PAT "
            "and EBITDA margin increased across FY2023–FY2025; the secondary-only "
            "structure and cyclicality merit a price-discipline approach."
        ),
        issuer_url="https://www.bonfiglioli.in/",
        investor_relations_url="https://www.bonfiglioli.in/Investor_Relations",
        rhp_url=f"{NSE_RHP_BASE_URL}/RHP_BONFIG.zip",
        coverage_note=(
            "Restated continuing-operations values in ₹ million, calendar-year "
            "periods; H1 2026 is six months and is not annualised."
        ),
    ),
    "JPL": IPOIssuerResearch(
        symbol="JPL",
        researched_on="RHP dated 2026",
        sector="Telecommunications and digital technology platforms",
        business=(
            "Operates digital connectivity and platform businesses, with "
            "Reliance Jio Infocomm Limited (RJIL) the principal operating "
            "subsidiary and earnings contributor."
        ),
        financials=(
            {"Period": "FY2024", "Revenue (₹m)": 1095581, "PAT (₹m)": 214340},
            {"Period": "FY2025", "Revenue (₹m)": 1282184, "PAT (₹m)": 261203},
            {"Period": "FY2026", "Revenue (₹m)": 1468853, "PAT (₹m)": 300527},
            {"Period": "Q1 FY2027", "Revenue (₹m)": 391729, "PAT (₹m)": 77641},
        ),
        peer_context=(
            "The RHP's peer KPI section should be used to assess telecom "
            "comparability. JPL's issuer-level platform metrics are not fully "
            "represented by connectivity KPIs; RJIL generated 87.74% of FY2026 "
            "revenue and 93.76% of PAT."
        ),
        expansion_and_investment=(
            "Fresh issue proceeds include up to ₹275,000m for partial or full "
            "prepayment/repayment of RJIL borrowings, as described in the RHP."
        ),
        order_book="Not applicable to the disclosed core connectivity/platform model; no quantified order book was reported.",
        risks=(
            "Results are highly dependent on RJIL and the wider Reliance Group, "
            "including related-party and shared-brand considerations.",
            "Network disruption, technology change, regulation, capital needs "
            "and competitive intensity are material risks.",
        ),
        view=(
            "Fundamentally profitable and growing in the reported periods, but "
            "the investment case is sensitive to issue valuation, capital "
            "structure and RJIL concentration. Review the final price and "
            "peer multiples before deciding."
        ),
        issuer_url="https://www.jio.com/",
        investor_relations_url="https://www.jio.com/about/investor-relations/",
        rhp_url=f"{NSE_RHP_BASE_URL}/RHP_JPL.zip",
        coverage_note=(
            "RHP restated consolidated values in ₹ million. The Q1 FY2027 period "
            "is three months and is not annualised."
        ),
    ),
    "PARASHLTH": IPOIssuerResearch(
        symbol="PARASHLTH",
        researched_on="RHP dated 2026",
        sector="Healthcare — multi-specialty hospitals",
        business=(
            "Operates a North India-focused hospital network. The RHP identifies "
            "eight hospitals and lists Apollo Hospitals Enterprise, Fortis "
            "Healthcare, Aster DM Quality Care and Global Health (Medanta) as "
            "listed peers."
        ),
        financials=(
            {
                "Period": "FY2024",
                "Revenue (₹m)": 11290.39,
                "EBITDA (₹m)": 1544.11,
                "EBITDA margin": "13.42%",
                "PAT (₹m)": -153.31,
                "Debt/equity": 2.73,
            },
            {
                "Period": "FY2025",
                "Revenue (₹m)": 12940.63,
                "EBITDA (₹m)": 1564.61,
                "EBITDA margin": "11.91%",
                "PAT (₹m)": -579.83,
                "Debt/equity": 4.39,
            },
            {
                "Period": "FY2026",
                "Revenue (₹m)": 16059.52,
                "EBITDA (₹m)": 3355.77,
                "EBITDA margin": "20.60%",
                "PAT (₹m)": 438.34,
                "Debt/equity": 3.53,
            },
            {
                "Period": "Q1 FY2026",
                "Revenue (₹m)": 3683.87,
                "EBITDA (₹m)": 663.09,
                "PAT (₹m)": -10.24,
            },
            {
                "Period": "Q1 FY2027",
                "Revenue (₹m)": 4702.78,
                "EBITDA (₹m)": 1134.77,
                "PAT (₹m)": 496.82,
            },
        ),
        peer_context=(
            "The RHP's listed peer set is Apollo Hospitals Enterprise, Fortis "
            "Healthcare, Aster DM Quality Care and Global Health (Medanta). "
            "The issuer reported losses in FY2024, FY2025 and Q1 FY2026, unlike "
            "mature listed hospital groups; compare margins and occupancy "
            "carefully rather than applying peer multiples unadjusted."
        ),
        expansion_and_investment=(
            "The RHP describes a strategy to scale existing hospitals and expand "
            "the network in North India, which entails additional capex. The "
            "offer comprises ₹5,000m fresh issue and ₹9,500m OFS."
        ),
        order_book="No quantified order book was verified; hospital services are not generally assessed on an order-book basis.",
        risks=(
            "The issuer returned to profit in FY2026, but recorded losses in "
            "FY2024, FY2025 and Q1 FY2026; some subsidiaries also incurred losses.",
            "Expansion capex, hiring and retaining specialists, and profitability "
            "at newer hospitals create execution and funding risks; FY2026 "
            "debt/equity was 3.53×.",
        ),
        view=(
            "High execution risk / wait for valuation review. Hospital demand "
            "and expansion offer potential, but historical losses and planned "
            "capex make a subscribe view dependent on a defensible price and "
            "evidence of improving hospital-level profitability."
        ),
        issuer_url="https://www.parashospitals.com/",
        investor_relations_url="https://www.parashospitals.com/investors/ipo-related-documents",
        rhp_url=f"{NSE_RHP_BASE_URL}/RHP_PARASHLTH.zip",
        coverage_note=(
            "Losses in ₹ million are as stated in RHP risk disclosures. A complete "
            "revenue, EBITDA, debt and cash-flow table was not verified here."
        ),
    ),
    "ZETWERK": IPOIssuerResearch(
        symbol="ZETWERK",
        researched_on="RHP dated 2026",
        sector="Manufacturing and supply-chain services",
        business=(
            "The company describes a technology-enabled manufacturing and "
            "supply-chain platform. Review the RHP business section for its "
            "specific product verticals, geographies and customer mix."
        ),
        financials=(
            {"Period": "FY2024", "Revenue (₹m)": 123643.72},
            {"Period": "FY2025", "Revenue (₹m)": 113318.56},
            {
                "Period": "FY2026",
                "Revenue (₹m)": 159133.23,
                "PAT (₹m)": -16061.71,
            },
        ),
        peer_context=(
            "No peer valuation or comparable KPI figures are included in this "
            "snapshot; assess the RHP's selected peer set and use the same "
            "reporting periods before drawing conclusions."
        ),
        expansion_and_investment=(
            "The RHP discusses capacity and operating scale as part of the "
            "business strategy; quantified capex commitments and use-of-proceeds "
            "figures require review of the final offer document."
        ),
        order_book="No quantified backlog value has been verified in this snapshot.",
        risks=(
            "FY2026 restated continuing-operations loss was ₹16,061.71m despite "
            "higher revenue; working-capital conversion, leverage and customer "
            "concentration need careful review.",
        ),
        view=(
            "High-risk / valuation-dependent. FY2026 revenue rebounded, but the "
            "large reported continuing-operations loss makes profitability, "
            "cash conversion and customer concentration central diligence items. "
            "Do not apply without reviewing the final valuation and use of proceeds."
        ),
        issuer_url="https://www.zetwerk.com/",
        investor_relations_url="https://www.zetwerk.com/",
        rhp_url=f"{NSE_RHP_BASE_URL}/RHP_ZETWERK.zip",
        coverage_note=(
            "Revenue and FY2026 continuing-operations loss are RHP restated "
            "figures in ₹ million; other margin, cash-flow and balance-sheet "
            "metrics are omitted because they were not verified in this review."
        ),
    ),
}


def get_issuer_research(symbol: str) -> IPOIssuerResearch | None:
    """Return the verified research snapshot for an NSE issue symbol."""
    return ISSUER_RESEARCH.get(symbol.strip().upper())
