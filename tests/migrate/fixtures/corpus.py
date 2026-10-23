"""Invented paper texts for the synthetic v2 fixtures.

Every title, author, venue, DOI (the ``10.5555`` test prefix), sentence and
reference below is made up. Each :class:`Asset` is one PDF and its Marker-style
markdown: three pages, ``{N}`` + 48 dashes page markers, a ``#`` title, an
abstract, a body and a reference list. Body sentences are built from a topic and
a driver so a quote can be anchored by searching the markdown for it.
"""

from __future__ import annotations

from dataclasses import dataclass

PAGE_MARK = "{%d}" + "-" * 48


@dataclass(frozen=True)
class Asset:
    key: str
    title: str                      # as printed in the PDF and the markdown
    authors: tuple[str, ...]
    topic: str
    driver: str
    refs: tuple[str, ...] = ()
    doi_printed: str | None = None  # a "doi:" line on page 1 of the PDF and markdown
    extra: tuple[str, ...] = ()     # extra body sentences
    pdf_text_layer: bool = True     # False: a scan (no text layer), markdown still has text
    page_markers: bool = True       # False: an external import without Marker page markers
    pdf_variant: str = ""           # changes the PDF bytes (a second download of one paper)

    @property
    def rq(self) -> str:
        return f"This paper asks how {self.topic} responds to {self.driver}."

    @property
    def result(self) -> str:
        return f"We find that {self.topic} moves by eleven percent when {self.driver} doubles."

    @property
    def sample(self) -> str:
        return "The made-up sample covers forty sites observed over nine seasons."

    @property
    def abstract(self) -> str:
        return f"We examine {self.topic} and {self.driver} with fabricated records."

    def pages(self) -> list[list[str]]:
        first = [self.title, ", ".join(self.authors)]
        if self.doi_printed:
            first.append(f"doi:{self.doi_printed}")
        first += ["Abstract", self.abstract]
        body = ["1 Introduction", self.rq, self.sample, self.result, *self.extra]
        last = ["References", *self.refs]
        if self.pdf_variant:
            last.append(f"Downloaded copy {self.pdf_variant}")
        return [first, body, last]

    def pdf_pages(self) -> list[list[str] | None]:
        pages: list[list[str] | None] = list(self.pages())
        if not self.pdf_text_layer:
            return [None for _ in pages]
        return pages

    def markdown(self) -> str:
        first, body, last = self.pages()
        doi = [f"doi:{self.doi_printed}"] if self.doi_printed else []
        page0 = "\n\n".join([f"# {first[0]}", first[1], *doi, "## Abstract", self.abstract])
        page1 = "\n\n".join(["## 1 Introduction", *body[1:]])
        page2 = "\n\n".join(["## References", *(f"- {r}" for r in last[1:])])
        if not self.page_markers:
            return "\n\n".join([page0, page1, page2]) + "\n"
        parts = [f"\n\n{PAGE_MARK % i}\n\n{text}" for i, text in enumerate([page0, page1, page2])]
        return "".join(parts) + "\n"


def _refs(*items: str) -> tuple[str, ...]:
    return tuple(items)


# Reference lines shared by the overlapping paper's two parsed lists.
REF_CITED = "Okafor, T. (2008). Survey of irrigation contracts. Journal of Made-Up Agronomy 3, 1-19."
REF_AMBIG = "Lindqvist, M. and Achebe, R. (2012). Pruning schedules revisited. Working paper."
REF_SUSPECT = "Haddad, S. (1999). On the yield of stone fruit. Fictional Review 4, 1-20."
REF_TP_ONLY = "Novak, P. (2017). Orchard insurance take-up. Mimeo."
REF_OVERLAP = "Quispe, L. and Barros, N. (2019). Panel evidence on orchard yields. Imaginary Economics Letters 7."
REF_TOLLS = "Ferreira, J. (1987). Harbour tolls of a river town. Pretend Historical Review 2."
REF_PORT = "Mwangi, D. (2016). Harbour queues. Notional Trade Quarterly 5."

ASSETS: dict[str, Asset] = {a.key: a for a in [
    Asset("overlap", "Panel Evidence on Orchard Yields", ("Lucia Quispe", "Nuno Barros"),
          "orchard yield", "pruning frequency", refs=_refs(REF_CITED, REF_AMBIG, REF_SUSPECT, REF_TP_ONLY),
          doi_printed="10.5555/sg.overlap"),
    Asset("dbl", "Cooperative Dairies and Milk Price Dispersion", ("Ingrid Solberg",),
          "milk price dispersion", "cooperative membership", doi_printed="10.5555/sg.double"),
    Asset("mix", "Study of Coastal Erosion Rates", ("Mei Tanaka", "Omar Haddad"),
          "shoreline retreat", "storm surge height", refs=_refs(REF_TOLLS), doi_printed="10.5555/sg.mix-a"),
    Asset("shared", "Seasonal Labor Supply in Fishing Villages", ("Ana Ribeiro",),
          "fishing labor supply", "monsoon timing"),
    Asset("foreign", "Bank Branch Density and Small Firm Credit", ("Kwame Asante",),
          "small firm credit", "branch closures"),
    Asset("victim", "Thermal Comfort in Open Plan Offices", ("Greta Lund", "Ravi Menon"),
          "office thermal comfort", "window placement"),
    Asset("noconv", "Ferry Timetables and Island Housing Markets", ("Elin Strand",),
          "island house prices", "ferry frequency"),
    Asset("lattice", "A Theory of Lattice Rents", ("Hugo Marchetti",),
          "lattice rent", "plot adjacency", refs=_refs(REF_OVERLAP)),
    Asset("ambig", "Notes on Matching Market Design", ("Sana Qureshi",),
          "match stability", "applicant ranking depth"),
    Asset("noid", "Hedonic Prices for Lighthouse Services", ("Tomas Varga",),
          "lighthouse fees", "shipping lane traffic"),
    Asset("moji", "SpeciÖcation Tests for E¢ cient Estimators of Di¤erence Models", ("Pia Kowalski",),
          "estimator e¢ ciency", "panel length",
          extra=("The speciÖcation test is e¢ cient and the di¤erence in power is small.",),
          doi_printed="10.5555/sg.moji"),
    Asset("mojititle", "Identification under Confinement Constraints", ("Yusuf Demir",),
          "identification strength", "confinement bounds"),
    Asset("wrongpdf", "Tide Tables for Estuary Navigation", ("Harriet Cole",),
          "estuary depth", "lunar phase"),
    Asset("malcached", "Sediment Transport in Braided River Deltas", ("Joao Pinto",),
          "sediment load", "channel braiding"),
    Asset("maluncached", "Volunteer Fire Brigades and Rural Insurance", ("Agnes Horvath",),
          "rural insurance premiums", "brigade coverage"),
    Asset("port_a", "Port Congestion and Export Timing", ("Daniel Mwangi",),
          "export shipment timing", "berth congestion", refs=_refs(REF_PORT), doi_printed="10.5555/sg.port"),
    Asset("port_b", "Port Congestion and Export Timing", ("Daniel Mwangi",),
          "export shipment timing", "berth congestion", refs=_refs(REF_PORT), pdf_variant="second scan"),
    Asset("auction", "Auction Formats for Fishing Quotas", ("Lena Petrova",),
          "quota auction revenue", "bidder entry"),
    Asset("unconv", "Mobile Libraries and Rural Literacy", ("Chidi Okeke",),
          "rural literacy", "mobile library visits"),
    Asset("fake", "Bus Rapid Transit and Commute Times", ("Marta Silva",),
          "commute duration", "dedicated bus lanes"),
    Asset("unheld", "Wind Shear Over Rooftop Gardens", ("Oskar Berg",),
          "rooftop wind shear", "parapet height"),
    Asset("external", "Municipal Bond Spreads After Floods", ("Fatima Zahra",),
          "municipal bond spreads", "flood damage", page_markers=False),
    Asset("scan", "Rent Control and Tenant Turnover", ("Pedro Alves",),
          "tenant turnover", "rent ceilings", pdf_text_layer=False),
]}
# A second download of the double-imported paper: same text, different PDF bytes.
ASSETS["dbl_copy"] = Asset(**{**ASSETS["dbl"].__dict__, "key": "dbl_copy", "pdf_variant": "second download"})


ANSWERS_TEXT = {
    "orchard": ("How does pruning frequency relate to orchard yield?",
                "In the fixture corpus, orchard yield moves with pruning frequency [E1]."),
    "credit": ("What happens to small firm credit after branch closures?",
               "The fixture paper reports a change in small firm credit [E1]."),
    "ports": ("Is berth congestion linked to export timing?",
              "One fixture paper links berth congestion to export shipment timing [E1]."),
    "runscoped": ("Which papers discuss lighthouse fees?",
                  "A run-scoped fixture answer that the migration must not carry."),
}
