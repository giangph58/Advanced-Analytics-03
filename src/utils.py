from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt

# Arxiv high-level category mapping
arxiv_main_category = {
    "astro-ph": [
        "astro-ph.CO", "astro-ph.EP", "astro-ph.GA", "astro-ph.HE",
        "astro-ph.IM", "astro-ph.SR"
    ],
    "cond-mat": [
        "cond-mat.dis-nn", "cond-mat.mes-hall", "cond-mat.mtrl-sci",
        "cond-mat.other", "cond-mat.quant-gas", "cond-mat.soft",
        "cond-mat.stat-mech", "cond-mat.str-el", "cond-mat.supr-con"
    ],
    "gr-qc": [
        "gr-qc"
    ],
    # "hep-ex": ["hep-ex"],
    # "hep-lat": ["hep-lat"],
    # "hep-ph": ["hep-ph"],
    # "hep-th": ["hep-th"],
    "hep": [
        "hep-ex", "hep-lat", "hep-ph", "hep-th"
    ],
    "math-ph": [
        "math-ph"
    ],
    "nlin": [
        "nlin.AO", "nlin.CG", "nlin.CD", "nlin.SI", "nlin.PS"
    ],
    # "nucl-ex": ["nucl-ex"],
    # "nucl-th": ["nucl-th"],
    "nucl": [
        "nucl-ex", "nucl-th"
    ],
    "quant-ph": [
        "quant-ph"
    ],
    "physics": [
        "physics.acc-ph", "physics.ao-ph", "physics.app-ph", "physics.atm-clus",
        "physics.atom-ph", "physics.bio-ph", "physics.chem-ph", "physics.class-ph",
        "physics.comp-ph", "physics.data-an", "physics.ed-ph", "physics.flu-dyn",
        "physics.gen-ph", "physics.geo-ph", "physics.hist-ph", "physics.ins-det",
        "physics.med-ph", "physics.optics", "physics.plasm-ph", "physics.pop-ph",
        "physics.soc-ph", "physics.space-ph"
    ],
    "math": [
        "math.AC", "math.AG", "math.AP", "math.AT", "math.CA", "math.CO",
        "math.CT", "math.CV", "math.DG", "math.DS", "math.FA", "math.GM",
        "math.GN", "math.GR", "math.GT", "math.HO", "math.IT", "math.KT",
        "math.LO", "math.MG", "math.MP", "math.NA", "math.NT", "math.OA",
        "math.OC", "math.PR", "math.QA", "math.RA", "math.RT", "math.SG",
        "math.SP", "math.ST"
    ],
    "cs": [
        "cs.AI", "cs.AR", "cs.CC", "cs.CE", "cs.CG", "cs.CL", "cs.CR", "cs.CV",
        "cs.CY", "cs.DB", "cs.DC", "cs.DL", "cs.DM", "cs.DS", "cs.ET", "cs.FL",
        "cs.GL", "cs.GR", "cs.GT", "cs.HC", "cs.IR", "cs.IT", "cs.LG", "cs.LO",
        "cs.MA", "cs.MM", "cs.MS", "cs.NA", "cs.NE", "cs.NI", "cs.OH", "cs.OS",
        "cs.PF", "cs.PL", "cs.RO", "cs.SC", "cs.SD", "cs.SE", "cs.SI", "cs.SY"
    ],
    "q-bio": [
        "q-bio.BM", "q-bio.CB", "q-bio.GN", "q-bio.MN", "q-bio.NC", "q-bio.OT",
        "q-bio.PE", "q-bio.QM", "q-bio.SC", "q-bio.TO"
    ],
    "q-fin": [
        "q-fin.CP", "q-fin.EC", "q-fin.GN", "q-fin.MF", "q-fin.PM", "q-fin.PR",
        "q-fin.RM", "q-fin.ST", "q-fin.TR"
    ],
    "stat": [
        "stat.AP", "stat.CO", "stat.ME", "stat.ML", "stat.OT", "stat.TH"
    ],
    "eess": [
        "eess.AS", "eess.IV", "eess.SP", "eess.SY"
    ],
    "econ": [
        "econ.EM", "econ.GN", "econ.TH"
    ],
}

category_to_parent = {child: parent for parent, children in arxiv_main_category.items() for child in children}
# Define a Python function for mapping (no need for UDF in pandas)
def map_category(category):
    return category_to_parent.get(category, "other")


def plot_confusion_matrix(y_preds, y_true, labels):
    cm = confusion_matrix(y_true, y_preds, normalize="true")
    fig, ax = plt.subplots(figsize=(10, 10))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", values_format=".2f", ax=ax, colorbar=False)
    plt.title("Normalized confusion matrix")
    plt.xticks(rotation=30, ha="right")
    plt.show()

