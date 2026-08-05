"""Build an IEEE-style implementation and experimental-analysis PDF."""

from __future__ import annotations

import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas
from reportlab.platypus import (
    BaseDocTemplate, Frame, KeepTogether, NextPageTemplate, PageBreak,
    PageTemplate, Paragraph, Spacer, Table, TableStyle,
)
from reportlab.graphics.shapes import Drawing, Line, Rect, String


ROOT = Path(__file__).resolve().parents[2]
METRICS = json.loads((ROOT / "models" / "training_metrics.json").read_text(encoding="utf-8"))
DRIFT = json.loads((ROOT / "logs" / "drift_experiment_result.json").read_text(encoding="utf-8"))
OUTPUT = ROOT / "outputs" / "AI_Silent_Failure_Detection_IEEE_Report.pdf"

PAGE_W, PAGE_H = A4
MARGIN = 0.52 * inch
COLUMN_GAP = 0.18 * inch
COLUMN_W = (PAGE_W - 2 * MARGIN - COLUMN_GAP) / 2


def esc(text: object) -> str:
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def add_page_number(canv: canvas.Canvas, doc: BaseDocTemplate) -> None:
    if doc.page > 1:
        canv.saveState()
        canv.setStrokeColor(colors.HexColor("#B0B7C3"))
        canv.line(MARGIN, PAGE_H - 0.35 * inch, PAGE_W - MARGIN, PAGE_H - 0.35 * inch)
        canv.setFont("Helvetica", 7)
        canv.setFillColor(colors.HexColor("#465161"))
        canv.drawString(MARGIN, PAGE_H - 0.27 * inch, "AI Silent Failure Detection in ML Pipelines")
        canv.drawRightString(PAGE_W - MARGIN, 0.30 * inch, str(doc.page))
        canv.restoreState()


def metric_chart() -> Drawing:
    """Bar chart comparing default and tuned-threshold fraud metrics."""
    default = METRICS["test_metrics_at_default_threshold"]
    tuned = METRICS["test_metrics_at_selected_threshold"]
    drawing = Drawing(COLUMN_W, 165)
    drawing.add(String(0, 153, "Threshold tuning: test-set metrics", fontName="Helvetica-Bold", fontSize=8))
    labels = ["Precision", "Recall", "F1"]
    values = [[default["precision"], default["recall"], default["f1"]], [tuned["precision"], tuned["recall"], tuned["f1"]]]
    chart_left, chart_bottom, chart_height = 34, 34, 94
    drawing.add(Line(chart_left, chart_bottom, chart_left, chart_bottom + chart_height, strokeColor=colors.grey))
    drawing.add(Line(chart_left, chart_bottom, COLUMN_W - 8, chart_bottom, strokeColor=colors.grey))
    for tick in [0, 0.25, 0.50, 0.75, 1.0]:
        y = chart_bottom + tick * chart_height
        drawing.add(Line(chart_left, y, COLUMN_W - 8, y, strokeColor=colors.HexColor("#E6EAF0")))
        drawing.add(String(1, y - 2, f"{tick:.2f}", fontSize=6, fillColor=colors.grey))
    for i, label in enumerate(labels):
        x = 66 + i * 62
        for j, value in enumerate([values[0][i], values[1][i]]):
            height = value * chart_height
            drawing.add(Rect(x + j * 15, chart_bottom, 11, height, fillColor=[colors.HexColor("#7E8FA6"), colors.HexColor("#1565C0")][j], strokeColor=None))
        drawing.add(String(x - 5, 20, label, fontSize=6))
    drawing.add(Rect(35, 138, 7, 7, fillColor=colors.HexColor("#7E8FA6"), strokeColor=None))
    drawing.add(String(45, 138, "Default 0.50", fontSize=6))
    drawing.add(Rect(104, 138, 7, 7, fillColor=colors.HexColor("#1565C0"), strokeColor=None))
    drawing.add(String(114, 138, f"Tuned {tuned['threshold']:.2f}", fontSize=6))
    return drawing


def psi_chart() -> Drawing:
    """Horizontal PSI bar chart for drifted experiment features."""
    items = [(name, info["psi"]) for name, info in DRIFT["feature_results"].items() if info["drift_detected"]]
    items.sort(key=lambda item: item[1], reverse=True)
    drawing = Drawing(COLUMN_W, 155)
    drawing.add(String(0, 142, "Injected production drift: PSI by affected feature", fontName="Helvetica-Bold", fontSize=8))
    max_value = max(value for _, value in items)
    for idx, (name, value) in enumerate(items):
        y = 113 - idx * 26
        drawing.add(String(0, y + 4, name.replace("_", " "), fontSize=6.5))
        width = (value / max_value) * 125
        drawing.add(Rect(101, y, width, 11, fillColor=colors.HexColor("#D1495B"), strokeColor=None))
        drawing.add(String(230, y + 3, f"{value:.2f}", fontSize=6.5))
    drawing.add(String(0, 12, "PSI >= 0.20 is configured as a drift alert threshold.", fontSize=6.5, fillColor=colors.grey))
    return drawing


styles = getSampleStyleSheet()
title = ParagraphStyle("Title", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=16, leading=19, alignment=TA_CENTER, spaceAfter=6)
authors = ParagraphStyle("Authors", parent=styles["Normal"], fontName="Helvetica", fontSize=8.3, leading=10, alignment=TA_CENTER, spaceAfter=10)
abstract = ParagraphStyle("Abstract", parent=styles["Normal"], fontName="Helvetica", fontSize=8.5, leading=11, alignment=TA_JUSTIFY, spaceAfter=6)
body = ParagraphStyle("Body", parent=styles["Normal"], fontName="Helvetica", fontSize=8, leading=10.2, alignment=TA_JUSTIFY, spaceAfter=5)
section = ParagraphStyle("Section", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=9.2, leading=12, spaceBefore=5, spaceAfter=4)
subsection = ParagraphStyle("Subsection", parent=styles["Heading3"], fontName="Helvetica-Bold", fontSize=8.4, leading=10, spaceBefore=4, spaceAfter=3)
caption = ParagraphStyle("Caption", parent=styles["Normal"], fontName="Helvetica-Oblique", fontSize=6.7, leading=8, alignment=TA_CENTER, spaceAfter=5)
ref = ParagraphStyle("Ref", parent=styles["Normal"], fontName="Helvetica", fontSize=7.1, leading=8.8, leftIndent=10, firstLineIndent=-10, spaceAfter=2)


def p(text: str, style: ParagraphStyle = body) -> Paragraph:
    return Paragraph(text, style)


def table(data: list[list[object]], widths: list[float]) -> Table:
    rendered = [[p(esc(value), body) if not isinstance(value, Paragraph) else value for value in row] for row in data]
    rendered[0] = [p(f"<b>{esc(value)}</b>", body) if not isinstance(value, Paragraph) else value for value in data[0]]
    tbl = Table(rendered, colWidths=widths, repeatRows=1, hAlign="LEFT")
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#D9E5F3")),
        ("LINEBELOW", (0, 0), (-1, 0), 0.4, colors.HexColor("#6688AA")),
        ("LINEBELOW", (0, 1), (-1, -1), 0.2, colors.HexColor("#D7DDE5")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return tbl


def build() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    first_frame = Frame(MARGIN, MARGIN, PAGE_W - 2 * MARGIN, PAGE_H - 1.05 * inch, leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
    left_frame = Frame(MARGIN, MARGIN, COLUMN_W, PAGE_H - 1.12 * inch, leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
    right_frame = Frame(MARGIN + COLUMN_W + COLUMN_GAP, MARGIN, COLUMN_W, PAGE_H - 1.12 * inch, leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
    doc = BaseDocTemplate(str(OUTPUT), pagesize=A4, leftMargin=MARGIN, rightMargin=MARGIN, topMargin=MARGIN, bottomMargin=MARGIN)
    doc.addPageTemplates([
        PageTemplate(id="first", frames=[first_frame]),
        PageTemplate(id="two", frames=[left_frame, right_frame], onPage=add_page_number),
    ])

    selected = METRICS["test_metrics_at_selected_threshold"]
    default = METRICS["test_metrics_at_default_threshold"]
    drifted = ", ".join(DRIFT["drifted_features"]).replace("_", " ")
    story = []
    story += [
        p("AI System for Detecting Silent Failures in ML Pipelines: Implementation and Experimental Analysis", title),
        p("Dishan Hari Seelam, Harsha M Gowda, Thejas Naika C, and Revathy B D<br/>Department of CSE (AI&amp;ML), Vidyavardhaka College of Engineering, Mysuru, India", authors),
        p("<b>Abstract - </b>This report implements and evaluates an AI-based silent failure detection system for an e-commerce fraud-classification pipeline. Building on the surveyed need for integrated ML observability, the system combines feature validation, Population Stability Index (PSI) and Kolmogorov-Smirnov (KS) drift detection, Isolation Forest anomaly detection, training-serving skew checks, and a unified health score. A 2,000-record synthetic e-commerce dataset with a 2.95% fraud rate was used to train a Random Forest classifier. Threshold selection improved fraud recall from 16.67% to 41.67% on the held-out test set, while the controlled drift experiment detected four shifted features with large PSI values. The implementation demonstrates how multiple weak monitoring signals can be converted into an actionable pipeline-level assessment without requiring immediate production labels.", abstract),
        p("<b>Index Terms - </b>ML observability, silent failure, data drift, PSI, KS test, Isolation Forest, training-serving skew, MLOps.", abstract),
        p("I. INTRODUCTION", section),
        p("Production machine learning systems can continue to return predictions after their assumptions have failed. The submitted survey identifies data drift, feature corruption, anomalous prediction behavior, and training-serving skew as silent failures because they are not necessarily accompanied by runtime exceptions. Individual monitoring methods exist, but the survey concludes that they are often fragmented and offer limited diagnosis [1]-[10].", body),
        p("This project implements the survey's proposed monitoring direction in an e-commerce fraud setting. Its contribution is not a new base classifier; rather, it is a unified decision layer that consolidates heterogeneous monitoring signals into a health score, severity, failure type, and recommended action.", body),
        p("II. RELATED WORK AND DESIGN RATIONALE", section),
        table([
            ["Reference", "Relevant contribution", "Project response"],
            ["Shankar et al. [4]", "Detection, diagnosis, and reaction; PSI and KS for unlabeled monitoring.", "Implements PSI/KS batch drift tests and presents feature-level results."],
            ["Leest et al. [3]", "Industry monitoring remains fragmented across metrics and metadata.", "Unifies validation, drift, anomaly, and skew findings in one dashboard score."],
            ["Dara and Cavus [8]", "Explainable drift diagnosis improves retraining decisions.", "Stores feature-level drift results; SHAP is retained as the next root-cause extension."],
            ["Reddy [9]; Sana [10]", "Anomaly monitoring requires historical context and can create false positives.", "Uses Isolation Forest as an additional signal rather than a sole failure decision."],
        ], [0.85 * inch, 2.25 * inch, 3.25 * inch]),
        p("<b>Table I.</b> Alignment of the implementation with the studies reviewed in the submitted survey.", caption),
        p("The framework deliberately prioritizes label-independent signals. This addresses the survey's observation that production ground truth may be unavailable or delayed. However, the current evaluation is synthetic and should be treated as implementation validation rather than proof of performance in a live business environment.", body),
        p("III. SYSTEM ARCHITECTURE", section),
        p("The deployed FastAPI service validates each incoming transaction, derives the same timestamp features used during training, produces a fraud probability, applies the tuned alert threshold, and evaluates an Isolation Forest anomaly detector. Batch monitoring compares current features with the training reference distribution using PSI and KS. A training contract checks feature order and preprocessing version. The dashboard exposes all saved artifacts and a unified health engine.", body),
        p("The health score starts at 100 and deducts 30 points for failed validation, up to 30 for drift, 20 for a prediction anomaly, and 20 for training-serving skew. Scores of 85 or more are healthy; 60-84 require attention; below 60 requires intervention.", body),
        NextPageTemplate("two"), PageBreak(),
        p("IV. EXPERIMENTAL SETUP", section),
        p(f"The experiment uses {METRICS['dataset_rows']:,} synthetic e-commerce transactions. The target is <i>is_fraudulent</i>; {METRICS['fraud_rate']:.2%} of transactions are positive. The data split contains {METRICS['training_rows']:,} training, {METRICS['validation_rows']:,} validation, and {METRICS['test_rows']:,} test records. Categorical fields are one-hot encoded and numerical fields use median imputation within one scikit-learn pipeline, preventing a preprocessing mismatch between model training and normal API inference.", body),
        p("A Random Forest classifier is trained with class weighting because fraud is rare. The decision threshold is selected on the validation partition using F2 score, which weights recall twice as strongly as precision. The test partition is never used for selecting the threshold.", body),
        p("A controlled production batch of 200 samples is created from the reference data. The experiment multiplies order value by 3.5, multiplies unit price by 2.8, increases customer age by 25 years (capped at 100), and adds 900 km to shipping distance. This known intervention enables deterministic drift verification.", body),
        p("V. MODEL RESULTS", section),
        metric_chart(), p("Fig. 1. Test-set precision, recall, and F1 before and after validation-selected threshold tuning.", caption),
        table([
            ["T", "P", "R", "F1", "CM (TN, FP, FN, TP)"],
            [f"{default['threshold']:.2f}", f"{default['precision']:.3f}", f"{default['recall']:.3f}", f"{default['f1']:.3f}", ", ".join(str(v) for row in default['confusion_matrix'] for v in row)],
            [f"{selected['threshold']:.2f}", f"{selected['precision']:.3f}", f"{selected['recall']:.3f}", f"{selected['f1']:.3f}", ", ".join(str(v) for row in selected['confusion_matrix'] for v in row)],
        ], [0.40 * inch, 0.45 * inch, 0.45 * inch, 0.45 * inch, 1.85 * inch]),
        p("<b>Table II.</b> Test-set threshold T, precision P, recall R, F1, and confusion-matrix counts. TN, FP, FN, and TP denote true negative, false positive, false negative, and true positive.", caption),
        p(f"The selected threshold is {selected['threshold']:.2f}. Fraud recall rises from {default['recall']:.2%} to {selected['recall']:.2%}, detecting 5 of 12 test fraud cases rather than 2 of 12. This benefit comes with lower precision ({selected['precision']:.2%}) and 17 false positives. In a fraud-review workflow, this is an explicit operating trade-off: more suspected orders are routed to review in return for fewer missed fraud cases. The threshold-independent ROC-AUC is {METRICS['roc_auc']:.3f} and average precision is {METRICS['average_precision']:.3f}; these moderate values are appropriate to report alongside accuracy because the class distribution is highly imbalanced.", body),
        p("VI. DRIFT-DETECTION EXPERIMENT", section),
        psi_chart(), p("Fig. 2. PSI values for intentionally shifted features in the 200-record production batch.", caption),
        p(f"The drift detector returned <b>{DRIFT['status']}</b> for the 200-record batch. The affected features were {drifted}. The largest PSI values were shipping distance ({DRIFT['feature_results']['shipping_distance_km']['psi']:.4f}) and customer age ({DRIFT['feature_results']['customer_age']['psi']:.4f}); both exceed the configured 0.20 warning threshold by a substantial margin. Their KS p-values are effectively zero after rounding, consistent with a changed distribution. Features not deliberately modified remained below the PSI threshold and had non-significant KS results, supporting the detector's specificity in this controlled experiment.", body),
        p("VII. ANALYSIS OF MONITORING COVERAGE", section),
        table([
            ["Failure signal", "Implementation evidence", "Actionable response"],
            ["Feature validation", "Rejects unseen categorical values and implausible numeric ranges before inference.", "Correct source schema / ETL issue."],
            ["Data drift", "PSI and KS identify shifted feature distributions in a batch.", "Review shift and retrain if sustained."],
            ["Prediction anomaly", "Isolation Forest evaluates probability and selected transaction features.", "Manually review anomalous transaction."],
            ["Training-serving skew", "Checks feature list, order, and preprocessing version against training contract.", "Synchronize serving pipeline."],
            ["Unified health", "Combines all signals into 0-100 score and severity.", "Prioritize intervention by aggregate risk."],
        ], [1.15 * inch, 2.85 * inch, 2.35 * inch]),
        p("<b>Table III.</b> Implemented monitoring coverage and the associated remediation path.", caption),
        p("This design directly addresses the survey's lack-of-unification gap. A drift alert alone can be noisy; an anomaly alone cannot prove data shift; and a skew check cannot identify a corrupt input. The health engine preserves each module's evidence but provides one operational status for a reviewer. The weights are transparent and can be adjusted based on business risk or alert burden.", body),
        p("VIII. LIMITATIONS AND FUTURE WORK", section),
        p("The results should not be overgeneralized. The e-commerce dataset is synthetic, only 59 of 2,000 observations are fraudulent, and the test set contains 12 fraud cases. Confidence intervals and external benchmark comparisons are therefore not claimed. PSI/KS detection measures distribution change, not necessarily concept drift or model performance degradation. The Isolation Forest is calibrated on synthetic baseline behavior. In production, monitored batches, feedback labels, threshold review, alert history, and periodic retraining policies are required.", body),
        p("The survey proposes SHAP-based root-cause analysis. The current project preserves this as the next extension: when health becomes warning or critical, SHAP can explain the classifier's top contributing features for the flagged transaction or batch. Additional future work includes automated alerting, a database-backed event log, time-series health trends, and evaluation on a real labeled e-commerce fraud dataset.", body),
        p("IX. CONCLUSION", section),
        p("The implemented system transforms the survey's unified-monitoring proposal into a functioning MLOps prototype. It demonstrates validation, anomaly, drift, and skew monitoring in one application and produces an interpretable health assessment. Threshold tuning improved fraud recall from 16.67% to 41.67%, and the controlled experiment detected all four intentionally shifted features. The main value is therefore not a claim of a new classifier, but a transparent integrated workflow for identifying and responding to silent failures before they become undetected operational risk.", body),
        p("REFERENCES", section),
    ]
    references = [
        "[1] R. Pathak et al., \"Detecting Silent Failures in Multi-Agentic AI Trajectories,\" arXiv preprint arXiv:2505.14144, 2025.",
        "[2] Y. Yang, \"Taming Silent Failures: A Framework for Verifiable AI Reliability,\" arXiv preprint, 2025.",
        "[3] J. Leest, \"Monitoring and Observability of Machine Learning Systems: Current Practices and Gaps,\" arXiv preprint arXiv:2505.19395, 2025.",
        "[4] R. Shankar, R. Arora, A. Jain, and A. Sinha, \"Towards Observability for Production Machine Learning Pipelines,\" in Proc. IEEE Int. Conf. Big Data, 2022.",
        "[5] Z. Jiang et al., \"Training with Confidence: Catching Silent Errors in Deep Learning Training with Automated Proactive Checks,\" Proc. ACM Program. Lang., 2025.",
        "[6] H. Zhao, Y. Guo, and X. Li, \"Lightweight Detection of Silent Data Corruption in Distributed Deep Learning,\" in Proc. IEEE IPDPS, 2025.",
        "[7] K. Nandamuri, \"A Comprehensive Guide to Monitoring and Observability in Machine Learning Infrastructure: From Metrics to Implementation,\" arXiv preprint, 2025.",
        "[8] E. Dara and N. Cavus, \"From XAI to MLOps: Explainable Concept Drift Detection with Profile Drift Detection,\" Expert Systems with Applications, vol. 271, 2025.",
        "[9] V. Reddy, \"Machine Learning-Driven Anomaly Detection in CI/CD Pipelines for Financial Applications,\" International Journal of Computer Applications, vol. 187, no. 12, pp. 15-24, 2024.",
        "[10] S. Sana, \"Developing an AI-Driven Anomaly Detection System for Cloud Data Pipelines: Minimizing Data Quality Issues by 40%,\" International Journal of Advanced Computer Science and Applications, vol. 16, no. 3, pp. 221-231, 2025.",
    ]
    story.extend(p(esc(item), ref) for item in references)
    doc.build(story)


if __name__ == "__main__":
    build()
