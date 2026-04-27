from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.pagesizes import landscape
from reportlab.lib.units import inch
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "presentation"
ASSET_DIR = OUT_DIR / "assets"
PDF_OUT = OUT_DIR / "RL_Final_Project_Presentation.pdf"
TEX_OUT = OUT_DIR / "slide.tex"
LOGO = OUT_DIR / "pic" / "CUHKSZ-Logo.png"

PAGE_W, PAGE_H = landscape((7.5 * inch, 13.333 * inch))
PURPLE = colors.Color(117 / 255, 15 / 255, 109 / 255)
GOLD = colors.Color(221 / 255, 163 / 255, 0 / 255)
LIGHT_GOLD = colors.Color(244 / 255, 223 / 255, 176 / 255)
DARK = colors.Color(38 / 255, 36 / 255, 42 / 255)
MUTED = colors.Color(96 / 255, 92 / 255, 102 / 255)
GREEN = colors.Color(35 / 255, 126 / 255, 86 / 255)
RED = colors.Color(170 / 255, 56 / 255, 56 / 255)
BLUE = colors.Color(45 / 255, 99 / 255, 150 / 255)


def load_json(path: str) -> dict:
    return json.loads((ROOT / path).read_text())


def selected_rollout() -> dict:
    data = load_json("logs/rollout_selected_50/v1_final_rollouts_50.json")
    return data["rollouts"][47]


def v1_summary() -> dict:
    return load_json("logs/rollout_selected_50/summary_v1_final_v3_step_40_v3_step_80_50.json")["versions"]


def fmt_pct(x: float) -> str:
    return f"{100 * x:.0f}%"


def wrapped_lines(text: str, font: str, size: int, width: float) -> list[str]:
    words = text.split()
    lines: list[str] = []
    cur = ""
    for word in words:
        candidate = word if not cur else f"{cur} {word}"
        if stringWidth(candidate, font, size) <= width:
            cur = candidate
        else:
            if cur:
                lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines


def draw_wrapped(c: canvas.Canvas, text: str, x: float, y: float, width: float, size: int = 18,
                 leading: float | None = None, color=DARK, font: str = "Helvetica") -> float:
    leading = leading or size * 1.25
    c.setFont(font, size)
    c.setFillColor(color)
    for line in wrapped_lines(text, font, size, width):
        c.drawString(x, y, line)
        y -= leading
    return y


def draw_header(c: canvas.Canvas, title: str, page_num: int, total: int) -> None:
    c.setFillColor(PURPLE)
    c.rect(0, PAGE_H - 0.72 * inch, PAGE_W, 0.72 * inch, stroke=0, fill=1)
    c.setFillColor(LIGHT_GOLD)
    c.setFont("Helvetica-Bold", 20)
    c.drawString(0.48 * inch, PAGE_H - 0.45 * inch, title)
    c.setFillColor(PURPLE)
    c.rect(0, 0, PAGE_W, 0.34 * inch, stroke=0, fill=1)
    c.setFillColor(LIGHT_GOLD)
    c.setFont("Helvetica", 9)
    c.drawString(0.48 * inch, 0.13 * inch, "RL Final Project | Negotiation Self-Play")
    c.drawRightString(PAGE_W - 0.48 * inch, 0.13 * inch, f"{page_num}/{total}")


def bullet(c: canvas.Canvas, text: str, x: float, y: float, width: float, size: int = 17,
           color=DARK, marker_color=GOLD) -> float:
    c.setFillColor(marker_color)
    c.circle(x, y + 0.06 * inch, 3.5, stroke=0, fill=1)
    return draw_wrapped(c, text, x + 0.18 * inch, y, width - 0.18 * inch, size=size, color=color)


def metric_card(c: canvas.Canvas, x: float, y: float, w: float, h: float, label: str, value: str,
                sub: str = "", color=PURPLE) -> None:
    c.setStrokeColor(colors.Color(0.84, 0.82, 0.86))
    c.setFillColor(colors.white)
    c.roundRect(x, y - h, w, h, 5, stroke=1, fill=1)
    c.setFillColor(color)
    c.setFont("Helvetica-Bold", 24)
    c.drawString(x + 0.16 * inch, y - 0.36 * inch, value)
    c.setFillColor(DARK)
    c.setFont("Helvetica-Bold", 11)
    c.drawString(x + 0.16 * inch, y - 0.64 * inch, label)
    if sub:
        c.setFillColor(MUTED)
        c.setFont("Helvetica", 9)
        c.drawString(x + 0.16 * inch, y - 0.84 * inch, sub)


def make_charts() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

    steps = [20, 40, 60, 80, 100]
    legal = [92.97, 85.94, 86.72, 84.38, 89.84]
    buyer = [55.95, 45.93, 43.78, 32.98, 37.86]
    seller = [32.18, 27.07, 31.23, 36.93, 44.54]
    fig, ax1 = plt.subplots(figsize=(7.4, 3.25), dpi=180)
    ax1.plot(steps, legal, marker="o", color="#237e56", label="Legal deal rate")
    ax1.set_ylabel("Legal deal rate (%)")
    ax1.set_ylim(75, 100)
    ax1.set_xlabel("Stage3 balanced V1 step")
    ax2 = ax1.twinx()
    ax2.plot(steps, buyer, marker="s", color="#2d6396", label="Raw buyer reward")
    ax2.plot(steps, seller, marker="^", color="#750f6d", label="Raw seller reward")
    ax2.set_ylabel("Raw reward")
    ax2.set_ylim(20, 65)
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="lower left", frameon=False)
    ax1.grid(axis="y", alpha=0.22)
    fig.tight_layout()
    fig.savefig(ASSET_DIR / "stage3_v1_eval.png", transparent=False, bbox_inches="tight")
    plt.close(fig)

    versions = v1_summary()
    names = ["V1 final", "V3 step 40", "V3 step 80"]
    keys = ["v1_final", "v3_step_40", "v3_step_80"]
    deal_rates = [versions[k]["summary"]["deal_rate"] * 100 for k in keys]
    seller_viol = [versions[k]["summary"]["violation_seller_rate"] * 100 for k in keys]
    buyer_viol = [versions[k]["summary"]["violation_buyer_rate"] * 100 for k in keys]
    fig, ax = plt.subplots(figsize=(7.2, 3.25), dpi=180)
    x = range(len(names))
    ax.bar([i - 0.22 for i in x], deal_rates, width=0.22, label="Deal", color="#237e56")
    ax.bar(x, seller_viol, width=0.22, label="Seller violation", color="#aa3838")
    ax.bar([i + 0.22 for i in x], buyer_viol, width=0.22, label="Buyer violation", color="#dda300")
    ax.set_xticks(list(x), names)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Rate (%) on 50 selected validation scenarios")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=False, ncol=3, loc="upper right")
    fig.tight_layout()
    fig.savefig(ASSET_DIR / "rollout_compare.png", transparent=False, bbox_inches="tight")
    plt.close(fig)

    rollout = selected_rollout()
    sc = rollout["scenario"]
    price = rollout["result"]["deal_price"]
    buyer_share = (sc["buyer_budget"] - price) / (sc["buyer_budget"] - sc["seller_cost"])
    seller_share = 1 - buyer_share
    fig, ax = plt.subplots(figsize=(6.4, 1.7), dpi=180)
    ax.barh([0], [buyer_share * 100], color="#2d6396", label=f"Buyer surplus {buyer_share * 100:.1f}%")
    ax.barh([0], [seller_share * 100], left=[buyer_share * 100], color="#750f6d",
            label=f"Seller surplus {seller_share * 100:.1f}%")
    ax.set_xlim(0, 100)
    ax.set_yticks([])
    ax.set_xlabel("Surplus split within bargaining zone")
    ax.legend(frameon=False, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 1.0))
    ax.grid(axis="x", alpha=0.15)
    fig.tight_layout()
    fig.savefig(ASSET_DIR / "selected_surplus.png", transparent=False, bbox_inches="tight")
    plt.close(fig)


class Deck:
    def __init__(self, path: Path, total: int):
        self.c = canvas.Canvas(str(path), pagesize=(PAGE_W, PAGE_H))
        self.page = 0
        self.total = total

    def slide(self, title: str):
        self.page += 1
        if self.page > 1:
            self.c.showPage()
        draw_header(self.c, title, self.page, self.total)

    def save(self):
        self.c.save()


def draw_title(deck: Deck) -> None:
    c = deck.c
    c.setFillColor(PURPLE)
    c.rect(0, 0, PAGE_W, PAGE_H, stroke=0, fill=1)
    c.setFillColor(GOLD)
    c.rect(0, 0, PAGE_W, 0.2 * inch, stroke=0, fill=1)
    if LOGO.exists():
        c.drawImage(str(LOGO), PAGE_W - 2.35 * inch, PAGE_H - 1.45 * inch, 1.75 * inch, 0.72 * inch,
                    preserveAspectRatio=True, mask="auto")
    c.setFillColor(LIGHT_GOLD)
    c.setFont("Helvetica-Bold", 37)
    c.drawString(0.72 * inch, 4.75 * inch, "RL for Bilateral")
    c.drawString(0.72 * inch, 4.22 * inch, "Price Negotiation")
    c.setFont("Helvetica", 18)
    c.drawString(0.76 * inch, 3.62 * inch, "Self-play GRPO with role-specific adapters")
    c.setFont("Helvetica", 13)
    c.drawString(0.76 * inch, 1.02 * inch, "The Chinese University of Hong Kong, Shenzhen")
    c.drawRightString(PAGE_W - 0.72 * inch, 1.02 * inch, "Final Project Presentation")


def build_pdf() -> None:
    total = 12
    deck = Deck(PDF_OUT, total)
    deck.page = 1
    draw_title(deck)

    c = deck.c
    deck.slide("Problem and Goal")
    y = PAGE_H - 1.25 * inch
    bullet(c, "Train one base LLM to act as either buyer or seller in second-hand marketplace negotiation.", 0.75 * inch, y, 5.6 * inch)
    y -= 0.72 * inch
    bullet(c, "Private values are asymmetric: buyer has a maximum budget; seller has a minimum acceptable cost.", 0.75 * inch, y, 5.6 * inch)
    y -= 0.72 * inch
    bullet(c, "The policy must bargain, close legal deals, avoid private-value leakage, and use strict action formats.", 0.75 * inch, y, 5.6 * inch)
    metric_card(c, 7.2 * inch, PAGE_H - 1.35 * inch, 1.65 * inch, 0.95 * inch, "Presentation rubric", "50 pts", "10 minute talk", PURPLE)
    metric_card(c, 7.2 * inch, PAGE_H - 2.55 * inch, 1.65 * inch, 0.95 * inch, "Key emphasis", "Demo", "what worked / failed", GOLD)
    c.setFillColor(colors.Color(0.96, 0.95, 0.97))
    c.roundRect(0.78 * inch, 1.02 * inch, 7.9 * inch, 1.05 * inch, 5, stroke=0, fill=1)
    draw_wrapped(c, "Core question: can RL improve a conversational model's economic behavior beyond format-following SFT?",
                 1.02 * inch, 1.68 * inch, 7.35 * inch, size=17, font="Helvetica-Bold", color=PURPLE)

    deck.slide("Environment")
    y = PAGE_H - 1.23 * inch
    bullet(c, "State: item metadata, market reference, hidden buyer budget, hidden seller cost, and dialogue history.", 0.72 * inch, y, 5.9 * inch)
    y -= 0.62 * inch
    bullet(c, "Action space: natural-language response plus a parseable action: offer, deal, or walkaway.", 0.72 * inch, y, 5.9 * inch)
    y -= 0.62 * inch
    bullet(c, "Terminal outcomes: legal deal, buyer/seller violation, walkaway, timeout, or format error.", 0.72 * inch, y, 5.9 * inch)
    c.setFillColor(colors.Color(0.98, 0.98, 0.98))
    c.roundRect(0.85 * inch, 1.15 * inch, 8.65 * inch, 1.35 * inch, 5, stroke=1, fill=1)
    c.setFont("Helvetica-Bold", 14)
    c.setFillColor(PURPLE)
    c.drawString(1.08 * inch, 2.1 * inch, "Strict output grammar")
    c.setFont("Courier", 13)
    c.setFillColor(DARK)
    c.drawString(1.08 * inch, 1.72 * inch, "[PRICE: 1220] ...")
    c.drawString(3.75 * inch, 1.72 * inch, "<deal>1220</deal>")
    c.drawString(6.25 * inch, 1.72 * inch, "<walkaway>")

    deck.slide("Data Pipeline")
    cards = [
        ("Scenario sampler", "5k RL scenarios", "rule-generated from product templates"),
        ("SFT dialogues", "1k scenarios", "API-generated demonstrations"),
        ("Role split", "buyer / seller", "same base model, separate behavior"),
        ("GRPO input", "online rollouts", "not fixed offline dialogues"),
    ]
    for i, (title, value, sub) in enumerate(cards):
        x = 0.72 * inch + (i % 2) * 4.45 * inch
        y = PAGE_H - (1.45 + (i // 2) * 1.8) * inch
        metric_card(c, x, y, 3.75 * inch, 1.24 * inch, title, value, sub, [PURPLE, BLUE, GOLD, GREEN][i])
    draw_wrapped(c, "RL scenario difficulty is controlled by bargaining-zone width: near-zero, narrow, balanced, and wide.",
                 1.02 * inch, 1.55 * inch, 7.85 * inch, size=18, font="Helvetica-Bold", color=DARK)

    deck.slide("Method")
    y = PAGE_H - 1.23 * inch
    bullet(c, "Start from an SFT checkpoint so the model already knows roles, tone, and output formats.", 0.72 * inch, y, 6.05 * inch)
    y -= 0.62 * inch
    bullet(c, "Attach two LoRA adapters: one optimized when acting as buyer, one when acting as seller.", 0.72 * inch, y, 6.05 * inch)
    y -= 0.62 * inch
    bullet(c, "Use GRPO over grouped self-play rollouts, updating only the active role's adapter.", 0.72 * inch, y, 6.05 * inch)
    y -= 0.62 * inch
    bullet(c, "Stage schedule: buyer-only -> seller-only -> alternating buyer/seller optimization.", 0.72 * inch, y, 6.05 * inch)
    c.setStrokeColor(GOLD)
    c.setLineWidth(2.5)
    for x in [1.25, 3.65, 6.05]:
        c.roundRect(x * inch, 1.25 * inch, 1.58 * inch, 0.75 * inch, 5, stroke=1, fill=0)
    c.setFillColor(PURPLE)
    c.setFont("Helvetica-Bold", 13)
    c.drawCentredString(2.04 * inch, 1.55 * inch, "SFT")
    c.drawCentredString(4.44 * inch, 1.55 * inch, "Self-play")
    c.drawCentredString(6.84 * inch, 1.55 * inch, "GRPO")
    c.line(2.83 * inch, 1.62 * inch, 3.65 * inch, 1.62 * inch)
    c.line(5.23 * inch, 1.62 * inch, 6.05 * inch, 1.62 * inch)

    deck.slide("Reward Design")
    y = PAGE_H - 1.22 * inch
    bullet(c, "Legal deal utilities are normalized within the bargaining zone.", 0.72 * inch, y, 6.1 * inch)
    y -= 0.55 * inch
    c.setFont("Courier", 15)
    c.setFillColor(DARK)
    c.drawString(1.02 * inch, y, "buyer = (budget - price) / (budget - cost)")
    y -= 0.32 * inch
    c.drawString(1.02 * inch, y, "seller = (price - cost) / (budget - cost)")
    y -= 0.58 * inch
    bullet(c, "Hard penalties handle invalid deals, format errors, extreme offers, and private-value leakage.", 0.72 * inch, y, 6.1 * inch)
    y -= 0.62 * inch
    bullet(c, "Balanced V1 adds a shared Nash-style term to reduce one-sided equilibria.", 0.72 * inch, y, 6.1 * inch)
    c.drawImage(str(ASSET_DIR / "selected_surplus.png"), 1.3 * inch, 1.05 * inch, 6.75 * inch, 1.62 * inch,
                preserveAspectRatio=True, mask="auto")

    deck.slide("Training and Checkpoint Selection")
    c.drawImage(str(ASSET_DIR / "stage3_v1_eval.png"), 0.66 * inch, 2.62 * inch, 7.7 * inch, 3.36 * inch,
                preserveAspectRatio=True, mask="auto")
    y = 2.14 * inch
    bullet(c, "V1 final was selected because it recovered legal-deal quality while keeping buyer/seller reward closer.", 0.78 * inch, y, 7.7 * inch, size=15)
    y -= 0.48 * inch
    bullet(c, "Step 80 had the closest reward split but lower legal-deal rate; final is a better presentation checkpoint.", 0.78 * inch, y, 7.7 * inch, size=15)

    deck.slide("Quantitative Results")
    versions = v1_summary()
    s = versions["v1_final"]["summary"]
    metric_card(c, 0.72 * inch, PAGE_H - 1.35 * inch, 1.62 * inch, 0.95 * inch, "Deal rate", fmt_pct(s["deal_rate"]), "V1 final, n=50", GREEN)
    metric_card(c, 2.58 * inch, PAGE_H - 1.35 * inch, 1.62 * inch, 0.95 * inch, "Format errors", fmt_pct(s["format_error_rate"]), "strict parser", PURPLE)
    metric_card(c, 4.44 * inch, PAGE_H - 1.35 * inch, 1.62 * inch, 0.95 * inch, "Avg rounds", f"{s['avg_rounds']:.1f}", "concise negotiation", BLUE)
    metric_card(c, 6.3 * inch, PAGE_H - 1.35 * inch, 1.62 * inch, 0.95 * inch, "Rewards", "31.6 / 23.2", "buyer / seller", GOLD)
    c.drawImage(str(ASSET_DIR / "rollout_compare.png"), 1.0 * inch, 1.18 * inch, 7.2 * inch, 3.38 * inch,
                preserveAspectRatio=True, mask="auto")

    deck.slide("Ablation Lessons")
    rows = [
        ("Cold seller guard", "Stopped buyer exploiting weak seller with below-cost deals."),
        ("Stage2 seller comparison", "Step100 seller kept 96.9% legal deals and more buyer surplus than best seller."),
        ("Balanced V1", "Best overall trade-off: 86% deal rate on 50-rollout sample, 0% format error."),
        ("V2/V3 guardrails", "Improved some behavior targets but caused more seller violations or lower deal rate."),
    ]
    y = PAGE_H - 1.28 * inch
    for label, text in rows:
        c.setFillColor(PURPLE)
        c.setFont("Helvetica-Bold", 14)
        c.drawString(0.82 * inch, y, label)
        draw_wrapped(c, text, 2.95 * inch, y, 5.95 * inch, size=14, color=DARK)
        y -= 0.72 * inch

    deck.slide("Selected V1 Rollout")
    rollout = selected_rollout()
    sc = rollout["scenario"]
    res = rollout["result"]
    z = sc["buyer_budget"] - sc["seller_cost"]
    buyer_share = (sc["buyer_budget"] - res["deal_price"]) / z
    seller_share = 1 - buyer_share
    metric_card(c, 0.72 * inch, PAGE_H - 1.35 * inch, 1.6 * inch, 0.95 * inch, "Scenario", "AirPods", "index 47", PURPLE)
    metric_card(c, 2.5 * inch, PAGE_H - 1.35 * inch, 1.6 * inch, 0.95 * inch, "Deal", "1220", "budget 1412", GREEN)
    metric_card(c, 4.28 * inch, PAGE_H - 1.35 * inch, 1.6 * inch, 0.95 * inch, "Split", f"{buyer_share*100:.1f}/{seller_share*100:.1f}", "buyer/seller", BLUE)
    metric_card(c, 6.06 * inch, PAGE_H - 1.35 * inch, 1.6 * inch, 0.95 * inch, "Reward", "56.9/56.0", "buyer/seller", GOLD)
    dialogue = [
        ("Seller", "1280: condition is close to new; this is already a meaningful concession."),
        ("Buyer", "1200: I am serious and can pay immediately if this works."),
        ("Seller", "1240: I can move down another step, but not much further."),
        ("Buyer", "1210: I can add 10 and close now."),
        ("Seller", "1230: final small concession."),
        ("Buyer", "1220: one more 10; if accepted I will buy now."),
        ("Seller", "deal at 1220."),
    ]
    y = 4.0 * inch
    for role, text in dialogue:
        c.setFillColor(PURPLE if role == "Seller" else BLUE)
        c.setFont("Helvetica-Bold", 11)
        c.drawString(0.82 * inch, y, role)
        draw_wrapped(c, text, 1.62 * inch, y, 7.45 * inch, size=12, color=DARK)
        y -= 0.38 * inch
    c.setFillColor(MUTED)
    c.setFont("Helvetica", 9)
    c.drawString(0.82 * inch, 0.72 * inch, "Displayed in English with history labels removed; source: logs/rollout_selected_50/v1_final_rollouts_50.json.")

    deck.slide("What Worked / What Did Not")
    left_x, right_x = 0.78 * inch, 5.08 * inch
    c.setFillColor(GREEN)
    c.setFont("Helvetica-Bold", 18)
    c.drawString(left_x, PAGE_H - 1.35 * inch, "Worked")
    y = PAGE_H - 1.86 * inch
    for text in [
        "Role conditioning plus SFT gave stable parseable negotiation behavior.",
        "GRPO improved strategic behavior beyond fixed demonstrations.",
        "Shared balance reward made qualitative demos less one-sided than seller-dominant checkpoints.",
    ]:
        y = bullet(c, text, left_x, y, 3.65 * inch, size=13, marker_color=GREEN) - 0.23 * inch
    c.setFillColor(RED)
    c.setFont("Helvetica-Bold", 18)
    c.drawString(right_x, PAGE_H - 1.35 * inch, "Limitations")
    y = PAGE_H - 1.86 * inch
    for text in [
        "V1 still sometimes leaks private-value language or accepts too quickly.",
        "More aggressive V2/V3 penalties increased violations on sampled rollouts.",
        "Evaluation is self-play based; human preference evaluation is still missing.",
    ]:
        y = bullet(c, text, right_x, y, 3.65 * inch, size=13, marker_color=RED) - 0.23 * inch

    deck.slide("Takeaways")
    y = PAGE_H - 1.32 * inch
    bullet(c, "Formulating bargaining as self-play RL is feasible, but reward design controls the equilibrium.", 0.78 * inch, y, 7.7 * inch, size=18)
    y -= 0.78 * inch
    bullet(c, "The best presentation checkpoint is V1 final: high deal rate, no format errors, and a clear balanced demo rollout.", 0.78 * inch, y, 7.7 * inch, size=18)
    y -= 0.78 * inch
    bullet(c, "The negative result is useful: stronger guardrails need better credit assignment, not just larger penalties.", 0.78 * inch, y, 7.7 * inch, size=18)
    c.setFillColor(colors.Color(0.96, 0.95, 0.97))
    c.roundRect(1.05 * inch, 1.03 * inch, 7.2 * inch, 1.1 * inch, 5, stroke=0, fill=1)
    draw_wrapped(c, "Next: add human-rated dialogue quality, explicit leakage classifiers, and opponent diversity during training.",
                 1.3 * inch, 1.7 * inch, 6.65 * inch, size=16, font="Helvetica-Bold", color=PURPLE)

    deck.save()


def tex_escape(s: str) -> str:
    return (s.replace("\\", "\\textbackslash{}")
             .replace("&", "\\&")
             .replace("%", "\\%")
             .replace("$", "\\$")
             .replace("#", "\\#")
             .replace("_", "\\_")
             .replace("{", "\\{")
             .replace("}", "\\}")
             .replace("~", "\\textasciitilde{}")
             .replace("^", "\\textasciicircum{}"))


def build_tex() -> None:
    rollout = selected_rollout()
    sc = rollout["scenario"]
    res = rollout["result"]
    z = sc["buyer_budget"] - sc["seller_cost"]
    buyer_share = (sc["buyer_budget"] - res["deal_price"]) / z
    seller_share = 1 - buyer_share
    tex = rf"""\documentclass{{beamer}}

\usepackage[T1]{{fontenc}}
\usepackage[utf8]{{inputenc}}
\usepackage{{hyperref}}
\usepackage{{amsmath,booktabs,graphicx}}
\usepackage{{CUHKSZ}}

\author{{Final Project Team}}
\title{{RL for Bilateral Price Negotiation}}
\institute{{The Chinese University of Hong Kong, Shenzhen}}
\date{{\today}}

\begin{{document}}

\begin{{frame}}
    \titlepage
    \begin{{figure}}
        \centering
        \includegraphics[width=0.3\linewidth]{{pic/CUHKSZ-Logo.pdf}}
    \end{{figure}}
\end{{frame}}

\begin{{frame}}{{Roadmap}}
    \tableofcontents
\end{{frame}}

\section{{Problem}}
\begin{{frame}}{{Problem and Goal}}
\begin{{itemize}}
    \item Train one LLM to act as buyer or seller in second-hand marketplace negotiation.
    \item Private values are asymmetric: buyer budget vs. seller cost.
    \item Success requires legal deals, bargaining skill, no private-value leakage, and strict output formats.
\end{{itemize}}
\end{{frame}}

\begin{{frame}}{{Environment}}
\begin{{itemize}}
    \item State: item metadata, private role value, market reference, and dialogue history.
    \item Actions: \texttt{{[PRICE: x]}}, \texttt{{<deal>x</deal>}}, or \texttt{{<walkaway>}}.
    \item Outcomes: legal deal, buyer/seller violation, walkaway, timeout, or format error.
\end{{itemize}}
\end{{frame}}

\section{{Method}}
\begin{{frame}}{{Method Overview}}
\begin{{itemize}}
    \item SFT teaches role style, format following, and basic multi-turn negotiation.
    \item GRPO uses online self-play rollouts rather than fixed dialogues.
    \item Buyer and seller use separate LoRA adapters on the same base model.
    \item Training stages: buyer-only, seller-only, then alternating optimization.
\end{{itemize}}
\end{{frame}}

\begin{{frame}}{{Reward Design}}
\begin{{align*}}
u_b &= \frac{{budget - price}}{{budget - cost}}\\
u_s &= \frac{{price - cost}}{{budget - cost}}
\end{{align*}}
\begin{{itemize}}
    \item Legal deal utility is normalized inside the bargaining zone.
    \item Penalties cover invalid deals, format errors, leakage, and extreme offers.
    \item Balanced V1 adds a shared Nash-style term to reduce one-sided equilibria.
\end{{itemize}}
\end{{frame}}

\section{{Results}}
\begin{{frame}}{{Training and Checkpoint Selection}}
\begin{{figure}}
\centering
\includegraphics[width=0.86\linewidth]{{assets/stage3_v1_eval.png}}
\end{{figure}}
\end{{frame}}

\begin{{frame}}{{Quantitative Results}}
\begin{{figure}}
\centering
\includegraphics[width=0.86\linewidth]{{assets/rollout_compare.png}}
\end{{figure}}
\end{{frame}}

\begin{{frame}}{{Selected V1 Rollout}}
\begin{{itemize}}
    \item Scenario: AirPods Pro 2; buyer budget 1412; seller cost 1034.
    \item Deal price: 1220.
    \item Surplus split: buyer {buyer_share * 100:.1f}\%, seller {seller_share * 100:.1f}\%.
    \item Rewards: buyer {res['buyer_reward']:.1f}, seller {res['seller_reward']:.1f}.
\end{{itemize}}
\begin{{figure}}
\centering
\includegraphics[width=0.72\linewidth]{{assets/selected_surplus.png}}
\end{{figure}}
\end{{frame}}

\begin{{frame}}{{Demo Dialogue: V1 Rollout Index 47}}
\small
\begin{{tabular}}{{@{{}}p{{0.16\linewidth}}p{{0.76\linewidth}}@{{}}}}
\toprule
Seller & 1280: condition is close to new; this is already a meaningful concession.\\
Buyer & 1200: I am serious and can pay immediately if this works.\\
Seller & 1240: I can move down another step, but not much further.\\
Buyer & 1210: I can add 10 and close now.\\
Seller & 1230: final small concession.\\
Buyer & 1220: one more 10; if accepted I will buy now.\\
Seller & \texttt{{<deal>1220</deal>}}\\
\bottomrule
\end{{tabular}}
\end{{frame}}

\section{{Discussion}}
\begin{{frame}}{{What Worked / What Did Not}}
\begin{{columns}}
\column{{0.48\linewidth}}
\textbf{{Worked}}
\begin{{itemize}}
    \item Stable parseable negotiation.
    \item GRPO improved strategic behavior.
    \item V1 gave a usable balanced demo.
\end{{itemize}}
\column{{0.48\linewidth}}
\textbf{{Limitations}}
\begin{{itemize}}
    \item Residual leakage and label artifacts.
    \item Stronger guardrails increased violations.
    \item Human preference evaluation is missing.
\end{{itemize}}
\end{{columns}}
\end{{frame}}

\begin{{frame}}{{Takeaways}}
\begin{{itemize}}
    \item Self-play RL is feasible for negotiation, but reward design determines the equilibrium.
    \item V1 final is the best presentation checkpoint: high deal rate, no format errors, and a clear demo.
    \item Negative results are informative: guardrails need better credit assignment, not just larger penalties.
\end{{itemize}}
\end{{frame}}

\end{{document}}
"""
    TEX_OUT.write_text(tex)


def main() -> None:
    make_charts()
    build_tex()
    build_pdf()
    print(f"Wrote {PDF_OUT}")
    print(f"Wrote {TEX_OUT}")


if __name__ == "__main__":
    main()
