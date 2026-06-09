"""
AP Guardian - Accounts Payable Risk & Recovery Assessment Platform
Enterprise PDF Report Generation System
"""

import os
import io
import json
from datetime import datetime, timedelta
from typing import List, Optional
from decimal import Decimal

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel

# Supabase is optional - only used for production database
try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False
    Client = None

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, mm
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, Image, Flowable, KeepTogether
)
from reportlab.graphics.shapes import Drawing, Rect, String, Line
from reportlab.graphics.charts.piecharts import Pie
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.pdfgen import canvas

app = FastAPI(title="AP Guardian", description="AP Risk & Recovery Assessment API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SUPABASE_URL = os.getenv("SUPABASE_URL") or os.getenv("VITE_SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("VITE_SUPABASE_ANON_KEY")
supabase = None
if SUPABASE_AVAILABLE and SUPABASE_URL and SUPABASE_KEY:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Color Palette - Enterprise Grade
NAVY = colors.HexColor("#0F172A")
SLATE = colors.HexColor("#475569")
BLUE = colors.HexColor("#2563EB")
ACCENT_BLUE = colors.HexColor("#3B82F6")
LIGHT_BLUE = colors.HexColor("#DBEAFE")
GRAY_100 = colors.HexColor("#F1F5F9")
GRAY_200 = colors.HexColor("#E2E8F0")
GRAY_300 = colors.HexColor("#CBD5E1")
GRAY_500 = colors.HexColor("#64748B")
GRAY_600 = colors.HexColor("#475569")
GRAY_700 = colors.HexColor("#334155")
WHITE = colors.white
RISK_LOW = colors.HexColor("#6B7280")
RISK_MEDIUM = colors.HexColor("#D97706")
RISK_HIGH = colors.HexColor("#DC2626")
RECOVERED = colors.HexColor("#059669")
GREEN_500 = colors.HexColor("#22C55E")

# Pydantic Models
class Finding(BaseModel):
    id: str
    vendor_name: str
    vendor_id: str
    finding_type: str
    severity: str
    amount: float
    currency: str = "USD"
    transaction_date: str
    description: str
    recommended_action: str
    status: str = "open"
    recovery_status: Optional[str] = None
    potential_recovery: Optional[float] = None

class VendorRisk(BaseModel):
    vendor_name: str
    vendor_id: str
    risk_score: int
    total_findings: int
    at_risk_amount: float
    recovered_amount: float
    outstanding_amount: float

class ReportSummary(BaseModel):
    client_name: str
    audit_period_start: str
    audit_period_end: str
    total_transactions: int
    total_findings: int
    critical_findings: int
    high_findings: int
    medium_findings: int
    low_findings: int
    total_at_risk: float
    total_recovered: float
    recovery_rate: float
    findings_by_type: dict
    vendors: List[VendorRisk]
    findings: List[Finding]


def create_styles():
    """Create custom paragraph styles for the report."""
    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(
        name='CoverTitle',
        parent=styles['Heading1'],
        fontSize=36,
        textColor=NAVY,
        alignment=TA_CENTER,
        spaceAfter=12,
        fontName='Helvetica-Bold',
        leading=42
    ))

    styles.add(ParagraphStyle(
        name='CoverSubtitle',
        parent=styles['Normal'],
        fontSize=14,
        textColor=SLATE,
        alignment=TA_CENTER,
        spaceAfter=30,
        fontName='Helvetica'
    ))

    styles.add(ParagraphStyle(
        name='SectionHeader',
        parent=styles['Heading1'],
        fontSize=18,
        textColor=NAVY,
        spaceBefore=20,
        spaceAfter=12,
        fontName='Helvetica-Bold',
        borderPadding=0,
    ))

    styles.add(ParagraphStyle(
        name='SubsectionHeader',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=SLATE,
        spaceBefore=16,
        spaceAfter=8,
        fontName='Helvetica-Bold'
    ))

    styles.add(ParagraphStyle(
        name='CustomBody',
        parent=styles['Normal'],
        fontSize=10,
        textColor=GRAY_700,
        alignment=TA_JUSTIFY,
        spaceBefore=6,
        spaceAfter=6,
        fontName='Helvetica',
        leading=14
    ))

    styles.add(ParagraphStyle(
        name='ExecutiveNarrative',
        parent=styles['Normal'],
        fontSize=11,
        textColor=GRAY_700,
        alignment=TA_JUSTIFY,
        spaceBefore=12,
        spaceAfter=12,
        fontName='Helvetica',
        leading=16
    ))

    styles.add(ParagraphStyle(
        name='MetricLabel',
        parent=styles['Normal'],
        fontSize=9,
        textColor=GRAY_500,
        alignment=TA_CENTER,
        fontName='Helvetica'
    ))

    styles.add(ParagraphStyle(
        name='MetricValue',
        parent=styles['Normal'],
        fontSize=24,
        textColor=NAVY,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    ))

    styles.add(ParagraphStyle(
        name='TableHeader',
        parent=styles['Normal'],
        fontSize=9,
        textColor=GRAY_600,
        alignment=TA_LEFT,
        fontName='Helvetica-Bold'
    ))

    styles.add(ParagraphStyle(
        name='TableCell',
        parent=styles['Normal'],
        fontSize=9,
        textColor=GRAY_700,
        alignment=TA_LEFT,
        fontName='Helvetica'
    ))

    styles.add(ParagraphStyle(
        name='FooterText',
        parent=styles['Normal'],
        fontSize=8,
        textColor=GRAY_500,
        alignment=TA_CENTER,
        fontName='Helvetica'
    ))

    styles.add(ParagraphStyle(
        name='Confidential',
        parent=styles['Normal'],
        fontSize=10,
        textColor=GRAY_500,
        alignment=TA_CENTER,
        fontName='Helvetica-Oblique'
    ))

    return styles


class NumberedCanvas(canvas.Canvas):
    """Custom canvas for page numbers and headers."""

    def __init__(self, *args, **kwargs):
        self._saved_page_states = []
        self.page_count = 0
        canvas.Canvas.__init__(self, *args, **kwargs)

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()
        self.page_count += 1

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_number(num_pages)
            canvas.Canvas.showPage(self)
        canvas.Canvas.save(self)

    def draw_page_number(self, page_count):
        self.setFont("Helvetica", 8)
        self.setFillColor(GRAY_500)
        page_num = self._pageNumber

        # Skip page number on cover page
        if page_num > 1:
            self.drawRightString(
                letter[0] - 0.75*inch,
                0.5*inch,
                f"Page {page_num} of {page_count}"
            )
            self.drawString(
                0.75*inch,
                0.5*inch,
                "AP Guardian - Confidential"
            )


def format_currency(amount: float, currency: str = "USD") -> str:
    """Format currency values."""
    if currency == "USD":
        return f"${amount:,.2f}"
    return f"{amount:,.2f} {currency}"


def format_date(date_str: str) -> str:
    """Format date strings."""
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        return dt.strftime("%B %d, %Y")
    except:
        return date_str


def get_severity_color(severity: str) -> colors.Color:
    """Get color for severity level."""
    severity_map = {
        "critical": RISK_HIGH,
        "high": RISK_HIGH,
        "medium": RISK_MEDIUM,
        "low": RISK_LOW,
    }
    return severity_map.get(severity.lower(), GRAY_500)


def create_cover_page(styles, summary: ReportSummary, story: List):
    """Create executive cover page."""
    story.append(Spacer(1, 1.5*inch))

    # Logo / Brand
    brand_table = Table(
        [[Paragraph('<font color="#2563EB">AP</font><font color="#0F172A">GUARDIAN</font>', styles['CoverTitle'])]],
        colWidths=[6.5*inch]
    )
    brand_table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(brand_table)

    story.append(Spacer(1, 0.3*inch))

    # Subtitle
    subtitle = Paragraph(
        "Accounts Payable Risk & Recovery Assessment",
        styles['CoverSubtitle']
    )
    story.append(subtitle)

    story.append(Spacer(1, 0.5*inch))

    # Line separator
    line_table = Table([['']], colWidths=[4*inch])
    line_table.setStyle(TableStyle([
        ('LINEABOVE', (0, 0), (-1, 0), 1, LIGHT_BLUE),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
    ]))
    story.append(line_table)

    story.append(Spacer(1, 0.5*inch))

    # Client and audit info
    info_data = [
        ["Client", summary.client_name],
        ["Audit Period", f"{format_date(summary.audit_period_start)} — {format_date(summary.audit_period_end)}"],
        ["Report Date", format_date(datetime.now().strftime("%Y-%m-%d"))],
        ["Prepared For", "Finance Leadership"],
    ]
    info_table = Table(info_data, colWidths=[1.5*inch, 4*inch])
    info_table.setStyle(TableStyle([
        ('TEXTCOLOR', (0, 0), (0, -1), GRAY_500),
        ('TEXTCOLOR', (1, 0), (1, -1), NAVY),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica'),
        ('FONTNAME', (1, 0), (1, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 11),
        ('ALIGN', (0, 0), (0, -1), 'RIGHT'),
        ('ALIGN', (1, 0), (1, -1), 'LEFT'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(info_table)

    story.append(Spacer(1, 0.8*inch))

    # Key Metrics on cover
    metrics_data = [
        [f"{summary.total_findings:,}", f"{format_currency(summary.total_at_risk)}", f"{format_currency(summary.total_recovered)}"],
        ["Total Findings", "At-Risk Amount", "Recovered Amount"],
    ]

    metrics_table = Table(metrics_data, colWidths=[2*inch, 2.5*inch, 2*inch])
    metrics_table.setStyle(TableStyle([
        ('TEXTCOLOR', (0, 0), (-1, 0), NAVY),
        ('TEXTCOLOR', (0, 1), (-1, 1), GRAY_500),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTNAME', (0, 1), (-1, 1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, 0), 28),
        ('FONTSIZE', (0, 1), (-1, 1), 9),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 4),
        ('TOPPADDING', (0, 1), (-1, 1), 0),
        ('BACKGROUND', (0, 0), (-1, -1), GRAY_100),
        ('BOX', (0, 0), (-1, -1), 0.5, GRAY_200),
        ('TOPPADDING', (0, 0), (-1, -1), 16),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 16),
    ]))
    story.append(metrics_table)

    story.append(Spacer(1, 1*inch))

    # Confidential notice
    confidential = Paragraph(
        "CONFIDENTIAL - For Internal Use Only",
        styles['Confidential']
    )
    story.append(confidential)

    story.append(PageBreak())


def create_executive_summary(styles, summary: ReportSummary, story: List):
    """Create executive summary page."""
    story.append(Paragraph("Executive Summary", styles['SectionHeader']))

    # Generate executive narrative
    recovery_pct = summary.recovery_rate * 100 if summary.recovery_rate else 0
    max_risk_type = max(summary.findings_by_type.items(), key=lambda x: x[1])[0] if summary.findings_by_type else "N/A"

    narrative = f"""Analysis of <b>{summary.total_transactions:,}</b> transactions identified <b>{summary.total_findings:,}</b> findings
    with an estimated exposure of <b>{format_currency(summary.total_at_risk)}</b>.

    The largest concentration of risk was associated with <b>{max_risk_type.replace('_', ' ').title()}</b> findings.

    Recovery opportunities totaling approximately <b>{format_currency(summary.total_recovered)}</b> have already been validated.
    This represents a recovery rate of <b>{recovery_pct:.1f}%</b>."""

    story.append(Paragraph(narrative, styles['ExecutiveNarrative']))

    story.append(Spacer(1, 0.3*inch))

    # Summary metrics in card format
    summary_data = [
        ["Transactions Scanned", "Total Findings", "Critical Findings", "At-Risk Amount", "Recovered"],
        [
            f"{summary.total_transactions:,}",
            f"{summary.total_findings:,}",
            f"{summary.critical_findings}",
            format_currency(summary.total_at_risk),
            format_currency(summary.total_recovered)
        ]
    ]
    summary_table = Table(summary_data, colWidths=[1.3*inch, 1.3*inch, 1.3*inch, 1.3*inch, 1.3*inch])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), NAVY),
        ('TEXTCOLOR', (0, 0), (-1, 0), WHITE),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 8),
        ('BACKGROUND', (0, 1), (-1, 1), GRAY_100),
        ('TEXTCOLOR', (0, 1), (-1, 1), NAVY),
        ('FONTNAME', (0, 1), (-1, 1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 1), (-1, 1), 12),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BOX', (0, 0), (-1, -1), 0.5, GRAY_300),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, GRAY_200),
        ('TOPPADDING', (0, 0), (-1, -1), 12),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
    ]))
    story.append(summary_table)

    story.append(Spacer(1, 0.4*inch))

    # Key Insights
    story.append(Paragraph("Key Observations", styles['SubsectionHeader']))

    # Top insights based on data
    findings_pct = (summary.total_findings / summary.total_transactions * 100) if summary.total_transactions > 0 else 0
    critical_pct = (summary.critical_findings / summary.total_findings * 100) if summary.total_findings > 0 else 0

    insights = [
        f"Finding rate of {findings_pct:.2f}% across all scanned transactions",
        f"{summary.critical_findings} critical findings identified ({critical_pct:.1f}% of total)",
        f"Potential recovery value of {format_currency(summary.total_at_risk - summary.total_recovered)} remains outstanding",
    ]

    for i, insight in enumerate(insights, 1):
        story.append(Paragraph(f"<b>{i}.</b> {insight}", styles['CustomBody']))

    story.append(PageBreak())


def create_risk_dashboard(styles, summary: ReportSummary, story: List):
    """Create risk dashboard page with KPI cards."""
    story.append(Paragraph("Risk Dashboard", styles['SectionHeader']))

    story.append(Spacer(1, 0.2*inch))

    # KPI Cards - Row 1
    kpi_row1 = [
        ["Total Findings", "Critical", "At-Risk Amount"],
        [
            f"{summary.total_findings:,}",
            f"{summary.critical_findings}",
            format_currency(summary.total_at_risk)
        ]
    ]

    kpi_table1 = Table(kpi_row1, colWidths=[2.1*inch, 2.1*inch, 2.1*inch])
    kpi_table1.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), GRAY_200),
        ('TEXTCOLOR', (0, 0), (-1, 0), GRAY_600),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, 0), 9),
        ('BACKGROUND', (0, 1), (-1, 1), WHITE),
        ('TEXTCOLOR', (0, 1), (0, 1), NAVY),
        ('TEXTCOLOR', (1, 1), (1, 1), RISK_HIGH),
        ('TEXTCOLOR', (2, 1), (2, 1), NAVY),
        ('FONTNAME', (0, 1), (-1, 1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 1), (-1, 1), 20),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('BOX', (0, 0), (0, -1), 0.5, GRAY_300),
        ('BOX', (1, 0), (1, -1), 0.5, GRAY_300),
        ('BOX', (2, 0), (2, -1), 0.5, GRAY_300),
        ('TOPPADDING', (0, 0), (-1, -1), 14),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 14),
    ]))
    story.append(kpi_table1)

    story.append(Spacer(1, 0.3*inch))

    # KPI Cards - Row 2
    kpi_row2 = [
        ["Recovered Amount", "Recovery Rate", "Outstanding"],
        [
            format_currency(summary.total_recovered),
            f"{summary.recovery_rate*100:.1f}%",
            format_currency(summary.total_at_risk - summary.total_recovered)
        ]
    ]

    kpi_table2 = Table(kpi_row2, colWidths=[2.1*inch, 2.1*inch, 2.1*inch])
    kpi_table2.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), GRAY_200),
        ('TEXTCOLOR', (0, 0), (-1, 0), GRAY_600),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, 0), 9),
        ('BACKGROUND', (0, 1), (-1, 1), WHITE),
        ('TEXTCOLOR', (0, 1), (0, 1), RECOVERED),
        ('TEXTCOLOR', (1, 1), (1, 1), RECOVERED),
        ('TEXTCOLOR', (2, 1), (2, 1), NAVY),
        ('FONTNAME', (0, 1), (-1, 1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 1), (-1, 1), 20),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('BOX', (0, 0), (0, -1), 0.5, GRAY_300),
        ('BOX', (1, 0), (1, -1), 0.5, GRAY_300),
        ('BOX', (2, 0), (2, -1), 0.5, GRAY_300),
        ('TOPPADDING', (0, 0), (-1, -1), 14),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 14),
    ]))
    story.append(kpi_table2)

    story.append(Spacer(1, 0.4*inch))

    # Findings Breakdown
    story.append(Paragraph("Findings by Severity", styles['SubsectionHeader']))

    severity_data = [
        ["Severity", "Count", "Percentage"],
        ["Critical", str(summary.critical_findings), f"{(summary.critical_findings/summary.total_findings*100) if summary.total_findings > 0 else 0:.1f}%"],
        ["High", str(summary.high_findings), f"{(summary.high_findings/summary.total_findings*100) if summary.total_findings > 0 else 0:.1f}%"],
        ["Medium", str(summary.medium_findings), f"{(summary.medium_findings/summary.total_findings*100) if summary.total_findings > 0 else 0:.1f}%"],
        ["Low", str(summary.low_findings), f"{(summary.low_findings/summary.total_findings*100) if summary.total_findings > 0 else 0:.1f}%"],
    ]

    severity_table = Table(severity_data, colWidths=[1.5*inch, 1.5*inch, 1.5*inch])
    severity_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), GRAY_200),
        ('TEXTCOLOR', (0, 0), (-1, 0), GRAY_700),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BACKGROUND', (0, 1), (0, 1), colors.HexColor("#FEE2E2")),
        ('BACKGROUND', (0, 2), (0, 2), colors.HexColor("#FEF3C7")),
        ('BACKGROUND', (0, 3), (0, 3), colors.HexColor("#FEF3C7")),
        ('BACKGROUND', (0, 4), (0, 4), GRAY_100),
        ('TEXTCOLOR', (0, 1), (0, 1), RISK_HIGH),
        ('TEXTCOLOR', (0, 2), (0, 2), RISK_HIGH),
        ('TEXTCOLOR', (0, 3), (0, 3), RISK_MEDIUM),
        ('TEXTCOLOR', (0, 4), (0, 4), RISK_LOW),
        ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('BOX', (0, 0), (-1, -1), 0.5, GRAY_300),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, GRAY_200),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(severity_table)

    story.append(PageBreak())


def create_findings_analysis(styles, summary: ReportSummary, story: List):
    """Create findings analysis page with charts."""
    story.append(Paragraph("Findings Analysis", styles['SectionHeader']))

    story.append(Spacer(1, 0.2*inch))

    # Findings by Type Table
    story.append(Paragraph("Findings by Type", styles['SubsectionHeader']))

    findings_by_type_data = [["Finding Type", "Count", "Percentage", "At-Risk Amount"]]
    total = sum(summary.findings_by_type.values()) if summary.findings_by_type else 0

    for finding_type, count in sorted(summary.findings_by_type.items(), key=lambda x: x[1], reverse=True):
        pct = (count / total * 100) if total > 0 else 0
        amount = summary.total_at_risk * (count / total) if total > 0 else 0
        findings_by_type_data.append([
            finding_type.replace('_', ' ').title(),
            str(count),
            f"{pct:.1f}%",
            format_currency(amount)
        ])

    findings_table = Table(findings_by_type_data, colWidths=[2.5*inch, 1*inch, 1*inch, 1.5*inch])
    findings_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), NAVY),
        ('TEXTCOLOR', (0, 0), (-1, 0), WHITE),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 9),
        ('BACKGROUND', (0, 1), (-1, -1), WHITE),
        ('TEXTCOLOR', (0, 1), (-1, -1), GRAY_700),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
        ('ALIGN', (0, 0), (0, -1), 'LEFT'),
        ('ALIGN', (3, 0), (3, -1), 'RIGHT'),
        ('BOX', (0, 0), (-1, -1), 0.5, GRAY_300),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, GRAY_200),
        ('TOPPADDING', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [WHITE, GRAY_100]),
    ]))
    story.append(findings_table)

    story.append(Spacer(1, 0.5*inch))

    # Create bar chart for findings by type
    story.append(Paragraph("Findings Distribution", styles['SubsectionHeader']))

    drawing = Drawing(500, 200)

    # Add bar chart
    bc = VerticalBarChart()
    bc.x = 50
    bc.y = 30
    bc.height = 150
    bc.width = 400

    data = list(summary.findings_by_type.values()) if summary.findings_by_type else [0]
    bc.data = [data]

    bc.strokeColor = colors.black
    bc.valueAxis.valueMin = 0
    bc.valueAxis.valueMax = max(data) * 1.2 if data else 10
    bc.valueAxis.valueStep = max(data) // 5 if data and max(data) > 5 else 1
    bc.categoryAxis.labels.boxAnchor = 'ne'
    bc.categoryAxis.labels.dx = -5
    bc.categoryAxis.labels.dy = -2
    bc.categoryAxis.labels.angle = 30
    bc.categoryAxis.labels.fontName = 'Helvetica'
    bc.categoryAxis.labels.fontSize = 8
    bc.categoryAxis.categoryNames = [
        ft.replace('_', '\n')[:15] for ft in summary.findings_by_type.keys()
    ] if summary.findings_by_type else []

    bc.bars[0].fillColor = ACCENT_BLUE
    bc.bars[0].strokeColor = NAVY

    drawing.add(bc)

    story.append(drawing)

    story.append(PageBreak())


def create_vendor_risk_scorecard(styles, summary: ReportSummary, story: List):
    """Create vendor risk scorecard page."""
    story.append(Paragraph("Vendor Risk Scorecard", styles['SectionHeader']))

    story.append(Paragraph("Top 10 Vendors by Risk Score", styles['SubsectionHeader']))

    # Sort vendors by risk score
    top_vendors = sorted(summary.vendors, key=lambda v: v.risk_score, reverse=True)[:10]

    vendor_data = [["Vendor", "Risk Score", "Findings", "At-Risk", "Recovered", "Outstanding"]]

    for vendor in top_vendors:
        vendor_data.append([
            vendor.vendor_name[:25],
            str(vendor.risk_score),
            str(vendor.total_findings),
            format_currency(vendor.at_risk_amount),
            format_currency(vendor.recovered_amount),
            format_currency(vendor.outstanding_amount)
        ])

    vendor_table = Table(vendor_data, colWidths=[1.5*inch, 0.9*inch, 0.8*inch, 1*inch, 1*inch, 1*inch])
    vendor_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), NAVY),
        ('TEXTCOLOR', (0, 0), (-1, 0), WHITE),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 8),
        ('BACKGROUND', (0, 1), (-1, -1), WHITE),
        ('TEXTCOLOR', (0, 1), (-1, -1), GRAY_700),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 8),
        ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
        ('ALIGN', (0, 0), (0, -1), 'LEFT'),
        ('ALIGN', (3, 0), (-1, -1), 'RIGHT'),
        ('BOX', (0, 0), (-1, -1), 0.5, GRAY_300),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, GRAY_200),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [WHITE, GRAY_100]),
    ]))

    # Add risk score color coding
    for i, vendor in enumerate(top_vendors, 1):
        if vendor.risk_score >= 70:
            vendor_table.setStyle(TableStyle([
                ('BACKGROUND', (1, i), (1, i), colors.HexColor("#FEE2E2")),
                ('TEXTCOLOR', (1, i), (1, i), RISK_HIGH),
            ]))
        elif vendor.risk_score >= 40:
            vendor_table.setStyle(TableStyle([
                ('BACKGROUND', (1, i), (1, i), colors.HexColor("#FEF3C7")),
                ('TEXTCOLOR', (1, i), (1, i), RISK_MEDIUM),
            ]))
        else:
            vendor_table.setStyle(TableStyle([
                ('BACKGROUND', (1, i), (1, i), GRAY_100),
                ('TEXTCOLOR', (1, i), (1, i), RISK_LOW),
            ]))

    story.append(vendor_table)

    story.append(Spacer(1, 0.3*inch))

    # Risk Score Legend
    legend_data = [
        ["Risk Score Legend", "", ""],
        ["0-39: Low Risk", "40-69: Medium Risk", "70-100: High Risk"],
    ]
    legend_table = Table(legend_data, colWidths=[2*inch, 2*inch, 2*inch])
    legend_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), GRAY_200),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('TEXTCOLOR', (0, 1), (0, 1), RISK_LOW),
        ('TEXTCOLOR', (1, 1), (1, 1), RISK_MEDIUM),
        ('TEXTCOLOR', (2, 1), (2, 1), RISK_HIGH),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('BOX', (0, 0), (-1, -1), 0.5, GRAY_300),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(legend_table)

    story.append(PageBreak())


def create_high_priority_findings(styles, summary: ReportSummary, story: List):
    """Create high priority findings page."""
    story.append(Paragraph("High Priority Findings", styles['SectionHeader']))

    # Filter critical and high findings
    high_priority = [f for f in summary.findings if f.severity.lower() in ['critical', 'high']][:15]

    if not high_priority:
        story.append(Paragraph("No critical or high severity findings identified.", styles['CustomBody']))
        story.append(PageBreak())
        return

    story.append(Paragraph(f"Showing {len(high_priority)} Critical and High Severity Findings", styles['SubsectionHeader']))

    findings_data = [["ID", "Vendor", "Type", "Severity", "Amount", "Action"]]

    for finding in high_priority:
        findings_data.append([
            finding.id[:8],
            finding.vendor_name[:20],
            finding.finding_type.replace('_', ' ')[:15],
            finding.severity.upper(),
            format_currency(finding.amount),
            finding.recommended_action[:25] + "..." if len(finding.recommended_action) > 25 else finding.recommended_action
        ])

    findings_table = Table(findings_data, colWidths=[0.6*inch, 1.3*inch, 1*inch, 0.8*inch, 1*inch, 1.5*inch])
    findings_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), NAVY),
        ('TEXTCOLOR', (0, 0), (-1, 0), WHITE),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 8),
        ('FONTSIZE', (0, 1), (-1, -1), 8),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('ALIGN', (4, 0), (4, -1), 'RIGHT'),
        ('BOX', (0, 0), (-1, -1), 0.5, GRAY_300),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, GRAY_200),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [WHITE, GRAY_100]),
    ]))

    # Color code severity column
    for i, finding in enumerate(high_priority, 1):
        if finding.severity.lower() == 'critical':
            findings_table.setStyle(TableStyle([
                ('BACKGROUND', (3, i), (3, i), colors.HexColor("#FEE2E2")),
                ('TEXTCOLOR', (3, i), (3, i), RISK_HIGH),
                ('FONTNAME', (3, i), (3, i), 'Helvetica-Bold'),
            ]))
        elif finding.severity.lower() == 'high':
            findings_table.setStyle(TableStyle([
                ('BACKGROUND', (3, i), (3, i), colors.HexColor("#FEF3C7")),
                ('TEXTCOLOR', (3, i), (3, i), RISK_HIGH),
                ('FONTNAME', (3, i), (3, i), 'Helvetica-Bold'),
            ]))

    story.append(findings_table)

    story.append(PageBreak())


def create_recovery_section(styles, summary: ReportSummary, story: List):
    """Create recovery opportunities section."""
    story.append(Paragraph("Recovery Opportunities", styles['SectionHeader']))

    # Recovery Summary Card
    story.append(Paragraph("Recovery Summary", styles['SubsectionHeader']))

    recovery_summary = [
        ["Total At-Risk Amount", "Total Recovered", "Outstanding", "Recovery Rate"],
        [
            format_currency(summary.total_at_risk),
            format_currency(summary.total_recovered),
            format_currency(summary.total_at_risk - summary.total_recovered),
            f"{summary.recovery_rate*100:.1f}%"
        ]
    ]

    recovery_table = Table(recovery_summary, colWidths=[1.5*inch, 1.5*inch, 1.5*inch, 1.5*inch])
    recovery_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), GRAY_200),
        ('TEXTCOLOR', (0, 0), (-1, 0), GRAY_600),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, 0), 9),
        ('BACKGROUND', (0, 1), (-1, 1), WHITE),
        ('TEXTCOLOR', (0, 1), (0, 1), NAVY),
        ('TEXTCOLOR', (1, 1), (1, 1), RECOVERED),
        ('TEXTCOLOR', (2, 1), (2, 1), NAVY),
        ('TEXTCOLOR', (3, 1), (3, 1), RECOVERED),
        ('FONTNAME', (0, 1), (-1, 1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 1), (-1, 1), 16),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('BOX', (0, 0), (-1, -1), 0.5, GRAY_300),
        ('TOPPADDING', (0, 0), (-1, -1), 14),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 14),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, GRAY_200),
    ]))
    story.append(recovery_table)

    story.append(Spacer(1, 0.3*inch))

    # Recovery narrative
    narrative = f"""Estimated recoverable value of <b>{format_currency(summary.total_recovered)}</b> has been identified through
    duplicate payment analysis and control exceptions. An additional <b>{format_currency(summary.total_at_risk - summary.total_recovered)}</b>
    in potential recoveries is pending validation."""

    story.append(Paragraph(narrative, styles['ExecutiveNarrative']))

    story.append(Spacer(1, 0.3*inch))

    # Vendor Recovery Table
    story.append(Paragraph("Recovery by Vendor", styles['SubsectionHeader']))

    vendor_recovery = sorted(summary.vendors, key=lambda v: v.outstanding_amount, reverse=True)[:10]

    recovery_data = [["Vendor", "Potential Recovery", "Status", "Recommended Action", "Est. Value"]]
    for vendor in vendor_recovery:
        status = "Validated" if vendor.recovered_amount > 0 else "Pending"
        recovery_data.append([
            vendor.vendor_name[:20],
            format_currency(vendor.outstanding_amount + vendor.recovered_amount),
            status,
            "Review & validate" if status == "Pending" else "Process refund",
            format_currency(vendor.recovered_amount)
        ])

    recovery_vendor_table = Table(recovery_data, colWidths=[1.5*inch, 1.2*inch, 0.9*inch, 1.3*inch, 1*inch])
    recovery_vendor_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), NAVY),
        ('TEXTCOLOR', (0, 0), (-1, 0), WHITE),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 8),
        ('FONTSIZE', (0, 1), (-1, -1), 8),
        ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
        ('ALIGN', (4, 0), (4, -1), 'RIGHT'),
        ('BOX', (0, 0), (-1, -1), 0.5, GRAY_300),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, GRAY_200),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [WHITE, GRAY_100]),
    ]))

    story.append(recovery_vendor_table)

    story.append(PageBreak())


def create_recommendations(styles, summary: ReportSummary, story: List):
    """Create management recommendations page."""
    story.append(Paragraph("Management Recommendations", styles['SectionHeader']))

    story.append(Paragraph(
        "Based on the findings from this audit period, the following actions are recommended:",
        styles['CustomBody']
    ))

    story.append(Spacer(1, 0.2*inch))

    recommendations = [
        {
            "title": "Strengthen Duplicate Payment Controls",
            "description": "Implement three-way matching for all invoices above $10,000. Configure systematic duplicate payment detection rules in the ERP system."
        },
        {
            "title": "Improve Vendor Master Governance",
            "description": "Establish quarterly vendor master file reviews. Implement segregation of duties for vendor creation and modification. Add bank account validation for new vendors."
        },
        {
            "title": "Enforce Approval Threshold Monitoring",
            "description": "Configure real-time alerts for transactions approaching approval limits. Review and update authorization matrices annually. Audit split transactions that bypass thresholds."
        },
        {
            "title": "Increase Invoice Matching Controls",
            "description": "Expand PO matching requirements. Implement receiving report validation for high-value purchases. Add unit price tolerance checks."
        },
        {
            "title": "Implement Periodic AP Reviews",
            "description": "Conduct monthly reviews of high-risk vendors. Perform quarterly analytics on payment patterns. Schedule annual AP process assessment with external auditors."
        },
    ]

    for i, rec in enumerate(recommendations, 1):
        story.append(Paragraph(f"<b>{i}. {rec['title']}</b>", styles['CustomBody']))
        story.append(Paragraph(rec['description'], styles['CustomBody']))
        story.append(Spacer(1, 0.15*inch))

    story.append(PageBreak())


def create_appendix(styles, summary: ReportSummary, story: List):
    """Create appendix with detailed findings."""
    story.append(Paragraph("Appendix: Detailed Findings", styles['SectionHeader']))

    story.append(Paragraph(
        f"Complete listing of all {len(summary.findings)} findings identified during the audit period.",
        styles['CustomBody']
    ))

    story.append(Spacer(1, 0.2*inch))

    # Paginate findings (25 per page)
    findings_per_page = 25
    total_findings = len(summary.findings)

    for start_idx in range(0, total_findings, findings_per_page):
        end_idx = min(start_idx + findings_per_page, total_findings)
        page_findings = summary.findings[start_idx:end_idx]

        findings_data = [["ID", "Vendor", "Date", "Type", "Severity", "Amount", "Status"]]
        for finding in page_findings:
            findings_data.append([
                finding.id[:8],
                finding.vendor_name[:15],
                finding.transaction_date,
                finding.finding_type.replace('_', ' ')[:12],
                finding.severity[:3].upper(),
                format_currency(finding.amount),
                finding.status[:8]
            ])

        findings_table = Table(findings_data, colWidths=[0.6*inch, 1.2*inch, 0.8*inch, 1*inch, 0.6*inch, 0.9*inch, 0.7*inch])
        findings_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), NAVY),
            ('TEXTCOLOR', (0, 0), (-1, 0), WHITE),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 7),
            ('FONTSIZE', (0, 1), (-1, -1), 7),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('ALIGN', (5, 0), (5, -1), 'RIGHT'),
            ('BOX', (0, 0), (-1, -1), 0.5, GRAY_300),
            ('INNERGRID', (0, 0), (-1, -1), 0.25, GRAY_200),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [WHITE, GRAY_100]),
        ]))

        # Color code severity
        for i, finding in enumerate(page_findings, 1):
            severity_color = get_severity_color(finding.severity)
            findings_table.setStyle(TableStyle([
                ('TEXTCOLOR', (4, i), (4, i), severity_color),
                ('FONTNAME', (4, i), (4, i), 'Helvetica-Bold'),
            ]))

        story.append(findings_table)

        if end_idx < total_findings:
            story.append(PageBreak())


def generate_pdf_report(summary: ReportSummary) -> io.BytesIO:
    """Generate the complete PDF report."""
    buffer = io.BytesIO()
    styles = create_styles()
    story = []

    # Build report sections
    create_cover_page(styles, summary, story)
    create_executive_summary(styles, summary, story)
    create_risk_dashboard(styles, summary, story)
    create_findings_analysis(styles, summary, story)
    create_vendor_risk_scorecard(styles, summary, story)
    create_high_priority_findings(styles, summary, story)
    create_recovery_section(styles, summary, story)
    create_recommendations(styles, summary, story)
    create_appendix(styles, summary, story)

    # Build PDF
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=0.75*inch,
        leftMargin=0.75*inch,
        topMargin=0.75*inch,
        bottomMargin=0.75*inch
    )

    doc.build(story, canvasmaker=NumberedCanvas)
    buffer.seek(0)
    return buffer


def get_sample_report_data() -> ReportSummary:
    """Generate sample report data for demonstration."""
    now = datetime.now()
    start_date = (now.replace(day=1) - timedelta(days=90)).strftime("%Y-%m-%d")
    end_date = now.strftime("%Y-%m-%d")

    findings = []
    vendors = {}
    findings_by_type = {}

    finding_types = ["duplicate_payment", "threshold_anomaly", "vendor_risk", "compliance_gap", "velocity_alert"]
    severities = ["critical", "high", "medium", "low"]
    vendor_names = [
        "Acme Logistics LLC", "Global Supply Co", "Metro Services Inc",
        "Tech Solutions Ltd", "Premier Vendors Corp", "Alpha Distributors",
        "Beta Industries", "Delta Partners", "Omega Suppliers", "Zeta Corp",
        "Innovate Tech", "Nexus Systems", "Vertex Solutions", "Prism Analytics"
    ]

    # Generate 63 findings
    for i in range(63):
        vendor = vendor_names[i % len(vendor_names)]
        finding_type = finding_types[i % len(finding_types)]
        severity = severities[i % 4] if i < 30 else severities[2]  # More medium/low

        amount = round(500 + (i * 127.5) % 5000, 2)
        potential_recovery = round(amount * (0.3 + (i % 5) * 0.15), 2) if severity in ["critical", "high"] else None

        finding = Finding(
            id=f"F2026-{str(i+1).zfill(4)}",
            vendor_name=vendor,
            vendor_id=f"V{str(i % len(vendor_names) + 1).zfill(4)}",
            finding_type=finding_type,
            severity=severity,
            amount=amount,
            transaction_date=(now - timedelta(days=i*2)).strftime("%Y-%m-%d"),
            description=f"Anomaly detected in transaction processing for vendor {vendor}",
            recommended_action="Review and validate transaction details",
            status="open" if i % 3 == 0 else "under_review",
            recovery_status="validated" if i % 4 == 0 else None,
            potential_recovery=potential_recovery
        )
        findings.append(finding)

        # Track by vendor
        if vendor not in vendors:
            vendors[vendor] = {
                "risk_score": 0,
                "findings": [],
                "at_risk": 0,
                "recovered": 0
            }
        vendors[vendor]["findings"].append(finding)
        vendors[vendor]["at_risk"] += amount
        if severity == "critical":
            vendors[vendor]["risk_score"] += 20
        elif severity == "high":
            vendors[vendor]["risk_score"] += 10
        elif severity == "medium":
            vendors[vendor]["risk_score"] += 5
        else:
            vendors[vendor]["risk_score"] += 2

        if potential_recovery and i % 4 == 0:
            vendors[vendor]["recovered"] += potential_recovery

        # Track by type
        findings_by_type[finding_type] = findings_by_type.get(finding_type, 0) + 1

    # Build vendor list
    vendor_list = []
    for vendor_name, data in vendors.items():
        vendor_list.append(VendorRisk(
            vendor_name=vendor_name,
            vendor_id=data["findings"][0].vendor_id,
            risk_score=min(100, data["risk_score"]),
            total_findings=len(data["findings"]),
            at_risk_amount=round(data["at_risk"], 2),
            recovered_amount=round(data["recovered"], 2),
            outstanding_amount=round(data["at_risk"] - data["recovered"], 2)
        ))

    # Calculate totals
    total_at_risk = sum(f.amount for f in findings)
    total_recovered = sum(f.potential_recovery or 0 for f in findings if f.recovery_status == "validated")

    return ReportSummary(
        client_name="Acme Logistics Corporation",
        audit_period_start=start_date,
        audit_period_end=end_date,
        total_transactions=5247,
        total_findings=63,
        critical_findings=sum(1 for f in findings if f.severity == "critical"),
        high_findings=sum(1 for f in findings if f.severity == "high"),
        medium_findings=sum(1 for f in findings if f.severity == "medium"),
        low_findings=sum(1 for f in findings if f.severity == "low"),
        total_at_risk=round(total_at_risk, 2),
        total_recovered=round(total_recovered, 2),
        recovery_rate=total_recovered / total_at_risk if total_at_risk > 0 else 0,
        findings_by_type=findings_by_type,
        vendors=vendor_list,
        findings=findings
    )


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "AP Guardian API"}


@app.get("/api/reports/{report_id}/download")
async def download_report(report_id: str):
    """
    Generate and download PDF report.
    Returns PDF as binary stream with proper headers for browser download.
    """
    # Generate report data (in production, fetch from Supabase)
    summary = get_sample_report_data()

    # Generate PDF
    pdf_buffer = generate_pdf_report(summary)

    # Generate filename with client name and date
    month_year = datetime.now().strftime("%B_%Y")
    filename = f"AP_Guardian_{summary.client_name.replace(' ', '_')}_{month_year}.pdf"

    # Return as streaming response with download headers
    headers = {
        "Content-Disposition": f'attachment; filename="{filename}"',
        "Access-Control-Expose-Headers": "Content-Disposition"
    }

    return StreamingResponse(
        pdf_buffer,
        media_type="application/pdf",
        headers=headers
    )


@app.post("/api/reports/generate")
async def generate_report(client_name: str = "Acme Logistics Corporation"):
    """
    Generate a new report with custom parameters.
    Returns PDF as binary stream with proper headers.
    """
    summary = get_sample_report_data()
    summary.client_name = client_name

    pdf_buffer = generate_pdf_report(summary)

    month_year = datetime.now().strftime("%B_%Y")
    filename = f"AP_Guardian_{client_name.replace(' ', '_')}_{month_year}.pdf"

    headers = {
        "Content-Disposition": f'attachment; filename="{filename}"',
        "Access-Control-Expose-Headers": "Content-Disposition"
    }

    return StreamingResponse(
        pdf_buffer,
        media_type="application/pdf",
        headers=headers
    )


@app.get("/api/reports/{report_id}")
async def get_report_summary(report_id: str):
    """Get report summary data."""
    summary = get_sample_report_data()
    return {
        "report_id": report_id,
        "client_name": summary.client_name,
        "total_findings": summary.total_findings,
        "total_at_risk": summary.total_at_risk,
        "total_recovered": summary.total_recovered,
        "recovery_rate": summary.recovery_rate,
        "audit_period": {
            "start": summary.audit_period_start,
            "end": summary.audit_period_end
        }
    }


@app.get("/api/dashboard")
async def get_dashboard():
    """Get dashboard data for frontend."""
    summary = get_sample_report_data()
    return {
        "total_findings": summary.total_findings,
        "critical_findings": summary.critical_findings,
        "total_at_risk": summary.total_at_risk,
        "total_recovered": summary.total_recovered,
        "recovery_rate": summary.recovery_rate,
        "findings_by_type": summary.findings_by_type,
        "top_vendors": [
            {
                "name": v.vendor_name,
                "risk_score": v.risk_score,
                "findings": v.total_findings,
                "at_risk": v.at_risk_amount
            }
            for v in sorted(summary.vendors, key=lambda x: x.risk_score, reverse=True)[:5]
        ]
    }
