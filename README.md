# AP Guardian

**Accounts Payable Risk & Recovery Assessment Platform**

A professional-grade fintech tool for detecting AP anomalies, duplicate payments, and control exceptions with enterprise-ready PDF audit reporting.

---

## Features

- **Risk Dashboard** - Real-time KPIs for at-risk amounts, findings count, and recovery rates
- **Findings Analysis** - Categorize by severity (Critical, High, Medium, Low) and type
- **Vendor Risk Scorecard** - Risk scoring (0-100) based on finding patterns
- **Executive PDF Reports** - Professional audit documents formatted for CFO/management review
- **Blob-Based Downloads** - Direct browser download without URL redirects

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| Frontend | React + TypeScript + Vite + Tailwind CSS |
| Backend | Python + FastAPI + ReportLab |
| Database | Supabase (PostgreSQL with RLS) |

---

## Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+

### Backend Setup

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```

### Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

### Environment Variables

Create a `.env` file (for production database):

```
SUPABASE_URL=your_supabase_url
SUPABASE_ANON_KEY=your_anon_key
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Health check |
| GET | `/api/dashboard` | Dashboard metrics |
| GET | `/api/reports/{id}/download` | Download PDF report |
| POST | `/api/reports/generate` | Generate custom report |

---

## PDF Report Structure

Generated reports include:

1. Executive Cover Page
2. Executive Summary
3. Risk Dashboard
4. Findings Analysis
5. Vendor Risk Scorecard
6. High Priority Findings
7. Recovery Opportunities
8. Management Recommendations
9. Appendix (Detailed Findings)

Design follows consulting firm standards (Deloitte, PwC, EY styling principles).

---

## Project Structure

```
ap-guardian/
├── backend/
│   ├── main.py              # FastAPI app + PDF generation
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── App.tsx          # Main React component
│   │   ├── main.tsx
│   │   └── index.css
│   ├── package.json
│   └── vite.config.ts
├── supabase/
│   └── migrations/          # Database schema
└── README.md
```

---

## Development

The sample data generator creates realistic test data with:

- 63 findings across 14 vendors
- 5 finding types
- Severity distribution and risk scoring
- Recovery calculations

---

## License

MIT
