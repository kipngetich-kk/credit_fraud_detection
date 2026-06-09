-- AP Guardian Database Schema
-- Tables for storing AP audit findings, vendors, and reports

-- Vendors table
CREATE TABLE vendors (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    vendor_id VARCHAR(50) UNIQUE NOT NULL,
    vendor_name VARCHAR(255) NOT NULL,
    vendor_type VARCHAR(100),
    contact_email VARCHAR(255),
    tax_id VARCHAR(50),
    bank_account_last4 VARCHAR(4),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    is_active BOOLEAN DEFAULT TRUE
);

-- Findings table
CREATE TABLE findings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    finding_id VARCHAR(50) UNIQUE NOT NULL,
    vendor_id UUID REFERENCES vendors(id) ON DELETE SET NULL,
    vendor_name VARCHAR(255) NOT NULL,
    finding_type VARCHAR(100) NOT NULL,
    severity VARCHAR(20) NOT NULL CHECK (severity IN ('critical', 'high', 'medium', 'low')),
    amount DECIMAL(15, 2) NOT NULL,
    currency VARCHAR(3) DEFAULT 'USD',
    transaction_date DATE NOT NULL,
    description TEXT,
    recommended_action TEXT,
    status VARCHAR(50) DEFAULT 'open' CHECK (status IN ('open', 'under_review', 'validated', 'resolved', 'dismissed')),
    recovery_status VARCHAR(50),
    potential_recovery DECIMAL(15, 2),
    actual_recovery DECIMAL(15, 2),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    resolved_at TIMESTAMPTZ,
    resolved_by VARCHAR(255),
    notes TEXT
);

-- Audit reports table
CREATE TABLE reports (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    report_name VARCHAR(255) NOT NULL,
    client_name VARCHAR(255) NOT NULL,
    audit_period_start DATE NOT NULL,
    audit_period_end DATE NOT NULL,
    total_transactions INTEGER DEFAULT 0,
    total_findings INTEGER DEFAULT 0,
    critical_findings INTEGER DEFAULT 0,
    high_findings INTEGER DEFAULT 0,
    medium_findings INTEGER DEFAULT 0,
    low_findings INTEGER DEFAULT 0,
    total_at_risk DECIMAL(15, 2) DEFAULT 0,
    total_recovered DECIMAL(15, 2) DEFAULT 0,
    recovery_rate DECIMAL(5, 4) DEFAULT 0,
    findings_by_type JSONB DEFAULT '{}',
    status VARCHAR(50) DEFAULT 'generated',
    generated_at TIMESTAMPTZ DEFAULT NOW(),
    generated_by VARCHAR(255),
    pdf_path VARCHAR(500)
);

-- Transactions table (scanned transactions)
CREATE TABLE transactions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    transaction_id VARCHAR(100) UNIQUE NOT NULL,
    vendor_id UUID REFERENCES vendors(id) ON DELETE SET NULL,
    invoice_number VARCHAR(100),
    amount DECIMAL(15, 2) NOT NULL,
    currency VARCHAR(3) DEFAULT 'USD',
    transaction_date DATE NOT NULL,
    posting_date DATE,
    payment_date DATE,
    description TEXT,
    gl_account VARCHAR(50),
    cost_center VARCHAR(50),
    approval_status VARCHAR(50),
    approved_by VARCHAR(255),
    is_flagged BOOLEAN DEFAULT FALSE,
    flag_reason TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Vendor risk scores table
CREATE TABLE vendor_risk_scores (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    vendor_id UUID REFERENCES vendors(id) ON DELETE CASCADE,
    risk_score INTEGER DEFAULT 0 CHECK (risk_score >= 0 AND risk_score <= 100),
    total_findings INTEGER DEFAULT 0,
    at_risk_amount DECIMAL(15, 2) DEFAULT 0,
    recovered_amount DECIMAL(15, 2) DEFAULT 0,
    outstanding_amount DECIMAL(15, 2) DEFAULT 0,
    last_calculated_at TIMESTAMPTZ DEFAULT NOW(),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(vendor_id)
);

-- Create indexes for performance
CREATE INDEX idx_findings_vendor ON findings(vendor_id);
CREATE INDEX idx_findings_severity ON findings(severity);
CREATE INDEX idx_findings_status ON findings(status);
CREATE INDEX idx_findings_type ON findings(finding_type);
CREATE INDEX idx_findings_date ON findings(transaction_date);
CREATE INDEX idx_transactions_vendor ON transactions(vendor_id);
CREATE INDEX idx_transactions_date ON transactions(transaction_date);
CREATE INDEX idx_reports_client ON reports(client_name);
CREATE INDEX idx_reports_period ON reports(audit_period_start, audit_period_end);

-- Enable Row Level Security
ALTER TABLE vendors ENABLE ROW LEVEL SECURITY;
ALTER TABLE findings ENABLE ROW LEVEL SECURITY;
ALTER TABLE reports ENABLE ROW LEVEL SECURITY;
ALTER TABLE transactions ENABLE ROW LEVEL SECURITY;
ALTER TABLE vendor_risk_scores ENABLE ROW LEVEL SECURITY;

-- RLS Policies for vendors
CREATE POLICY "select_vendors" ON vendors FOR SELECT TO authenticated USING (true);
CREATE POLICY "insert_vendors" ON vendors FOR INSERT TO authenticated WITH CHECK (true);
CREATE POLICY "update_vendors" ON vendors FOR UPDATE TO authenticated USING (true) WITH CHECK (true);
CREATE POLICY "delete_vendors" ON vendors FOR DELETE TO authenticated USING (true);

-- RLS Policies for findings
CREATE POLICY "select_findings" ON findings FOR SELECT TO authenticated USING (true);
CREATE POLICY "insert_findings" ON findings FOR INSERT TO authenticated WITH CHECK (true);
CREATE POLICY "update_findings" ON findings FOR UPDATE TO authenticated USING (true) WITH CHECK (true);
CREATE POLICY "delete_findings" ON findings FOR DELETE TO authenticated USING (true);

-- RLS Policies for reports
CREATE POLICY "select_reports" ON reports FOR SELECT TO authenticated USING (true);
CREATE POLICY "insert_reports" ON reports FOR INSERT TO authenticated WITH CHECK (true);
CREATE POLICY "update_reports" ON reports FOR UPDATE TO authenticated USING (true) WITH CHECK (true);
CREATE POLICY "delete_reports" ON reports FOR DELETE TO authenticated USING (true);

-- RLS Policies for transactions
CREATE POLICY "select_transactions" ON transactions FOR SELECT TO authenticated USING (true);
CREATE POLICY "insert_transactions" ON transactions FOR INSERT TO authenticated WITH CHECK (true);
CREATE POLICY "update_transactions" ON transactions FOR UPDATE TO authenticated USING (true) WITH CHECK (true);
CREATE POLICY "delete_transactions" ON transactions FOR DELETE TO authenticated USING (true);

-- RLS Policies for vendor_risk_scores
CREATE POLICY "select_risk_scores" ON vendor_risk_scores FOR SELECT TO authenticated USING (true);
CREATE POLICY "insert_risk_scores" ON vendor_risk_scores FOR INSERT TO authenticated WITH CHECK (true);
CREATE POLICY "update_risk_scores" ON vendor_risk_scores FOR UPDATE TO authenticated USING (true) WITH CHECK (true);
CREATE POLICY "delete_risk_scores" ON vendor_risk_scores FOR DELETE TO authenticated USING (true);

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Triggers for updated_at
CREATE TRIGGER update_vendors_updated_at
    BEFORE UPDATE ON vendors
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_findings_updated_at
    BEFORE UPDATE ON findings
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_transactions_updated_at
    BEFORE UPDATE ON transactions
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_vendor_risk_scores_updated_at
    BEFORE UPDATE ON vendor_risk_scores
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();
