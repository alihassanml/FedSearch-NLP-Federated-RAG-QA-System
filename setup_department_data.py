import os
import shutil

def create_department_data():
    """Create separate data directories for each department"""
    
    departments = {
        'hr': {
            'hr_policy.txt': '''ACME Corporation - Human Resources Policy Manual

1. LEAVE POLICY
- Annual Leave: 20 days per year for full-time employees
- Sick Leave: 10 days per year with medical certificate required after 3 consecutive days
- Parental Leave: 12 weeks paid leave for primary caregivers
- Public Holidays: All national holidays are observed

2. WORKING HOURS
- Standard working hours: Monday to Friday, 9:00 AM to 5:00 PM
- Flexible working arrangements available after 6 months of employment
- Remote work policy: Up to 3 days per week for eligible positions

3. COMPENSATION AND BENEFITS
- Salary reviews conducted annually in January
- Performance bonuses up to 20% of annual salary
- Health insurance coverage for employee and immediate family
- Retirement contribution: Company matches up to 5%

4. TRAINING AND DEVELOPMENT
- Annual training budget: $2000 per employee
- Conference attendance supported
- Internal mentorship program available
- Leadership development programs

5. EMPLOYEE CONDUCT
- Professional behavior expected at all times
- Zero tolerance for harassment or discrimination
- Confidentiality agreements must be maintained
- Social media policy guidelines''',
            
            'benefits_guide.txt': '''Employee Benefits Guide

HEALTH INSURANCE
- Medical coverage starts day 1
- Dental and vision included
- Family coverage available
- $500 annual deductible

RETIREMENT PLANS
- 401(k) with 5% company match
- Vesting schedule: 3 years
- Financial planning services included

WELLNESS PROGRAMS
- Gym membership reimbursement: $50/month
- Mental health support services
- Annual health screenings
- Ergonomic workspace assessments

TIME OFF
- Vacation: 20 days
- Sick leave: 10 days
- Personal days: 5 days
- Holidays: 12 days'''
        },
        
        'it': {
            'it_sop.txt': '''ACME Corporation - IT Standard Operating Procedures

1. SYSTEM ACCESS
- New employee access provisioned within 24 hours
- Default access: Email, intranet, project tools
- Special access requires manager approval
- Access reviews conducted quarterly

2. PASSWORD POLICY
- Minimum 12 characters required
- Must include uppercase, lowercase, numbers, symbols
- Password expiry: Every 90 days
- No password reuse for last 5 passwords
- Multi-factor authentication (MFA) mandatory

3. DATA BACKUP
- Automated daily backups at 2:00 AM
- Weekly full backups on Sundays
- Monthly archives stored off-site
- Backup retention: Daily (7 days), Weekly (4 weeks), Monthly (12 months)

4. INCIDENT REPORTING
- Security incidents reported immediately to security@acme.com
- IT helpdesk: extension 5555, helpdesk@acme.com
- Priority levels: Critical (1h), High (4h), Medium (1d), Low (3d)

5. SOFTWARE MANAGEMENT
- Only approved software from company catalog
- Request new software through IT portal
- Personal software prohibited on company devices

6. NETWORK SECURITY
- VPN required for remote access
- Guest WiFi available for visitors
- USB drives must be encrypted
- Email attachments scanned automatically''',
            
            'security_guidelines.txt': '''IT Security Guidelines

DEVICE SECURITY
- Screen lock after 5 minutes
- Full disk encryption required
- Antivirus must be updated
- Lost devices reported within 2 hours

REMOTE WORK
- Use company VPN only
- Secure home WiFi networks
- No public WiFi for sensitive data
- Virtual desktop for high-security work

DATA HANDLING
- Sensitive data encryption required
- No personal cloud storage
- Secure file sharing via company tools
- Data classification: Public, Internal, Confidential, Secret'''
        },
        
        'legal': {
            'legal_doc.txt': '''ACME Corporation - Legal Guidelines and Compliance

1. INTELLECTUAL PROPERTY
- All work products belong to ACME Corporation
- Patent disclosures required for inventions
- Copyright: Company owns all created works
- Non-compete: 12 months post-employment

2. DATA PROTECTION AND PRIVACY
- GDPR compliance mandatory for EU data
- Personal data collection requires consent
- Data retention: 7 years after last transaction
- Right to deletion: 30-day processing
- Breach notification: 72 hours to authorities

3. CONTRACT MANAGEMENT
- Contracts over $10,000 require legal review
- Standard vendor contracts in legal portal
- Signing authority: Directors and above
- Purchase orders valid for 90 days

4. REGULATORY COMPLIANCE
- Quarterly and annual audits mandatory
- ISO 27001, SOC 2 Type II maintained
- Export control compliance required
- Anti-bribery: No gifts over $50 value

5. LITIGATION PROCEDURES
- Document preservation if litigation anticipated
- Pre-approved external counsel list
- Settlement authority: CFO for amounts over $50,000''',
            
            'compliance_checklist.txt': '''Compliance Requirements Checklist

ANNUAL REQUIREMENTS
□ Financial audit (Q4)
□ SOC 2 certification renewal
□ ISO 27001 audit
□ GDPR data protection review
□ Export control training
□ Anti-bribery certification

QUARTERLY REQUIREMENTS
□ Access control reviews
□ Vendor contract audits
□ Data retention compliance
□ Policy updates review

MONTHLY REQUIREMENTS
□ Security incident reports
□ Compliance training completion
□ Risk assessment updates'''
        }
    }
    
    print("Creating department data directories...")
    
    for dept, files in departments.items():
        # Create department directory
        dept_dir = f"data/department_{dept}"
        os.makedirs(dept_dir, exist_ok=True)
        
        # Create files
        for filename, content in files.items():
            filepath = os.path.join(dept_dir, filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✓ Created: {filepath}")
    
    print("\n✅ Department data setup complete!")
    print("\nCreated directories:")
    print("  - data/department_hr/")
    print("  - data/department_it/")
    print("  - data/department_legal/")
    print("\nEach department has 2-3 private documents")


if __name__ == "__main__":
    create_department_data()