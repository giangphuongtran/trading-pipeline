# Data Science/Engineering Project Documentation Template

This template provides a comprehensive structure for documenting data science and engineering projects. Use this as a guide to ensure your projects are well-documented, maintainable, and reproducible.

---

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Setup and Installation](#2-setup-and-installation)
3. [Architecture and Design](#3-architecture-and-design)
4. [Data Documentation](#4-data-documentation)
5. [Code Documentation](#5-code-documentation)
6. [Bug Tracking](#6-bug-tracking)
7. [Testing](#7-testing)
8. [Deployment](#8-deployment)
9. [Maintenance](#9-maintenance)
10. [Lessons Learned](#10-lessons-learned)

---

## 1. Project Overview

### 1.1 Purpose
**What problem does this project solve?**
- Clear statement of the business/technical problem
- Success criteria and KPIs

### 1.2 Scope
**What's included and what's not?**
- In-scope features
- Out-of-scope items (future work)
- Assumptions and constraints

### 1.3 Key Stakeholders
- Product owners
- Data scientists/engineers
- End users
- Operations team

### 1.4 Timeline
- Project start date
- Key milestones
- Current status

---

## 2. Setup and Installation

### 2.1 Prerequisites
- Python version
- Required system packages
- External services (databases, APIs, etc.)
- API keys and credentials needed

### 2.2 Installation Steps
```bash
# Step-by-step commands
# Include exact versions when critical
```

### 2.3 Environment Setup
- Virtual environment creation
- `.env` file configuration
- Docker setup (if applicable)
- Database initialization

### 2.4 Verification
- How to verify installation worked
- Test commands to run
- Expected outputs

---

## 3. Architecture and Design

### 3.1 System Architecture
```
[Diagram or description of system components]
```

### 3.2 Data Flow
- Where data comes from
- How it's processed
- Where it's stored
- How it's consumed

### 3.3 Key Design Decisions
- Why certain technologies were chosen
- Trade-offs considered
- Alternatives evaluated

### 3.4 Directory Structure
```
project/
├── app/              # Application code
├── ml/               # Machine learning code
├── db/               # Database scripts
├── tests/            # Test files
├── docs/             # Documentation
└── ...
```

---

## 4. Data Documentation

### 4.1 Data Sources
- External APIs
- Databases
- File systems
- Third-party services

### 4.2 Data Schema
**For each table/dataset:**
- Table name
- Column names and types
- Primary/foreign keys
- Indexes
- Sample data

### 4.3 Data Quality
- Known data quality issues
- Missing data patterns
- Data validation rules
- Data freshness requirements

### 4.4 Data Dependencies
- Upstream dependencies
- Downstream consumers
- Data lineage

### 4.5 Known Issues and Limitations
- Holiday gaps (for time series)
- API rate limits
- Data availability windows
- Known bugs in source data

---

## 5. Code Documentation

### 5.1 Module Overview
**For each major module:**
- Purpose
- Key functions/classes
- Dependencies
- Usage examples

### 5.2 API Documentation
- Function signatures
- Parameters and return types
- Example usage
- Error handling

### 5.3 Configuration
- Configuration files
- Environment variables
- Default values
- How to override

### 5.4 Common Patterns
- Code patterns used throughout
- Best practices followed
- Conventions (naming, structure)

---

## 6. Bug Tracking

### 6.1 Bug Log Template
For each bug, document:

**Bug ID**: [Unique identifier]
**Date Discovered**: [YYYY-MM-DD]
**Date Fixed**: [YYYY-MM-DD]
**Severity**: [Critical/High/Medium/Low]
**Status**: [Open/In Progress/Fixed/Closed/Won't Fix]

**Description**:
- What the bug is
- How it manifests
- Steps to reproduce

**Impact**:
- Who/what is affected
- Business impact
- Workarounds available

**Root Cause**:
- Why it happened
- Underlying issue

**Solution**:
- How it was fixed
- Code changes made
- Files modified

**Testing**:
- How to verify the fix
- Test cases added

**Prevention**:
- How to prevent similar bugs
- Process improvements

### 6.2 Known Issues
- Current open bugs
- Limitations
- Technical debt

---

## 7. Testing

### 7.1 Test Strategy
- Unit tests
- Integration tests
- End-to-end tests
- Data quality tests

### 7.2 Test Coverage
- Current coverage percentage
- Critical paths tested
- Gaps in coverage

### 7.3 Running Tests
```bash
# Commands to run tests
```

### 7.4 Test Data
- Where test data comes from
- How to generate test data
- Test data refresh process

---

## 8. Deployment

### 8.1 Deployment Process
- Steps to deploy
- Environments (dev/staging/prod)
- Rollback procedures

### 8.2 Monitoring
- What to monitor
- Alerting rules
- Dashboards

### 8.3 Troubleshooting
- Common issues
- How to debug
- Log locations

---

## 9. Maintenance

### 9.1 Regular Tasks
- Data refresh schedules
- Model retraining
- Database maintenance
- Dependency updates

### 9.2 Change Log
**Version X.X.X - YYYY-MM-DD**
- Added: [new features]
- Changed: [modifications]
- Fixed: [bug fixes]
- Deprecated: [features to be removed]
- Removed: [deleted features]

### 9.3 Upgrade Paths
- How to upgrade from previous versions
- Breaking changes
- Migration scripts

---

## 10. Lessons Learned

### 10.1 What Went Well
- Successful patterns
- Good decisions
- Effective tools/processes

### 10.2 What Could Be Improved
- Mistakes made
- Better approaches
- Tools/processes to reconsider

### 10.3 Recommendations for Future Projects
- Best practices to carry forward
- Things to avoid
- Tools to use/avoid

---

## Additional Sections (Add as Needed)

### Performance
- Performance benchmarks
- Optimization strategies
- Bottlenecks identified

### Security
- Security considerations
- Authentication/authorization
- Data privacy

### Scalability
- Current capacity
- Scaling strategies
- Limitations

### Compliance
- Regulatory requirements
- Data retention policies
- Audit trails

---

## Quick Reference

### Common Commands
```bash
# Most frequently used commands
```

### Important Links
- Documentation
- Dashboards
- Issue tracker
- CI/CD pipelines

### Contact Information
- Team members
- On-call rotation
- Escalation paths

---

## Documentation Maintenance

**Last Updated**: [YYYY-MM-DD]
**Maintained By**: [Name/Team]
**Review Schedule**: [Monthly/Quarterly/etc.]

**Update Checklist**:
- [ ] Update change log
- [ ] Review and update architecture diagrams
- [ ] Update known issues
- [ ] Review and update setup instructions
- [ ] Update API documentation
- [ ] Review lessons learned

---

## Tips for Good Documentation

1. **Write for your future self**: Assume you'll forget everything in 6 months
2. **Keep it updated**: Outdated docs are worse than no docs
3. **Use examples**: Show, don't just tell
4. **Include context**: Explain why, not just what
5. **Make it searchable**: Use clear headings and structure
6. **Version control**: Keep docs in git, track changes
7. **Review regularly**: Schedule doc review sessions
8. **Get feedback**: Have others read your docs
9. **Link everything**: Cross-reference related docs
10. **Keep it concise**: Long docs are hard to maintain

---

## Example: Bug Entry

**Bug ID**: BUG-001
**Date Discovered**: 2024-01-15
**Date Fixed**: 2024-01-16
**Severity**: High
**Status**: Fixed

**Description**:
The intraday gap detection was flagging rows but not identifying specific missing timestamps. When a gap occurred between 09:05 and 09:15, the system would flag row 375 but wouldn't indicate that 09:10 was missing.

**Impact**:
- Made it impossible to know which timestamps to backfill
- Required manual inspection of timestamps
- Slowed down data quality improvement process

**Root Cause**:
The `_warn_if_timestamp_gaps()` function only calculated gap duration but didn't generate expected timestamps in the gap.

**Solution**:
Enhanced the function to:
1. Accept `expected_interval_minutes` parameter
2. Generate list of missing timestamps between gaps
3. Add `missing_timestamps` and `missing_count` columns to output
4. Improve console output to show missing timestamps

**Code Changes**:
- Modified `ml/scripts/prepare_features.py::_warn_if_timestamp_gaps()`
- Updated return DataFrame schema
- Added timestamp generation logic

**Testing**:
- Verified with known gaps in AAPL data
- Confirmed missing timestamps are correctly identified
- Tested with 5-minute and 1-minute intervals

**Prevention**:
- Add unit tests for gap detection
- Include edge cases (single missing bar, multiple missing bars)
- Document expected behavior in function docstring

---

## Example: Data Quality Issue Entry

**Issue ID**: DQ-001
**Date Discovered**: 2024-01-20
**Status**: Documented (Not Yet Fixed)

**Description**:
Daily bar gaps of 4 days (Friday to Tuesday) are being flagged as data quality issues, but many are legitimate market holidays where the market is closed.

**Impact**:
- False positives in data quality reports
- Manual filtering required
- Unclear which gaps are real issues vs holidays

**Known Holidays Causing Gaps**:
- Christmas Day + Weekend
- New Year's Day + Weekend
- MLK Day + Weekend
- Presidents' Day + Weekend
- Good Friday + Weekend
- Memorial Day + Weekend
- Independence Day + Weekend
- Labor Day + Weekend

**Proposed Solution**:
1. Integrate `pandas_market_calendars` library
2. Check if gaps contain trading days
3. Add `is_holiday_gap` flag to gap warnings
4. Filter holiday gaps from quality reports

**Workaround**:
Manually review gap warnings and filter known holidays using provided holiday list.

---

This template should be adapted to your specific project needs. Not all sections will be relevant for every project, but having a comprehensive structure ensures nothing important is forgotten.

