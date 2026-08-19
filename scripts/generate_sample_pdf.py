#!/usr/bin/env python3
"""Create sample_data/vendor_security.pdf for local demos."""

from pathlib import Path

import pymupdf

TEXT = """ACME CLOUD INC.
Vendor Security and Compliance Statement
Confidential

1. Incident Notification
Acme Cloud maintains formally defined criteria for notifying a client during an incident
that might impact the security of their data or systems. A Security Incident is any
confirmed event that actually or potentially jeopardizes the confidentiality, integrity,
or availability of customer data or systems.

Notification SLA: Affected customers are notified within 24 hours of incident
confirmation. Status updates are provided at least every 24 hours until resolution.
Critical incidents impacting availability may include an initial notification within
1 hour of confirmation.

2. Third-Party Processing of Personal Information
Yes. Personal information is transmitted, processed, stored, and disclosed to
subprocessors retained by Acme Cloud. Third parties include Amazon Web Services (AWS)
for infrastructure hosting and backups, Google Workspace for corporate email and
collaboration, Stripe for payment processing, and Datadog for monitoring. These parties
process data under written Data Processing Agreements and confidentiality obligations.
Acme Cloud does not sell personal information.

3. Cloud Providers
Acme Cloud relies on Amazon Web Services (AWS) as the primary cloud provider and
Google Cloud Platform (GCP) for selected analytics workloads.

4. Data Center Locations
The primary data center location/region of the underlying cloud infrastructure used to
host the service(s) is AWS us-east-1 (Northern Virginia, USA). Backup and disaster
recovery copies are stored in AWS us-west-2 (Oregon, USA). Google Cloud analytics
workloads run in us-central1.

5. Monitoring
The following are performed as part of the monitoring process for the service:
- Application Performance Monitoring (APM): Yes. Implemented with Datadog APM on all
  production services.
- End User Monitoring (EUM): Yes. Real-user monitoring is enabled for the web application.
- Digital Experience Monitoring (DEM): No. DEM is not currently part of the standard
  monitoring process.
"""


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output = root / "sample_data" / "vendor_security.pdf"
    document = pymupdf.open()
    page = document.new_page()
    page.insert_textbox(pymupdf.Rect(56, 56, 540, 760), TEXT, fontsize=10, fontname="helv")
    document.save(output)
    document.close()
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
