# Organizational Role Discovery Report

**Algorithm**: Burattin, Sperduti, & Veluscek (2013) — *Business models enhancement through discovery of roles*
**Aggregation Threshold (θ)**: `0.35`
**Total Discovered Roles**: `7`
**Total Distinct Workers**: `653`

---

## Discovered Organizational Departments & Schedules

| Role / Department | Activities Assigned | Qualified Staff Count | Concurrency ($C_{role}$) | Operating Shift Window | Active Days |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Role_0_Authorization_Reques** | `Authorization Requested`<br/>`Evaluating Request (NO registered letter)`<br/>`Network Adjustment Requested`<br/>`Request created` | 634 workers | **629** | `08:00 - 18:00` | Mon, Tue, Wed, Thu, Fri |
| **Role_1_Service_closure_Requ** | `Service closure Request with network responsibility` | 598 workers | **297** | `08:00 - 18:00` | Mon, Tue, Wed, Thu, Fri |
| **Role_2_Evaluating_Request_(** | `Evaluating Request (WITH registered letter)` | 393 workers | **25** | `09:00 - 18:00` | Mon, Tue, Wed, Thu, Fri |
| **Role_3_FrontOffice_Processing** | `Request completed with customer recovery` | 49 workers | **1** | `08:00 - 18:00` | Mon, Tue, Wed, Thu, Fri |
| **Role_4_FrontOffice_Processing** | `Back-Office Adjustment Requested`<br/>`Pending Liquidation Request`<br/>`Pending Request for Reservation Closure`<br/>`Pending Request for acquittance of heirs`<br/>`Request completed with account closure`<br/>`Service closure Request with BO responsibility` | 26 workers | **26** | `03:00 - 17:00` | Mon, Tue, Wed, Thu, Fri, Sat |
| **Role_5_Request_deleted** | `Request deleted` | 16 workers | **1** | `09:00 - 18:00` | Mon, Tue, Wed, Thu, Fri |
| **Role_6_Pending_Request_for_** | `Pending Request for Network Information` | 3 workers | **3** | `09:00 - 18:00` | Mon, Tue, Wed, Thu, Fri |

---

## Multi-Skilled Human Staffing Analysis

In this hybrid model, each enterprise worker is tracked as an individual biological entity with an empirical concurrency quota ($C_w$), eliminating worker ubiquity across departments while allowing natural task interleaving.

- **Median Worker Concurrency Quota ($C_w$)**: `3` simultaneous open cases
- **Max Worker Concurrency Quota**: `1859` simultaneous open cases

### Top Staff Roster & Concurrency Quotas

| Worker ID | Concurrency Quota ($C_w$) | Department Memberships |
| :--- | :---: | :--- |
| `BOF` | **1859** | Role_0_Authorization_Reques, Role_4_FrontOffice_Processing |
| `APPLICANT` | **177** | Role_1_Service_closure_Requ, Role_5_Request_deleted |
| `332` | **129** | Role_0_Authorization_Reques, Role_3_FrontOffice_Processing, Role_4_FrontOffice_Processing, Role_5_Request_deleted, Role_6_Pending_Request_for_ |
| `48` | **98** | Role_0_Authorization_Reques, Role_3_FrontOffice_Processing, Role_4_FrontOffice_Processing, Role_5_Request_deleted, Role_6_Pending_Request_for_ |
| `303` | **56** | Role_0_Authorization_Reques, Role_4_FrontOffice_Processing |
| `270` | **31** | Role_0_Authorization_Reques, Role_1_Service_closure_Requ |
| `BACK-OFFICE` | **25** | Role_4_FrontOffice_Processing |
| `539` | **18** | Role_0_Authorization_Reques, Role_2_Evaluating_Request_(, Role_4_FrontOffice_Processing |
| `93` | **16** | Role_0_Authorization_Reques, Role_1_Service_closure_Requ, Role_2_Evaluating_Request_(, Role_3_FrontOffice_Processing |
| `140` | **15** | Role_0_Authorization_Reques, Role_1_Service_closure_Requ, Role_2_Evaluating_Request_(, Role_3_FrontOffice_Processing |
| `32` | **15** | Role_0_Authorization_Reques, Role_1_Service_closure_Requ, Role_2_Evaluating_Request_( |
| `156` | **14** | Role_0_Authorization_Reques, Role_1_Service_closure_Requ, Role_2_Evaluating_Request_(, Role_3_FrontOffice_Processing |
| `455` | **13** | Role_0_Authorization_Reques, Role_1_Service_closure_Requ, Role_2_Evaluating_Request_( |
| `21` | **12** | Role_0_Authorization_Reques, Role_1_Service_closure_Requ, Role_2_Evaluating_Request_( |
| `306` | **12** | Role_0_Authorization_Reques, Role_1_Service_closure_Requ, Role_2_Evaluating_Request_( |
