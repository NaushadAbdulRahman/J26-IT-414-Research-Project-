# Component 2 – Risk Register (updated 21 Oct 2026)

Likelihood / impact: H = high, M = medium, L = low.

| # | Risk | L | I | Mitigation (what we do) | Status |
|---|---|---|---|---|---|
| R1 | Local (Kelaniya) data is small or arrives late, so no group can be found locally | H | H | The tool reruns on any dataset, so the contribution does not depend on one data source. The US prototype shows the method works. "No group found" is reported as a valid result | Prototype done on US data. Local data request to follow once the prototype is shown to the supervisor |
| R2 | Groups found on US patients do not transfer to Sri Lankan patients (for example BMI 35 or more is rare locally) | H | H | Do not carry the list over. Rerun the full search on local data with bins suited to local patients. Every result is labelled "candidate on US data" | Labels are on the page and in the README. Local bins are planned for PP2 |
| R3 | A group is real-looking but only luck | M | H | Benjamini–Hochberg correction, retest on unseen patients (20 fixed splits), and a null test with fake data | Done. Null test: 4 of 200 runs flagged a group (2.0%). The BMI groups held in only 5 to 6 of 20 splits, so they are stated as candidates, not proven |
| R4 | A better screening model leaves no group worth a scan | M | M | The audit reports "no group passes" as a valid result and still shows the blind-spot map, so the tool stays useful as a checking tool | Realised on Component 1 (no group among 240). Handled and shown on the page |
| R5 | Component 1 predictions do not match our patient table (format or cohort) | M | M | A fixed file contract (SEQN, probability, referred). The audit refuses files without the right columns and prints how many patients matched | Connection working (1,011 of 1,011 matched). One shared cohort definition to be agreed after PP1 |
| R6 | The live demo fails on the day | M | H | Screenshots of every screen saved. A recorded fallback demo and backup run | Screenshots saved. Recording in progress |
| R7 | Teams channel and Planner not set up for our group | H | M | Email sent to the RP office on 21 Oct. GitHub history and the signed logbook already provide evidence of progress | Email sent. Awaiting reply |
| R8 | Privacy and ethics when local patient data is used | L | H | Only public, de-identified NHANES data is used now. Local data will be used only with the hospital's ethical clearance and without patient identifiers in the repository | Data files are excluded by .gitignore. Clearance to be arranged with the supervisor |