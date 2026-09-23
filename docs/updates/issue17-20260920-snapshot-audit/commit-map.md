# Per-commit audit map — paused at user request

**Resume only on explicit user request and after rechecking upstream/audit applicability.** Kernel k01 alone was replayed as `c66a48486dcd6b76336ec8a8b984a4700f89e1d9`; hooks passed, native build/GPU qualification pending. No vLLM source replay.

All 27 dedicated reports are parent-reviewed and reconciled. Final source SHAs and qualification remain pending. Exact requirements, dependencies, report hashes and gates are in `commit-map.json`.

| ID | Original SHA | Agent | Disposition | Owner/fold | Report |
| --- | --- | --- | --- | --- | --- |
| v01 | `f1056606736e` | /root/audit_v01 | DROP | v01 | [audit](audits/v01-f1056606736e.md) |
| v02 | `cf0b834cd66c` | /root/audit_v02 | KEEP | v02 | [audit](audits/v02-cf0b834cd66c.md) |
| v03 | `7de1e1c5ec5f` | /root/audit_v03 | KEEP | v03 | [audit](audits/v03-7de1e1c5ec5f.md) |
| v04 | `c32113ed68b8` | /root/audit_v04 | ADAPT | v04 | [audit](audits/v04-c32113ed68b8.md) |
| v05 | `fcab69494f17` | /root/audit_v05 | ADAPT | v05 | [audit](audits/v05-fcab69494f17.md) |
| v06 | `6700b7cad87b` | /root/audit_v06 | DROP | v06 | [audit](audits/v06-6700b7cad87b.md) |
| v07 | `2352837c5eb1` | /root/audit_v07 | ADAPT | v05 | [audit](audits/v07-2352837c5eb1.md) |
| v08 | `3b6c822e8a4f` | /root/audit_v08 | ADAPT | v05 | [audit](audits/v08-3b6c822e8a4f.md) |
| v09 | `74a764d57a8c` | /root/audit_v09 | KEEP | v09 | [audit](audits/v09-74a764d57a8c.md) |
| v10 | `80d5166fcb0b` | /root/audit_v10 | ADAPT | v05 | [audit](audits/v10-80d5166fcb0b.md) |
| v11 | `73fb9e7aa230` | /root/audit_v11 | ADAPT | v05 | [audit](audits/v11-73fb9e7aa230.md) |
| v12 | `6519bf709dbb` | /root/audit_v12 | ADAPT | v05 | [audit](audits/v12-6519bf709dbb.md) |
| v13 | `d6cb2e6cb2de` | /root/audit_v13 | ADAPT | v05 | [audit](audits/v13-d6cb2e6cb2de.md) |
| v14 | `32289022a4e3` | /root/audit_v14 | ADAPT | v05 | [audit](audits/v14-32289022a4e3.md) |
| v15 | `2e010408b971` | /root/audit_v15 | ADAPT | v05 | [audit](audits/v15-2e010408b971.md) |
| v16 | `2fdd9d71f689` | /root/audit_v16 | ADAPT | v05 | [audit](audits/v16-2fdd9d71f689.md) |
| k01 | `8530de9d4298` | /root/audit_k01 | KEEP | k01 | [audit](audits/k01-8530de9d4298.md) |
| k02 | `27a9c28f8598` | /root/audit_k02 | KEEP | k02 | [audit](audits/k02-27a9c28f8598.md) |
| k03 | `32f424757748` | /root/audit_k03 | KEEP | k03 | [audit](audits/k03-32f424757748.md) |
| k04 | `09925af1f9cb` | /root/audit_k04 | ADAPT | k04 | [audit](audits/k04-09925af1f9cb.md) |
| k05 | `9e2ac704b862` | /root/audit_k05 | KEEP | k05 | [audit](audits/k05-9e2ac704b862.md) |
| k06 | `be4d79b04ab6` | /root/audit_k06 | ADAPT | k06 | [audit](audits/k06-be4d79b04ab6.md) |
| k07 | `8900abc2df3e` | /root/audit_k07 | ADAPT | k07 | [audit](audits/k07-8900abc2df3e.md) |
| k08 | `9bb1e2d428b8` | /root/audit_k08 | KEEP | k08 | [audit](audits/k08-9bb1e2d428b8.md) |
| k09 | `f26ef2323399` | /root/audit_k09 | KEEP | k08 | [audit](audits/k09-f26ef2323399.md) |
| k10 | `c69cfad0586b` | /root/audit_k10 | KEEP | k08 | [audit](audits/k10-c69cfad0586b.md) |
| k11 | `2c39287deec5` | /root/audit_k11 | KEEP | k08 | [audit](audits/k11-2c39287deec5.md) |
