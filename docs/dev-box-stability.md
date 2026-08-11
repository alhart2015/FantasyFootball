# Dev-Box Stability — Diagnosis & Fix Guide

> **TL;DR:** The repeated worker crashes and BSODs we hit running the
> variance-model H2H backtest are **not** a CPU or thermal limit. The
> best-supported (but not yet *proven*) cause is **marginal memory margin under
> the enabled XMP/EXPO profile** (RAM runs at DDR5-5600 vs JEDEC 4800). The
> decisive, cheap test is to **disable XMP and re-run** — do that first. If the
> crashes vanish it was the memory profile; if they persist, escalate to
> MemTest86 + minidump driver attribution (§2).
>
> **Evidence caveat — SUPERSEDED, see §9.** This block previously read "the event
> log shows **no WHEA hardware-error events** … not a confirmed hardware fault."
> **That was wrong.** It came from a `StartTime=-7d` query that sat entirely *after*
> the error cluster. A full-history query finds **22 Processor-Core Corrected
> Machine Checks** (internal-parity / TLB errors on APIC 32/33). This **is** a
> confirmed CPU hardware fault — the Raptor Lake degradation defect on the
> i9-14900KF. The XMP-margin hypothesis below is retired. **Read §9 first.**
>
> **Update 2026-06-17 — AWCC overclock-off is NOT the fix (test log §7).**
> Disabling overclocking in **Alienware Command Center** does **not** touch the
> memory XMP/EXPO profile — RAM stayed at **5600** (`ConfiguredClockSpeed=5600`),
> and a 600-seed stress run still produced native `0xC0000005` worker crashes
> (5/5 retries on one chunk). AWCC's overclock toggle governs **CPU/GPU clocks**,
> not BIOS memory training. The decisive memory test (Step 1) is still pending —
> XMP must come off in **BIOS**, not AWCC.
>
> **Update 2026-06-17 #2 — leading suspect is now the CPU, not the RAM (§8).**
> Hardware identified: **Alienware Aurora R16 + Intel i9-14900KF** — the poster
> child for the Intel 13th/14th-gen "Raptor Lake" instability/**degradation**
> defect, whose symptoms (`0xC0000005` under load, corruption-flavored BSODs,
> empty WHEA, worsening over time, crashing even at idle) match ours almost
> exactly. Microcode is already current (`0x133` > the 0x12B fix), so the chip
> already has the mitigation that *prevents further* degradation but **cannot
> reverse damage already done** — and it still crashes. Likely an **RMA** (via
> Dell; Intel extended the 13/14th-gen warranty to 5 years). Also: 5600 is
> *native* for this CPU (not an XMP overclock vs 4800 — that was an Alder-Lake
> framing), so memory is now a weaker, secondary suspect.

This box is an i9 with 64 GB DDR5. The hardware is *capable*; it goes *unstable*
specifically under our sustained, CPU-bound, large-array float64 Monte-Carlo
load — a system-RAM stress pattern most "heavy computing" (GPU-bound games/
encode, bursty compiles) never exercises. This doc is the runbook for making it
stable and confirming the fix.

---

## 1. What the evidence actually shows

This section was revised after pulling the real Windows event log (2026-06-15).
The earlier draft asserted "WHEA corrected errors" — that was wrong; there are
none. Here is what the logs actually contain.

**Confirmed (from the event log):**

- **5 real BSODs** in 21 days (`Kernel-Power 41` + `WER-SystemErrorReporting
  1001`), clustered **6/11–6/14** — exactly the heavy-backtest days — plus one
  on 5/30.
- Bugcheck codes were **assorted and driver/kernel-exception flavored**:
  `0x1E` (KMODE_EXCEPTION_NOT_HANDLED), `0x3B` (SYSTEM_SERVICE_EXCEPTION, arg
  `0xC0000005`), `0x3D` (INTERRUPT_EXCEPTION_NOT_HANDLED), and `0x00020001` ×2
  (non-standard, identical signature both times).
- **Zero WHEA-Logger events** — the hardware machine-check log is empty. The CPU
  / memory controller did **not** report catching any bit errors.
- **No `python` Application-Error entries** — the user-mode worker "native
  crashes" never registered as Windows faults; the runner detected dead workers
  by exit code and retried.
- **No minidumps survived** on disk (dumps are *configured* — `CrashDumpEnabled=3`
  — but none were preserved), so the faulting driver can't be named from a dump
  yet.
- RAM is **64 GB DDR5 SK Hynix at 5600** (`ConfiguredClockSpeed=5600`), i.e.
  **XMP/EXPO is enabled** (JEDEC base for these modules is 4800).

**What this does and doesn't tell us:**

- It is **not** a confirmed hardware fault. The classic bad-RAM bugchecks
  (`0x124` WHEA_UNCORRECTABLE, `0x9C` MACHINE_CHECK, `0x1A` MEMORY_MANAGEMENT,
  `0x50`) are absent, and WHEA is empty.
- It is **not** a third-party security driver — only Windows Defender is present.
- The crashes correlate tightly with **our specific workload**: sustained,
  single-threaded, large-array float64 Monte-Carlo churn that hammers system-RAM
  bandwidth/capacity. The runner spawns workers *sequentially* (one at a time)
  with native threads already pinned to 1, so this is **not** thread
  oversubscription.

**Leading hypothesis — marginal memory margin under XMP.** The assorted codes +
**no WHEA** + retry-succeeds pattern fits silent memory corruption under an
aggressive XMP profile better than anything else: DDR5 on-die ECC masks on-die
single-bit flips, while bus/timing errors under tightened XMP timings are *not*
ECC-covered, so they corrupt silently and never reach WHEA. A flip in user
memory → a worker faults and the runner retries; a flip in kernel memory → a
BSOD carrying whatever "assorted" exception that corruption happens to throw.

This is a **hypothesis, not a verdict.** A buggy kernel driver tripped by the
workload can't be fully excluded without a minidump. That's why §2 leads with the
cheap falsifiable test (disable XMP, re-run) before any hardware replacement.

---

## 2. Fix order (do these in sequence, stop when stable)

Work top to bottom. Re-run the validation stress test (§4) after each step
before moving on — don't change five things at once.

### Step 1 — Disable XMP / EXPO (highest yield, do this first)

> **Do this in BIOS, not AWCC.** Turning overclocking off in **Alienware Command
> Center** does *not* disable XMP — it only lowers CPU/GPU clocks; the memory
> profile is untouched (verified 2026-06-17: RAM stayed at 5600 and crashes
> persisted, §7). XMP/EXPO lives in firmware.

1. Reboot into BIOS/UEFI (usually `Del` or `F2` at the splash).
2. Find the memory overclock profile — labeled **XMP** (Intel) or **EXPO**
   (AMD), often under "AI Tweaker", "Extreme Tweaker", "OC", or "Ai Overclock".
   On this Dell/Alienware board the menu may be hidden or differently named
   ("Memory Profile", "Advanced → Memory Configuration"), or unlock-gated — try
   the advanced-mode key combo. If it's genuinely locked, jump to MemTest86
   (Step 2) and treat the chunked runner as the durable workaround (§4).
3. Set it to **Disabled** (RAM falls back to JEDEC SPD defaults, e.g.
   DDR5-4800). You lose a little bandwidth; you gain correctness. Correctness
   wins — this project makes real decisions off these numbers.
4. Save & exit, boot, run the validation stress test (§4). Confirm the drop
   landed: `(Get-CimInstance Win32_PhysicalMemory).ConfiguredClockSpeed` should
   read ~4800, not 5600.

If stable here, you can *optionally* try re-enabling XMP later at a lower speed
tier or looser timings — but only if you care about the bandwidth. Stable-slow
beats fast-wrong.

### Step 2 — MemTest86 (confirm the RAM itself)

If Step 1 didn't fully stabilize it, test the modules directly:

1. Download **MemTest86** (PassMark, free edition) and write it to a USB stick
   with their imaging tool.
2. Boot from the USB (may need to disable Secure Boot temporarily, or use their
   Secure-Boot-signed image).
3. Run **at least 4 full passes** (overnight). One pass is not enough — marginal
   cells often only fail on specific patterns in later passes.
4. **Any** error = bad/marginal module or still-too-aggressive timing.
   - If errors with XMP off: test one DIMM at a time to find the bad stick (RMA
     it), or the bad slot.
   - If clean with XMP off but errors with XMP on: the modules are fine, the
     profile is the problem — leave XMP off (Step 1).

### Step 3 — Cooling / airflow

Heat narrows memory margins, so this *supports* stability even though it's not
the root cause:

- Ensure case intake/exhaust fans are actually spinning and unobstructed.
- Clear dust from filters and heatsinks.
- Confirm the CPU cooler is seated and its fan curve isn't set to silent.
- If the DIMMs run hot (some boards report DIMM temp in BIOS / HWiNFO), improve
  airflow over the RAM specifically.

### Step 4 — BIOS + chipset firmware update

Memory stability (especially DDR5 training) improves substantially across BIOS
revisions:

1. Note your motherboard model (BIOS shows it; or `Get-CimInstance Win32_BaseBoard`).
2. Download the **latest BIOS** from the board vendor's support page and flash it
   per their instructions (BIOS Flashback / EZ-Flash). **Re-disable XMP after
   flashing** — updates often reset to defaults *with* XMP, or change training.
3. Install the latest **chipset drivers** (Intel ME / chipset INF, or AMD
   chipset package) from the vendor.

---

## 3. How to read the evidence in Event Viewer

To confirm what crashed and why, after any future crash:

Open **Event Viewer** (`eventvwr.msc`) → **Windows Logs** → **System**, and look
for these sources/IDs around the crash time:

| Source | Event ID | Meaning |
|---|---|---|
| `WER-SystemErrorReporting` | **1001** | Bugcheck (BSOD) summary — contains the bugcheck code & minidump path. |
| `Microsoft-Windows-Kernel-Power` | **41** | "Kernel-Power 41" — the box lost power / reset without a clean shutdown (hard crash or BSOD reboot). |
| `Microsoft-Windows-WHEA-Logger` | **17 / 18 / 19 / 47** | Hardware error reported by the Machine Check Architecture. **"Corrected"** = hardware caught a bit error. **Id 19 = Processor-Core Corrected Machine Check** (the smoking gun here — see §9; 22 of them, internal-parity/TLB on APIC 32/33). **Id 17 = PCIe-root-port corrected** (incidental). ⚠️ **Query without a date window** — the prior `StartTime=-7d` pull missed the whole cluster and wrongly reported "none." |
| `BugCheck` / `WER-SystemErrorReporting` 1001 | — | The bugcheck *code* is the tell. `0x124`/`0x9C`/`0x1A`/`0x50` ⇒ memory/hardware. `0x1E`/`0x3B`/`0x3D` (what we saw) ⇒ kernel-exception, could be corruption *or* a driver. |

PowerShell quick-pull (last 7 days):

```powershell
Get-WinEvent -FilterHashtable @{
  LogName='System'
  ProviderName='Microsoft-Windows-WHEA-Logger','Microsoft-Windows-Kernel-Power','Microsoft-Windows-WER-SystemErrorReporting'
  StartTime=(Get-Date).AddDays(-7)
} | Select-Object TimeCreated, Id, ProviderName, Message | Format-List
```

⚠️ **Do not add a `StartTime` window to the WHEA pull** — the newest Processor-Core
machine check is 2026-05-13, so any "last N days" query run in June finds nothing and
falsely reports an empty log (this is exactly how the earlier "zero WHEA" error
happened). Query the whole log. WHEA-Logger **Id 19 / "Processor Core" / "Corrected
Machine Check"** events ARE present (§9) and are the direct confirmation that the
fault is the **CPU**, not our code. Note these are *core-internal* (parity/TLB), so
they implicate the processor itself, not the memory subsystem.

---

## 4. Validation stress test — confirm the fix

The variance-model H2H backtest **is** our memory stress test. It's a sustained,
multi-process, large-working-set numeric load — exactly what surfaced the
instability. If it runs clean to completion, the box is stable for our work.

Run the **chunked, resumable** runner (per the
[H2H backtest native crash](../C--Users-HartAlden-FantasyFootball/memory/h2h-backtest-native-crash.md)
memory note — never a single long-lived process on this box) for the
`season_value` vs `season_value_var` A/B:

- A clean full run (all chunks, **zero native-crash retries**, no BSOD) =
  stable. The retry count is the canary: during the failing runs we saw dozens of
  `0xC0000005`/`0xC0000409` worker retries; post-fix it should be **zero**.
- **Disabling XMP alone fixing it** confirms the memory-margin hypothesis (the
  profile was too aggressive — not "broken" silicon, and no RMA needed).
- If retries/BSODs persist after Step 1 (XMP off): run MemTest86 (Step 2). Clean
  MemTest + persistent crashes points at a **driver**, not the modules — capture a
  minidump (§3) and identify the faulting driver before considering any RMA. Only
  MemTest86 errors justify treating the modules as bad.

As a lighter smoke test before committing to a full A/B, any prolonged NumPy
workload works (e.g. a large repeated matmul loop across several worker
processes). But the backtest is the real acceptance test because it's the
workload we actually need to trust.

---

## 5. Environment-variable note (important — avoids a red herring)

The chunked runner (`scripts/h2h_backtest_chunked.py`) already sets these in the
worker env. For the record on what each one does:

- **`KMP_DUPLICATE_LIB_OK=TRUE` is *not* a no-op** (an earlier note in this doc
  claimed it was — wrong). The venv has numpy *and* scikit-learn/scipy, and those
  bundle their own OpenMP runtime (`libomp`) alongside numpy's BLAS. Two OpenMP
  runtimes loaded into one process triggers a hard abort; this variable suppresses
  that abort. The runner sets it deliberately and the code comment confirms it
  "works around the duplicate-OpenMP abort." Keep it.
- **`OMP_NUM_THREADS` / `OPENBLAS_NUM_THREADS` / `MKL_NUM_THREADS = 1`** pin the
  native math libs to one thread per worker. The runner spawns workers
  *sequentially*, so this is less about oversubscription and more about keeping
  each worker's footprint predictable and removing native-threading as a crash
  variable. The worker still crashes *and* the box still BSODs even with all of
  these set — which is exactly why the remaining suspect is the memory profile
  (§1), not threading.

These are *mitigations*, **not a fix**. The fix candidate is §2 (disable XMP).
Native threading is already neutralized and the instability persists, so don't
keep tuning env vars — run the XMP-off test.

---

## 6. Status of the work this blocked

- The **performance-variance model and the `season_value_var` variant are
  complete and committed** (branch `feat/performance-variance-model`, PR #68).
  All gates were green (pytest, mypy strict, ruff, ruff format) before this
  hardware issue surfaced. The code is unaffected.
- What's **deferred** is only the **A/B verdict** — i.e., empirically does
  `season_value_var` draft better than the deterministic `season_value` — which
  needs a clean full backtest run, which needs a stable box. Run §4 once the box
  is fixed to get that verdict.

---

## 7. Test log

### 2026-06-17 — AWCC overclock disabled: insufficient (crashes persist, no BSOD)

**Change under test:** user disabled overclocking in **Alienware Command Center**
(BIOS reportedly exposes no overclock menu). **RAM clock was not affected** —
`ConfiguredClockSpeed` still **5600** on both DIMMs (SK Hynix `HMCG88AGBUA081N`,
2×32 GB). So this tested *CPU/GPU-overclock-off only*; the suspected memory
profile (§1) was untouched.

**Stress run:** chunked runner, `season=2025`, `league_espn_half_16team`,
**600 seeds × n_sims 250**, jitter 8, chunk 20, `--chunk-timeout 300`
(`_h2h_ckpt_stress/`). Deliberately heavier than the prior failing run (3× seeds).

**Result — instability NOT fixed:**

- Chunks 0–280 (14/30) completed clean on first attempt (~18 min).
- Chunk **280–300 failed all 5 retries** with `rc=3221225477` = **`0xC0000005`
  (STATUS_ACCESS_VIOLATION)** — the same native worker-crash signature as before.
  Runner aborted, exit 1. Canary (native-crash retries) is **5, not 0** → fail.
- **No BSOD this run.** Event log clean for the window: no Kernel-Power 41, no
  WER 1001, **no WHEA**, no `Application Error`, no new minidump. The box stayed
  up; the crash stayed contained to worker processes.

**Reading:** removing the CPU overclock alone does not eliminate the native
crashes, consistent with the memory profile (still 5600) being the real culprit.
The 14-clean-then-5-consecutive-fails shape (failure only after ~18 min of
sustained churn, then *reproducible* at that chunk) fits the "margin shrinks as
the box warms" story in §1 better than an input-specific bug. The absence of a
BSOD *might* mean AWCC-off reduced severity from "box down" to "contained worker
crash" — but that's **one run**; do not treat it as a proven improvement (the
prior BSODs were themselves intermittent).

**Next:** get XMP/EXPO off in **BIOS** (§2 Step 1) — that is the still-untested
decisive lever. If the BIOS truly locks it, run MemTest86 (Step 2) and rely on
the chunked runner as the workaround. The A/B verdict (§6) remains blocked.

### 2026-06-17 (addendum) — idle BSOD ~9 min after the stress run

The box then **BSOD'd at ~09:36**, ~9 min after the stress run aborted (09:27)
and while only light diagnostic commands were running (near-idle):

- Bugcheck **`0x1E` KMODE_EXCEPTION_NOT_HANDLED, param1 `0xC0000005`** (kernel
  access violation), faulting IP `0xfffff8077eddc330`, referenced address `-1`
  (invalid). This is the **same `0xC0000005` access-violation signature the
  workers threw in user mode** an hour earlier — now in kernel space, so it took
  the box down instead of being retried.
- **Kernel-Power 41** hard reset 09:36:16. **WHEA still empty.** No preceding
  disk/driver errors — the box was quiet, then crashed.
- Repeats the 6/15 "crash at idle after the heavy run" pattern ⇒ the box is
  unstable **generally** now, not only under MC load.

**A minidump DID survive this time:** `C:\WINDOWS\Minidump\061726-10343-01.dmp`
(read is access-denied without elevation — which is almost certainly why §1
reported "none survived": a non-elevated `Get-ChildItem` of the protected folder
returns empty *silently*). `volmgr` events 161→162 confirm a minidump was written
for both the 6/16 and 6/17 crashes. **Driver attribution from this dump is the
long-missing step** to settle marginal-memory vs a buggy kernel driver — needs an
elevated copy + a dump analyzer (BlueScreenView, or `cdb !analyze -v`).

---

## 8. Major revision (2026-06-17) — the prime suspect is the CPU, not the RAM

Hardware finally identified: **Alienware Aurora R16, Intel Core i9-14900KF**
(Raptor Lake-S Refresh, Family 6 / Model 183 / Stepping 1), BIOS 2.23.0
(2025-12-28). This reframes the whole investigation.

**The i9-14900KF is the poster child for the documented Intel 13th/14th-gen
"Raptor Lake" instability/degradation defect.** Its symptom set is a near-exact
match to ours: repeated `0xC0000005` access-violation crashes under sustained
compute, BSODs with corruption-flavored bugcheck codes, an **empty WHEA log**,
and instability that **worsens over time and eventually happens even at idle** —
the signature of a chip that has physically degraded from prolonged elevated
voltage. Intel's remedies: latest microcode + "Intel Default Settings", and an
**RMA if the chip already degraded** (Intel extended the 13/14th-gen desktop
warranty to 5 years; for an OEM system like this the RMA goes through **Dell**,
not Intel directly).

**Microcode is already current: revision `0x133`** (newer than the 0x125 eTVB /
0x129 / 0x12B instability fixes). So the mitigation that *prevents further*
degradation is in place — but microcode **cannot repair a chip that already
degraded.** Continuing to crash on up-to-date microcode is itself evidence the
chip is likely already degraded.

**The memory framing in §1 was platform-wrong.** The i9-14900KF *natively*
supports DDR5-5600 (Intel ARK), and the installed config — 2× 32 GB SK Hynix
`HMCG88AGBUA081N`, 1 DIMM/channel, dual-rank — is within Intel's 1-DPC dual-rank
support. So **5600 here is not an "XMP overclock vs 4800 JEDEC"** (that came from
12th-gen Alder Lake, where DDR5 base was 4800). Marginal memory is now a weaker,
secondary suspect; the CPU is the stronger one.

**Revised fix order (supersedes §2 for this box):**

1. **Run fully stock and retest.** In BIOS: ensure **Intel Default Settings**
   (no Alienware/AWCC OC profile applied) and set memory to JEDEC default / XMP
   off. This removes *both* CPU-power and memory as variables in one shot. Re-run
   the §4 stress test.
2. **If it still crashes at full stock → treat the CPU as degraded → RMA via
   Dell** under the 13/14th-gen extended warranty. A clean-stock crash IS the
   evidence Dell/Intel require; doing step 1 first also satisfies their
   "try default settings" precondition.
3. MemTest86 (memory confirmation) and minidump driver attribution remain
   available but are now **secondary** to the CPU/RMA path.

### 2026-06-17 (addendum #2) — BIOS is locked; the tuning levers don't exist → RMA path

User went into the Aurora R16 BIOS: **no XMP / memory-speed control** (5600 is
shown but read-only), **no CPU voltage options** even after enabling the
"Overclocking Feature", and overclocking was **already disabled**. BIOS **2.23.0
is the latest** (confirmed on Dell support) and microcode is already **0x133**.

So every user-tunable software/firmware lever is **exhausted or absent**, and the
box still crashes at full stock — including at idle. Step 1 (XMP-off) is
**impossible on this locked OEM board**. The investigation collapses to a
**hardware fault on a fully-updated, locked system → Dell warranty / RMA.**

- **Service Tag: `CJFDH04`** (Aurora R16, SKU 0CD2). SupportAssist is installed.
- **Crash history (last 30d):** 8 BSODs — 6/17 `0x1E`, 6/16 `0x20001`, 6/15
  `0x20001`, 6/14 `0x1E`, 6/13 `0x3B`, 6/12 `0x20001`, 6/11 `0x3D`, 5/30
  `0x20001`. All corruption/kernel-exception flavored; WHEA empty throughout.
- **RMA prep:** run Dell **pre-boot diagnostics** (F12 → Diagnostics; extended
  memory + CPU stress) to get an error/validation code; open a Dell case with the
  Service Tag + this crash list. MemTest86 + minidump attribution optional to
  tighten the case. Intel's 5-yr 13/14th-gen CPU extended warranty is the
  backstop if the Dell system warranty has lapsed.
- **Stopgap while RMA pends:** cap Windows "max processor state" ~99% (kills the
  top turbo/voltage bin where Raptor Lake degradation crashes worst); keep heavy
  jobs on the chunked runner.

**Sources:** Intel Community microcode 0x129/0x12B advisories; Hardware Times and
Tom's Hardware coverage of 13/14th-gen instability; Dell Aurora R16 owner's
manual / community threads on XMP behavior.

---

## 9. Resolved (2026-06-17) — CONFIRMED CPU fault: processor-core machine checks in WHEA

Two findings close the investigation.

**(a) The Dell pre-boot diagnostic — QUICK *and* THOROUGH — both PASSED. This does
not clear the chip.** Corrected machine checks are *corrected*: the CPU detects and
fixes the parity/TLB error, so it still returns correct results during a short
functional test. The fault only escalates to an uncorrectable BSOD under sustained
real load at peak boost voltage/thermals, which the diagnostic never reproduces. A
passing Dell test is the **expected** result for a degrading-but-still-correcting
CPU — it is not evidence of health.

**(b) WHEA is NOT empty — the earlier "zero WHEA" claim was wrong.** Re-querying the
System log *without* a 7-day window (the prior query used `StartTime=-7d`, which sits
entirely after the error cluster, so it found nothing) surfaces **32 WHEA-Logger
events**, of which **22 are Processor-Core Corrected Machine Checks**:

- Error source **Corrected Machine Check**, component **Processor Core**.
- Error types: **Internal parity error** (21) + **Translation Lookaside Buffer
  error** (1).
- Localized to **APIC ID 32/33** — one physical P-core and its SMT sibling.
- Dated **2025-11-14 → 2026-05-13** (a cluster of 19 on 2026-05-11).
- The other 10 WHEA events are **PCIe-root-port corrected errors**
  (`VEN_8086&DEV_7ABC`, bus 0:1C:0) — a separate, incidental marginal link, **not**
  the crash cause.

Healthy CPUs do not log processor-core internal-parity machine checks. Combined with
the BSOD scatter of `0xC0000005` (access violation), `0xC0000096` (privileged
instruction) and `0xC000001D` (illegal instruction) at random kernel addresses — the
fingerprint of corrupted instruction execution — plus near-daily June crash
acceleration, this is the **Intel Raptor Lake degradation defect, confirmed on the
i9-14900KF.**

The corrected errors **stop after 2026-05-13** while hard crashes continue and
accelerate through June. That is not recovery — it is consistent with degradation
progressing **past the correction envelope** (corrected → uncorrected), i.e. errors
that used to be caught and logged now go straight to a BSOD.

**Verdict: the CPU is degraded → RMA via Dell.** Evidence package written to
`_crashdump/whea-cpu-evidence.txt` (the full Processor-Core CMC list + bugcheck
history + the why-Dell-passes explanation). For the Dell/Intel case the
Processor-Core CMC log is the single most persuasive artifact — far stronger than "my
workload crashes." **If the Dell tech leans on the passing diagnostic, the rebuttal
is point (a): a corrected machine check by definition passes a functional test.**

**Stopgap until the RMA:** cap Windows "max processor state" to ~99%
(`powercfg /setacvalueindex SCHEME_CURRENT SUB_PROCESSOR PROCTHROTTLEMAX 99` then
`powercfg /setactive SCHEME_CURRENT`) — this drops the top turbo/voltage bin where
degradation faults worst — and keep heavy jobs on the chunked runner. Mitigates, does
not fix.

**RMA filed 2026-06-17.** Dell warranty report submitted. Service request
**`227929375`**, Express Service Code **`27296068180`** (Service Tag `CJFDH04`). Dell
to make contact **2026-06-18**. When they call: lead with the Processor-Core WHEA
machine checks (evidence package at `_crashdump/whea-cpu-evidence.txt`); if they cite
the passing diagnostic, use the point-(a) rebuttal above.

**Dell response (2026-06-18) — CPU replacement is pre-authorized.** Dell tech
(Samson H.) sent a 3-step protocol: (1) install Dell-recommended NVIDIA driver, (2)
update BIOS/microcode, (3) run onboard diagnostics (F12 → Diagnostics, ESC to skip
the extended memory test). Crucial line: **"If no errors reported, we would replace
the processor to resolve the issue."** So the passing diagnostic is not a dead end —
it is the *trigger* Dell uses to approve the CPU swap (a corrected machine check
passes functional tests by design). Status of their steps: BIOS **already latest**
(`2.23.0`, the newest Dell publishes, + microcode `0x133`) — nothing to do;
diagnostic **already passes**; only open item is the **NVIDIA driver** (currently
`595.95`/Mar-2026 — install Dell's validated build from the support page for tag
`CJFDH04`). Plan: update GPU driver → re-run onboard diagnostic per their steps →
reply that no errors were reported and request the processor replacement.
