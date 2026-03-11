import { useState, useEffect, useRef } from "react";

// ─── Path to the bundled amfAR CSV ───────────────────────────────────────────
// Place the downloaded file in your project's public folder and update this
// name to match exactly. Common filenames from opioid.amfar.org:
//   "opioid_indicators_county.csv"  or  "amfar_opioiddata_county.csv"
const CSV_PATH = "./opioid_indicators_county.csv";

// ─── Design Tokens ──────────────────────────────────────────────────────────
const C = {
  bg: "#f5efe6", surface: "#faf6f0", darker: "#ede5d8",
  border1: "#d4c4ae", border2: "#c4b09a",
  ink: "#2c1f0e", muted: "#8a7560",
  terra: "#b85c38", sage: "#4a7c59", gold: "#c47f2a", slate: "#4a5a72",
  terraBg: "#fdf0eb", sageBg: "#eef5f0", goldBg: "#fdf6e8", slateBg: "#edf0f5",
};
const F = {
  display: "'Playfair Display', Georgia, serif",
  body: "'Source Serif 4', Georgia, serif",
  mono: "'JetBrains Mono', 'Courier New', monospace",
};

// ─── Load Google Fonts ───────────────────────────────────────────────────────
(function loadFonts() {
  if (typeof document === "undefined") return;
  const link = document.createElement("link");
  link.rel = "stylesheet";
  link.href =
    "https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&family=Source+Serif+4:wght@400;600&family=JetBrains+Mono:wght@400;600&display=swap";
  document.head.appendChild(link);
})();

// ─── Seeded PRNG (mulberry32) ────────────────────────────────────────────────
function seededRand(seed) {
  let s = seed >>> 0;
  return function () {
    s |= 0; s = s + 0x6d2b79f5 | 0;
    let t = Math.imul(s ^ s >>> 15, 1 | s);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}
function hashStr(str) {
  let h = 5381;
  for (let i = 0; i < str.length; i++) h = (h * 33 ^ str.charCodeAt(i)) >>> 0;
  return h;
}

// ─── JS Ports of math_engine.py ─────────────────────────────────────────────
// JS port of poisson_mle() in math_engine.py
function poissonMLE(odDeaths, population) {
  if (!population) return 0;
  return (odDeaths / population) * 100000;
}
// JS port of combinatorics() in math_engine.py
function combinatorics(n, k) {
  if (k > n || k < 0) return 0;
  if (k === 0 || k === n) return 1;
  k = Math.min(k, n - k);
  let result = 1;
  for (let i = 0; i < k; i++) result = result * (n - i) / (i + 1);
  return Math.round(result);
}
// JS port of _effectiveness() in math_engine.py
function eff(miles) {
  if (miles <= 5) return 0.40;
  if (miles <= 20) return 0.40 * (20 - miles) / 15;
  return 0;
}
// JS port of conditional_prob() in math_engine.py
function conditionalProb(baselineP, distanceMiles) {
  return baselineP * (1 - eff(distanceMiles));
}
// JS port of bootstrap_pvalue() in math_engine.py
function bootstrapPvalue(optimalScore, randomScores) {
  if (!randomScores.length) return 1;
  return randomScores.filter(s => s <= optimalScore).length / randomScores.length;
}
// JS port of clt_confidence_interval() in math_engine.py
function cltCI(lambdaList, z = 1.96) {
  const mu = lambdaList.reduce((a, b) => a + b, 0);
  const sigma = Math.sqrt(mu);
  return { mu, sigma, lo: mu - z * sigma, hi: mu + z * sigma };
}

// ─── Helpers ─────────────────────────────────────────────────────────────────
const fmt = (n, d = 1) => typeof n === "number" ? n.toFixed(d) : "—";
const fmtInt = n => Math.round(n).toLocaleString();

function scorePlacement(idxs, cands) {
  // score = sum of expected deaths WITH site (distance = 0 for selected county)
  // unselected counties get their baseline (no benefit)
  const selectedSet = new Set(idxs);
  return cands.reduce((sum, c, i) => {
    const d = selectedSet.has(i) ? 0 : 999;
    return sum + conditionalProb(c.od_rate / 100000, d) * (c.population / 100000) * 100000;
  }, 0);
}

// ─── CSV Parser ───────────────────────────────────────────────────────────────
function parseAmfAR(text) {
  const lines = text.trim().split(/\r?\n/);
  const headers = lines[0].split(",").map(h => h.trim().replace(/^"|"$/g, "").toLowerCase());
  const get = (row, name) => {
    const i = headers.indexOf(name);
    return i === -1 ? "" : (row[i] || "").replace(/^"|"$/g, "").trim();
  };
  const raw = {};
  for (let i = 1; i < lines.length; i++) {
    const row = lines[i].split(",");
    const county = get(row, "county");
    const state = get(row, "stateabbreviation");
    const indicator = get(row, "indicator");
    const value = parseFloat(get(row, "value"));
    const year = parseInt(get(row, "year"));
    if (!county || !state || isNaN(value)) continue;
    const key = `${state}||${county}`;
    if (!raw[key]) raw[key] = { county, state, years: {} };
    if (!raw[key].years[indicator]) raw[key].years[indicator] = {};
    if (!raw[key].years[indicator][year] || year > Object.keys(raw[key].years[indicator]).reduce((a,b) => Math.max(a,parseInt(b)), 0)) {
      raw[key].years[indicator][year] = value;
    }
  }
  function latest(obj) {
    if (!obj) return null;
    const yrs = Object.keys(obj).map(Number).filter(y => !isNaN(y));
    if (!yrs.length) return null;
    return obj[Math.max(...yrs)];
  }
  const counties = [];
  for (const key of Object.keys(raw)) {
    const r = raw[key];
    const population = latest(r.years["Population"]);
    const od_deaths = latest(r.years["drugdeaths"]);
    const poverty_pct = latest(r.years["pct_poverty"]);
    if (!population || !od_deaths || population <= 0 || od_deaths <= 0) continue;
    const od_rate = poissonMLE(od_deaths, population);
    counties.push({ county: r.county, state: r.state, population, od_deaths, poverty_pct: poverty_pct || 0, od_rate });
  }
  return counties;
}

// ─── Rejection Sampling ───────────────────────────────────────────────────────
function sampleCandidates(stateCounties, stateName, maxK = 15) {
  const rand = seededRand(hashStr(stateName));
  const total = stateCounties.reduce((s, c) => s + c.od_rate, 0);
  const pool = [...stateCounties];
  const selected = [];
  while (selected.length < maxK && pool.length > 0) {
    const r = rand() * pool.reduce((s, c) => s + c.od_rate, 0);
    let acc = 0;
    for (let i = 0; i < pool.length; i++) {
      acc += pool[i].od_rate;
      if (r <= acc) { selected.push(pool.splice(i, 1)[0]); break; }
    }
  }
  return selected;
}

// ─── Sub-components ──────────────────────────────────────────────────────────
function Toast({ msg, type, visible }) {
  const borderColor = type === "success" ? C.sage : type === "gold" ? C.gold : C.terra;
  return (
    <div style={{
      position: "fixed", bottom: 32, left: "50%", transform: `translateX(-50%) translateY(${visible ? 0 : 80}px)`,
      transition: "transform 0.3s ease", zIndex: 9999,
      background: C.surface, border: `2px solid ${borderColor}`, borderRadius: 10,
      padding: "12px 24px", fontFamily: F.body, color: C.ink, fontSize: 14,
      boxShadow: "0 4px 20px rgba(44,31,14,0.15)", pointerEvents: "none",
    }}>{msg}</div>
  );
}

function ProgressBar({ value, max, color = C.terra }) {
  return (
    <div style={{ background: C.darker, borderRadius: 4, height: 6, overflow: "hidden", margin: "8px 0" }}>
      <div style={{ width: `${Math.min(100, (value / max) * 100)}%`, height: "100%", background: color, transition: "width 0.3s" }} />
    </div>
  );
}

function ScoreBar({ label, score, baseline, color }) {
  const pct = baseline > 0 ? Math.min(100, (score / baseline) * 100) : 0;
  return (
    <div style={{ marginBottom: 8 }}>
      <div style={{ display: "flex", justifyContent: "space-between", fontFamily: F.mono, fontSize: 12, color: C.muted, marginBottom: 4 }}>
        <span>{label}</span><span style={{ color }}>{fmtInt(score)}</span>
      </div>
      <div style={{ background: C.darker, borderRadius: 4, height: 10, overflow: "hidden" }}>
        <div style={{ width: `${pct}%`, height: "100%", background: color, borderRadius: 4 }} />
      </div>
    </div>
  );
}

function ResultCard({ title, value, subtitle, color, bg, span = 1 }) {
  return (
    <div style={{ gridColumn: `span ${span}`, background: bg, border: `2px solid ${color}`, borderRadius: 12, padding: "20px 24px" }}>
      <div style={{ fontFamily: F.mono, fontSize: 11, color, textTransform: "uppercase", letterSpacing: 1, marginBottom: 4 }}>{title}</div>
      <div style={{ fontFamily: F.display, fontSize: 42, color, lineHeight: 1.1 }}>{value}</div>
      {subtitle && <div style={{ fontFamily: F.body, fontSize: 13, color: C.muted, marginTop: 6 }}>{subtitle}</div>}
    </div>
  );
}

function MathAccordion({ pill, pillColor, title, description, liveValue, steps }) {
  const [open, setOpen] = useState(false);
  return (
    <div style={{ border: `1px solid ${C.border1}`, borderRadius: 10, marginBottom: 8, background: C.surface, overflow: "hidden" }}>
      <div onClick={() => setOpen(o => !o)} style={{ padding: "12px 16px", cursor: "pointer", display: "flex", alignItems: "center", gap: 10 }}>
        <span style={{ background: pillColor, color: "#fff", fontFamily: F.mono, fontSize: 10, borderRadius: 20, padding: "2px 10px", flexShrink: 0 }}>{pill}</span>
        <span style={{ fontFamily: F.body, fontSize: 14, color: C.ink, flex: 1 }}>{title}</span>
        <span style={{ fontFamily: F.mono, fontSize: 13, color: pillColor, marginRight: 8 }}>{liveValue}</span>
        <span style={{ color: C.muted, fontSize: 12 }}>{open ? "▼" : "▶"} Math</span>
      </div>
      {open && (
        <div style={{ padding: "0 16px 16px", borderTop: `1px solid ${C.darker}` }}>
          <p style={{ fontFamily: F.body, fontSize: 13, color: C.muted, margin: "10px 0 12px" }}>{description}</p>
          {steps.map((st, i) => (
            <div key={i} style={{ marginBottom: 12 }}>
              <div style={{ fontFamily: F.mono, fontSize: 11, color: C.muted, marginBottom: 4 }}>Step {i + 1} — {st.label}</div>
              <div style={{ background: C.darker, borderLeft: `3px solid ${pillColor}`, padding: "8px 12px", fontFamily: F.mono, fontSize: 12, color: C.ink, whiteSpace: "pre-wrap", borderRadius: "0 6px 6px 0" }}>{st.formula}</div>
              {st.note && <div style={{ fontFamily: F.body, fontSize: 12, color: C.muted, marginTop: 4, fontStyle: "italic" }}>{st.note}</div>}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function CountyCard({ c, selected, locked, riskTier, onClick }) {
  const tierColor = riskTier === 0 ? C.terra : riskTier === 1 ? C.gold : C.slate;
  const borderColor = selected ? C.slate : C.border1;
  const bg = selected ? C.slateBg : C.surface;
  return (
    <div onClick={locked && !selected ? undefined : onClick} style={{
      border: `2px solid ${borderColor}`, borderRadius: 10, padding: "14px", background: bg,
      opacity: locked && !selected ? 0.38 : 1,
      cursor: locked && !selected ? "not-allowed" : "pointer",
      transition: "all 0.15s",
    }}>
      <div style={{ fontFamily: F.display, fontSize: 15, color: C.ink, fontWeight: 700, marginBottom: 2 }}>{c.county}</div>
      <div style={{ fontFamily: F.mono, fontSize: 10, color: C.slate, marginBottom: 8 }}>{c.state}</div>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6, fontFamily: F.mono, fontSize: 11, color: C.muted }}>
        <span>OD rate</span><span style={{ color: tierColor, fontWeight: 600 }}>{fmt(c.od_rate)} / 100k</span>
      </div>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 8, fontFamily: F.mono, fontSize: 11, color: C.muted }}>
        <span>Poverty</span><span>{fmt(c.poverty_pct)}%</span>
      </div>
      <div style={{ background: C.darker, borderRadius: 3, height: 5, overflow: "hidden" }}>
        <div style={{ width: `${Math.min(100, c.od_rate / 3)}%`, height: "100%", background: tierColor }} />
      </div>
    </div>
  );
}

function StepPanel({ step, active, completed, title, children }) {
  return (
    <div style={{ opacity: completed && !active ? 0.55 : 1, marginBottom: 24, pointerEvents: completed && !active ? "none" : "auto" }}>
      <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 14 }}>
        <div style={{ width: 28, height: 28, borderRadius: "50%", background: active ? C.terra : completed ? C.sage : C.border2, color: "#fff", display: "flex", alignItems: "center", justifyContent: "center", fontFamily: F.mono, fontSize: 13, flexShrink: 0 }}>{step}</div>
        <h2 style={{ fontFamily: F.display, fontSize: 20, color: C.ink, margin: 0 }}>{title}</h2>
      </div>
      {children}
    </div>
  );
}

// ─── Math Accordion Data Builders ────────────────────────────────────────────
function buildMathAccordions(cands, selected, n, k) {
  const selCands = selected.map(i => cands[i]);
  const lambdas = selCands.map(c => c.od_rate);
  const baselineP = 0.0003;
  const avgEff = eff(0); // all selected counties have site at distance 0
  const baselineScore = cands.reduce((s, c) => s + (c.od_rate / 100000) * c.population, 0);
  const { mu, sigma, lo, hi } = cltCI(lambdas.length ? lambdas : [0]);

  // generate fake random scores for bootstrap
  const rng = seededRand(hashStr((selCands[0] || {}).county || "x"));
  const mu_score = scorePlacement(selected, cands);
  const randomScores = Array.from({ length: 1000 }, () => {
    const rs = Array.from({ length: k }, () => Math.floor(rng() * cands.length));
    const rs_unique = [...new Set(rs)];
    return scorePlacement(rs_unique, cands);
  });
  const pval = bootstrapPvalue(mu_score, randomScores);
  const countGood = randomScores.filter(s => s <= mu_score).length;
  const meanRandom = randomScores.reduce((a, b) => a + b, 0) / randomScores.length;

  // Build step showing calculations for ALL selected counties
  const allCountiesFormula = selCands.map(c =>
    `${c.county}: λ̂ = (${fmtInt(c.od_deaths)} / ${fmtInt(c.population)}) × 100,000 = ${fmt(c.od_rate)}`
  ).join("\n");

  const totalLambda = lambdas.reduce((sum, l) => sum + l, 0);

  return [
    {
      pill: "Poisson MLE", pillColor: C.terra, title: "Poisson MLE — Rate Estimation",
      description: "Overdose deaths are rare, independent events in a fixed time window — a Poisson process. MLE gives us the best single estimate of the rate λ̂ for each county.",
      liveValue: `Σλ̂ = ${fmt(totalLambda)}`,
      steps: [
        { label: "Selected counties", formula: selCands.map(c => `• ${c.county} — Population: ${fmtInt(c.population)}, OD deaths: ${fmtInt(c.od_deaths)}`).join("\n"), note: "These are your selected counties where sites will be placed." },
        { label: "Apply MLE formula for each county", formula: allCountiesFormula, note: "For a Poisson distribution, the MLE is simply the observed rate — maximizing the log-likelihood gives λ̂ = k." },
        { label: "Total λ̂ across all selected counties", formula: `Σλ̂ = ${lambdas.map(l => fmt(l)).join(" + ") || "—"}\n   = ${fmt(totalLambda)}`, note: "The sum of independent Poisson RVs is also Poisson — this total drives the CLT CI below." },
      ]
    },
    {
      pill: "C(n,k)", pillColor: C.gold, title: "Combinatorics — Search Space Size",
      description: "Order doesn't matter: placing a site in County A + County B is the same as B + A. C(n,k) tells us exactly how many distinct placements the brute-force optimizer checked.",
      liveValue: `C(${n},${k}) = ${fmtInt(combinatorics(n, k))}`,
      steps: [
        { label: "The formula", formula: `C(n, k) = n! / (k! × (n−k)!)\n         = n choose k`, note: "We divide by k! because order of selection doesn't change the placement result." },
        { label: "Plug in actual values", formula: `C(${n}, ${k}) = ${n}! / (${k}! × ${n - k}!)\n        = ${fmtInt(combinatorics(n, k))} distinct placements`, note: "The optimizer evaluated every single one of these combinations." },
        { label: "How the space grows (current n)", formula: `C(${n}, 3) = ${fmtInt(combinatorics(n, 3))}\nC(${n}, 5) = ${fmtInt(combinatorics(n, 5))}\nC(${n}, 8) = ${fmtInt(combinatorics(n, 8))}`, note: "Larger budgets dramatically expand the search — capping candidates at 15 keeps brute force tractable." },
      ]
    },
    {
      pill: "P(A|B)", pillColor: C.terra, title: "Conditional Probability — Site Effectiveness",
      description: "A site's presence changes P(death). Closer sites are more effective: full 40% risk reduction within 5 miles, decaying linearly to zero at 20 miles.",
      liveValue: `eff = ${fmt(avgEff * 100, 0)}%`,
      steps: [
        { label: "Baseline P(death per person)", formula: `baseline_p = 0.0003\n(~30 deaths per 100,000 residents, national avg)`, note: "This is the counterfactual: probability of fatal OD with no nearby site." },
        { label: "Effectiveness function", formula: `eff(d) = 0.40          if d ≤ 5 mi\n       = 0.40×(20-d)/15  if 5 < d ≤ 20 mi\n       = 0              if d > 20 mi\n\nP(death | site at d) = baseline_p × (1 − eff(d))`, note: "Linear decay models diminishing accessibility as travel distance increases." },
        { label: "Site effectiveness for your selected counties", formula: selCands.map(c => {
          const countyBaseline = c.od_rate / 100000;
          const withSite = countyBaseline * (1 - avgEff);
          const livesSaved = (countyBaseline - withSite) * (c.population / 100000) * 100000;
          return `${c.county}:\n  Baseline P = ${fmt(countyBaseline, 5)}\n  With site (d=0): P = ${fmt(withSite, 5)}\n  Lives saved/yr ≈ ${fmt(livesSaved, 1)}`;
        }).join("\n\n"), note: "Each selected county gets a site directly in it (d=0) — maximum 40% effectiveness." },
      ]
    },
    {
      pill: "Bootstrap", pillColor: C.terra, title: "Bootstrap p-value — Is This Better Than Random?",
      description: "We generated 1,000 random k-county placements and scored each. The p-value is the fraction that scored as well as your placement — small p means your choice was not luck.",
      liveValue: `p = ${pval.toFixed(3)}`,
      steps: [
        { label: "Your selected counties", formula: selCands.map(c => `• ${c.county} (OD rate: ${fmt(c.od_rate)})`).join("\n") + `\n\nYour score: ${fmt(mu_score, 1)} expected deaths`, note: "This is the placement we're testing against random chance." },
        { label: "Random baseline", formula: `1,000 random ${k}-county placements simulated\nMean random score: ${fmt(meanRandom, 1)} expected deaths`, note: "Random placements ignore OD rates — they serve as the null hypothesis." },
        { label: "Compute p-value", formula: `p = #{random ≤ your_score} / 1000\n  = ${countGood} / 1000\n  = ${pval.toFixed(3)}`, note: "Lower expected deaths = better placement. p counts how often random does equally well." },
        { label: "Conclusion", formula: pval < 0.05 ? `p = ${pval.toFixed(3)} < 0.05\n→ REJECT null hypothesis\n→ Your placement is significantly better than random` : `p = ${pval.toFixed(3)} ≥ 0.05\n→ Cannot reject null\n→ Consider the optimal placement`, note: pval < 0.05 ? "The optimizer's structure adds real value — this is not a chance result." : "The optimal placement performs significantly better." },
      ]
    },
    {
      pill: "CLT 95% CI", pillColor: C.sage, title: "CLT Confidence Interval — Uncertainty Around Expected Deaths",
      description: "The sum of independent Poisson RVs is Poisson. When the total rate is large, the CLT lets us approximate it as Normal and compute a confidence interval.",
      liveValue: `CI: [${fmt(Math.max(0, lo), 0)}, ${fmt(hi, 0)}]`,
      steps: [
        { label: "Selected counties' λ values", formula: selCands.map(c => `${c.county}: λ̂ = ${fmt(c.od_rate)}`).join("\n"), note: "Each county contributes an independent Poisson(λᵢ) random variable." },
        { label: "Sum of Poissons is Poisson", formula: `Σλᵢ = ${lambdas.map(l => fmt(l)).join(" + ") || "—"}\n   = ${fmt(mu)}`, note: "The sum of independent Poisson RVs is also Poisson(Σλᵢ)." },
        { label: "CLT: large Poisson ≈ Normal", formula: `μ = Σλᵢ = ${fmt(mu)}\nσ = √μ  = ${fmt(sigma)}\n(For Poisson: mean = variance = λ)`, note: "When μ > 30, Normal approximation is reliable — the CLT applies here." },
        { label: "95% Confidence Interval", formula: `CI = μ ± 1.96σ\n   = ${fmt(mu)} ± 1.96 × ${fmt(sigma)}\n   = (${fmt(Math.max(0, lo), 1)}, ${fmt(hi, 1)})`, note: `We are 95% confident total deaths across selected counties fall in this range.` },
      ]
    },
  ];
}

// ─── Expected Deaths Math Accordion Builder ──────────────────────────────────
function buildExpectedDeathsAccordion(cands, selected, totalScore) {
  const selectedSet = new Set(selected);
  const selCands = selected.map(i => cands[i]);
  const unselCands = cands.filter((_, i) => !selectedSet.has(i));

  const selectedContribs = selCands.map(c => {
    const p = c.od_rate / 100000;
    const withSite = conditionalProb(p, 0); // d=0 → full 40% effectiveness
    const deaths = withSite * c.population;
    const saved = p * c.population - deaths;
    return { c, deaths, saved };
  });

  const selTotal = selectedContribs.reduce((s, x) => s + x.deaths, 0);
  const unselTotal = unselCands.reduce((s, c) => s + (c.od_rate / 100000) * c.population, 0);

  return {
    pill: "E[Deaths]", pillColor: C.gold,
    title: "Expected Deaths — Placement Scoring",
    description: "For each county, expected deaths = P(death | site at distance d) × population. Counties with a site (d = 0) get the full 40% effectiveness reduction. All other counties have no site (d = 999 → eff = 0%).",
    liveValue: `${fmtInt(totalScore)} total`,
    steps: [
      {
        label: `Selected counties with site (d=0, eff=40%) — ${selCands.length} counti${selCands.length === 1 ? "y" : "es"}`,
        formula: selectedContribs.map(({ c, deaths, saved }) =>
          `${c.county}:\n  P = (${fmt(c.od_rate)}/100k) × (1 − 0.40) = ${fmt(c.od_rate * 0.6 / 100000, 6)}\n  Deaths = P × ${fmtInt(c.population)} pop = ${fmt(deaths, 1)}\n  Lives saved vs baseline: ${fmt(saved, 1)}`
        ).join("\n\n") + `\n\nSubtotal (selected): ${fmt(selTotal, 1)} expected deaths`,
        note: "Each selected county gets a needle-exchange site at d=0 miles — maximum 40% risk reduction."
      },
      {
        label: `Unselected counties (no site, full rate) — ${unselCands.length} counties`,
        formula: unselCands.slice(0, 8).map(c =>
          `${c.county}: (${fmt(c.od_rate)}/100k) × ${fmtInt(c.population)} pop = ${fmt((c.od_rate / 100000) * c.population, 1)}`
        ).join("\n") + (unselCands.length > 8 ? `\n… and ${unselCands.length - 8} more` : "") +
        `\n\nSubtotal (unselected): ${fmt(unselTotal, 1)} expected deaths`,
        note: "d=999 → eff=0 → no benefit for counties without a site."
      },
      {
        label: "Total expected deaths (the score)",
        formula: `Total = selected subtotal + unselected subtotal\n     = ${fmt(selTotal, 1)} + ${fmt(unselTotal, 1)}\n     = ${fmtInt(totalScore)}`,
        note: "The optimizer minimizes this total across all possible k-county placements. Lower = better."
      }
    ]
  };
}

// ─── Brute-Force Optimizer ───────────────────────────────────────────────────
function bruteForceOptimal(cands, k) {
  let bestScore = Infinity, bestIdxs = [];
  function combo(start, chosen) {
    if (chosen.length === k) {
      const s = scorePlacement(chosen, cands);
      if (s < bestScore) { bestScore = s; bestIdxs = [...chosen]; }
      return;
    }
    for (let i = start; i < cands.length; i++) combo(i + 1, [...chosen, i]);
  }
  combo(0, []);
  return { idxs: bestIdxs, score: bestScore };
}

// ─── Main App ─────────────────────────────────────────────────────────────────
export default function App() {
  const [allCounties, setAllCounties] = useState([]);
  const [states, setStates] = useState([]);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState("");
  const [selectedState, setSelectedState] = useState("");
  const [candidates, setCandidates] = useState([]);
  const [budget, setBudget] = useState(3);
  const [userPicks, setUserPicks] = useState([]);
  const [step, setStep] = useState(1);
  const [optimal, setOptimal] = useState(null);
  const [toast, setToast] = useState({ msg: "", visible: false, type: "info" });
  const toastTimer = useRef(null);

  // Load the bundled CSV once on mount
  useEffect(() => {
    fetch(CSV_PATH)
      .then(r => {
        if (!r.ok) throw new Error(`HTTP ${r.status} — is "${CSV_PATH}" in your public folder?`);
        return r.text();
      })
      .then(text => {
        const counties = parseAmfAR(text);
        if (!counties.length) throw new Error("CSV parsed but no valid county rows found. Check column names.");
        setAllCounties(counties);
        setStates([...new Set(counties.map(c => c.state))].sort());
        setLoading(false);
      })
      .catch(err => {
        setLoadError(err.message);
        setLoading(false);
      });
  }, []);

  function showToast(msg, type = "info") {
    clearTimeout(toastTimer.current);
    setToast({ msg, visible: true, type });
    toastTimer.current = setTimeout(() => setToast(t => ({ ...t, visible: false })), 3200);
  }

  function pickState(st) {
    setSelectedState(st);
    const sc = allCounties.filter(c => c.state === st);
    const cands = sampleCandidates(sc, st, 15);
    setCandidates(cands);
    setUserPicks([]);
    setBudget(3);
    setOptimal(null);
    showToast(`Selected ${st}: ${cands.length} candidate counties`, "gold");
    setStep(2);
  }

  function toggleCounty(i) {
    if (userPicks.includes(i)) {
      setUserPicks(p => p.filter(x => x !== i));
      showToast(`Removed ${candidates[i].county}`, "info");
    } else if (userPicks.length < budget) {
      const next = [...userPicks, i];
      setUserPicks(next);
      if (next.length === budget) showToast(`Budget full! ${budget} sites placed.`, "gold");
      else showToast(`Added ${candidates[i].county}`, "success");
    }
  }

  function runOptimal() {
    const result = bruteForceOptimal(candidates, budget);
    setOptimal(result);
    setStep(5);
  }

  const maxOD = candidates.length ? Math.max(...candidates.map(c => c.od_rate)) : 1;
  const minOD = candidates.length ? Math.min(...candidates.map(c => c.od_rate)) : 0;
  const tierOf = c => {
    const pct = (c.od_rate - minOD) / (maxOD - minOD || 1);
    return pct > 0.67 ? 0 : pct > 0.33 ? 1 : 2;
  };

  const baselineScore = candidates.reduce((s, c) => s + c.od_rate * (c.population / 100000), 0);
  const userScore = userPicks.length === budget ? scorePlacement(userPicks, candidates) : null;
  const optScore = optimal ? optimal.score : null;

  const userLivesSaved = userScore != null ? Math.max(0, baselineScore - userScore) : null;
  const optLivesSaved = optScore != null ? Math.max(0, baselineScore - optScore) : null;

  const sBox = { background: C.surface, border: `1px solid ${C.border1}`, borderRadius: 16, padding: "28px 32px", marginBottom: 24 };

  return (
    <div style={{ minHeight: "100vh", background: C.bg, fontFamily: F.body, color: C.ink }}>
      <Toast msg={toast.msg} type={toast.type} visible={toast.visible} />
      <div style={{ maxWidth: 860, margin: "0 auto", padding: "40px 20px" }}>
        {/* Header */}
        <div style={{ textAlign: "center", marginBottom: 40 }}>
          <div style={{ fontFamily: F.mono, fontSize: 11, color: C.muted, letterSpacing: 3, textTransform: "uppercase", marginBottom: 8 }}>Harm Reduction Planning Tool</div>
          <h1 style={{ fontFamily: F.display, fontSize: 38, color: C.ink, margin: "0 0 10px" }}>Needle Exchange Site Optimizer</h1>
          <p style={{ fontFamily: F.body, fontSize: 16, color: C.muted, margin: 0, maxWidth: 560, marginInline: "auto" }}>Select a state and find where harm-reduction sites save the most lives.</p>
        </div>

        {/* STEP 1 */}
        <div style={sBox}>
          <StepPanel step={1} active={step === 1} completed={step > 1} title="Select State">
            {loading ? (
              <div style={{ textAlign: "center", padding: "32px 0", fontFamily: F.mono, fontSize: 13, color: C.muted }}>
                <div style={{ fontSize: 28, marginBottom: 12 }}>⏳</div>
                Loading county data…
              </div>
            ) : loadError ? (
              <div style={{ background: C.terraBg, border: `1px solid ${C.terra}`, borderRadius: 10, padding: "20px 24px" }}>
                <div style={{ fontFamily: F.mono, fontSize: 12, color: C.terra, marginBottom: 6 }}>⚠ Could not load data file</div>
                <div style={{ fontFamily: F.body, fontSize: 14, color: C.ink, marginBottom: 8 }}>{loadError}</div>
                <div style={{ fontFamily: F.mono, fontSize: 12, color: C.muted }}>
                  Make sure <strong>{CSV_PATH}</strong> is in your project's <code>public/</code> folder and the filename matches exactly.
                </div>
              </div>
            ) : (
              <div>
                <div style={{ marginBottom: 14, fontFamily: F.body, color: C.sage, fontSize: 14 }}>✓ {allCounties.length.toLocaleString()} counties loaded across {states.length} states</div>
                <label style={{ fontFamily: F.mono, fontSize: 12, color: C.muted, display: "block", marginBottom: 6 }}>Select a State</label>
                <select value={selectedState} onChange={e => pickState(e.target.value)} style={{ width: "100%", padding: "10px 14px", borderRadius: 8, border: `1px solid ${C.border2}`, background: C.surface, fontFamily: F.body, fontSize: 15, color: C.ink }}>
                  <option value="">— Choose a state —</option>
                  {states.map(s => <option key={s} value={s}>{s}</option>)}
                </select>
                {candidates.length > 0 && (
                  <div style={{ marginTop: 16 }}>
                    <div style={{ fontFamily: F.mono, fontSize: 11, color: C.muted, marginBottom: 8 }}>CANDIDATE COUNTIES (rejection-sampled)</div>
                    <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
                      {candidates.map((c, i) => (
                        <span key={i} style={{ background: C.darker, border: `1px solid ${C.border1}`, borderRadius: 20, padding: "3px 10px", fontFamily: F.mono, fontSize: 11, color: C.ink }}>
                          {c.county} <span style={{ color: C.terra }}>{fmt(c.od_rate)}</span>
                        </span>
                      ))}
                    </div>
                    <p style={{ fontFamily: F.body, fontSize: 12, color: C.muted, marginTop: 10, fontStyle: "italic" }}>Rejection sampling selects candidates using each county's overdose rate as its probability weight — higher-risk counties are more likely to become candidates.</p>
                  </div>
                )}
              </div>
            )}
          </StepPanel>
        </div>

        {/* STEP 2 */}
        {step >= 2 && (
          <div style={sBox}>
            <StepPanel step={2} active={step === 2} completed={step > 2} title="Set Your Budget">
              <div style={{ textAlign: "center", marginBottom: 20 }}>
                <div style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: 20, marginBottom: 16 }}>
                  <button onClick={() => setBudget(b => Math.max(1, b - 1))} style={{ width: 40, height: 40, borderRadius: "50%", border: `2px solid ${C.terra}`, background: "none", color: C.terra, fontSize: 22, cursor: "pointer", fontFamily: F.display }}>−</button>
                  <span style={{ fontFamily: F.display, fontSize: 72, color: C.terra, lineHeight: 1 }}>{budget}</span>
                  <button onClick={() => setBudget(b => Math.min(10, b + 1))} style={{ width: 40, height: 40, borderRadius: "50%", border: `2px solid ${C.terra}`, background: "none", color: C.terra, fontSize: 22, cursor: "pointer", fontFamily: F.display }}>+</button>
                </div>
                <input type="range" min={1} max={Math.min(10, candidates.length)} value={budget} onChange={e => setBudget(+e.target.value)} style={{ width: "60%", accentColor: C.terra }} />
                <div style={{ display: "inline-flex", alignItems: "center", gap: 8, background: C.goldBg, border: `1px solid ${C.gold}`, borderRadius: 8, padding: "10px 18px", marginTop: 14 }}>
                  <span style={{ fontFamily: F.mono, fontSize: 13, color: C.gold }}>C({candidates.length}, {budget}) = {fmtInt(combinatorics(candidates.length, budget))} possible placements — the model checks every one.</span>
                </div>
              </div>
              {step === 2 && <button onClick={() => setStep(3)} style={{ background: C.terra, color: "#fff", border: "none", borderRadius: 8, padding: "12px 28px", fontFamily: F.mono, fontSize: 13, cursor: "pointer" }}>Choose Locations →</button>}
            </StepPanel>
          </div>
        )}

        {/* STEP 3 */}
        {step >= 3 && (
          <div style={sBox}>
            <StepPanel step={3} active={step === 3} completed={step > 3} title="Place Your Sites">
              <ProgressBar value={userPicks.length} max={budget} color={C.slate} />
              <div style={{ fontFamily: F.mono, fontSize: 12, color: C.muted, marginBottom: 14 }}>{userPicks.length} of {budget} sites placed</div>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(160px, 1fr))", gap: 10, marginBottom: 16 }}>
                {candidates.map((c, i) => (
                  <CountyCard key={i} c={c} selected={userPicks.includes(i)} locked={userPicks.length >= budget} riskTier={tierOf(c)} onClick={() => toggleCounty(i)} />
                ))}
              </div>
              {userPicks.length === budget && step === 3 && (
                <button onClick={() => setStep(4)} style={{ background: C.terra, color: "#fff", border: "none", borderRadius: 8, padding: "12px 28px", fontFamily: F.mono, fontSize: 13, cursor: "pointer" }}>See My Results →</button>
              )}
            </StepPanel>
          </div>
        )}

        {/* STEP 4 */}
        {step >= 4 && userScore !== null && (
          <div style={sBox}>
            <StepPanel step={4} active={step === 4} completed={step > 4} title="Your Results & Math">
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12, marginBottom: 20 }}>
                <ResultCard title="Lives Saved" value={fmtInt(userLivesSaved)} subtitle="vs. no sites" color={C.sage} bg={C.sageBg} span={2} />
                <ResultCard title="Baseline Expected Deaths" value={fmtInt(baselineScore)} subtitle="without any sites" color={C.terra} bg={C.terraBg} />
                <ResultCard title="Your Expected Deaths" value={fmtInt(userScore)} subtitle="with your placement" color={C.gold} bg={C.goldBg} />
              </div>
              <ScoreBar label="Baseline" score={baselineScore} baseline={baselineScore} color={C.terra} />
              <ScoreBar label="Your placement" score={userScore} baseline={baselineScore} color={C.gold} />
              <div style={{ marginTop: 20, marginBottom: 8, fontFamily: F.mono, fontSize: 11, color: C.muted, textTransform: "uppercase", letterSpacing: 1 }}>The Math Behind Your Placement</div>
              <MathAccordion {...buildExpectedDeathsAccordion(candidates, userPicks, userScore)} />
              {buildMathAccordions(candidates, userPicks, candidates.length, budget).map((a, i) => <MathAccordion key={i} {...a} />)}
              {step === 4 && <button onClick={runOptimal} style={{ marginTop: 16, background: C.gold, color: "#fff", border: "none", borderRadius: 8, padding: "12px 28px", fontFamily: F.mono, fontSize: 13, cursor: "pointer" }}>See the Optimal Answer →</button>}
            </StepPanel>
          </div>
        )}

        {/* STEP 5 */}
        {step >= 5 && optimal && (
          <div style={sBox}>
            <StepPanel step={5} active={step === 5} completed={false} title="Optimal Result & Math">
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12, marginBottom: 20 }}>
                <ResultCard title="★ Optimal Lives Saved" value={fmtInt(optLivesSaved)} subtitle={`Your choice: ${fmtInt(userLivesSaved)} — difference: ${fmtInt(optLivesSaved - userLivesSaved)}`} color={C.gold} bg={C.goldBg} span={2} />
                <ResultCard title="Baseline Expected Deaths" value={fmtInt(baselineScore)} subtitle="without any sites" color={C.terra} bg={C.terraBg} />
                <ResultCard title="Optimal Expected Deaths" value={fmtInt(optScore)} subtitle="best possible placement" color={C.sage} bg={C.sageBg} />
              </div>
              <div style={{ marginBottom: 12 }}>
                {optimal.idxs.map(i => (
                  <span key={i} style={{ display: "inline-block", margin: "4px 4px", background: C.goldBg, border: `2px solid ${C.gold}`, borderRadius: 20, padding: "4px 14px", fontFamily: F.mono, fontSize: 12, color: C.gold }}>★ {candidates[i].county}</span>
                ))}
              </div>
              <ScoreBar label="Baseline" score={baselineScore} baseline={baselineScore} color={C.terra} />
              <ScoreBar label="Your placement" score={userScore} baseline={baselineScore} color={C.slate} />
              <ScoreBar label="Optimal placement" score={optScore} baseline={baselineScore} color={C.gold} />
              <div style={{ marginTop: 20, marginBottom: 8, fontFamily: F.mono, fontSize: 11, color: C.muted, textTransform: "uppercase", letterSpacing: 1 }}>The Math Behind the Optimal Placement</div>
              <MathAccordion {...buildExpectedDeathsAccordion(candidates, optimal.idxs, optScore)} />
              {buildMathAccordions(candidates, optimal.idxs, candidates.length, budget).map((a, i) => <MathAccordion key={i} {...a} />)}
              <button onClick={() => { setSelectedState(""); setCandidates([]); setUserPicks([]); setOptimal(null); setStep(1); }} style={{ marginTop: 16, background: C.slate, color: "#fff", border: "none", borderRadius: 8, padding: "12px 28px", fontFamily: F.mono, fontSize: 13, cursor: "pointer" }}>Try a Different State →</button>
            </StepPanel>
          </div>
        )}
      </div>
    </div>
  );
}