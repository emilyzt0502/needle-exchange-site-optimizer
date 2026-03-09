"""
math_engine.py
==============
Pure Python backend for the Needle Exchange Site Placement Optimizer.
No file I/O, no data loading, no side effects — just math.

Each function demonstrates one probability/statistics concept used to
decide WHERE to place harm-reduction (syringe-service) sites across
counties in a US state.

Run this file directly to verify all functions:
    python math_engine.py
"""

import math
import random


# ════════════════════════════════════════════════════════
# CONCEPT: Poisson Maximum Likelihood Estimation (MLE)
# ════════════════════════════════════════════════════════
# WHAT IT IS:
#   A Poisson process models the number of rare, independent events that
#   occur in a fixed time window given a constant average rate λ (lambda).
#   Maximum Likelihood Estimation (MLE) finds the parameter value that
#   makes the observed data most probable; for the Poisson distribution
#   that estimator is simply the sample mean.
#
# WHY WE USE IT HERE:
#   Overdose deaths in a county in a single year are rare relative to the
#   population, arrive (roughly) independently of one another, and are
#   counted over a fixed 12-month window — the classic Poisson setup.
#   λ̂ gives us a normalized, comparable "risk rate" for every county.
#
# THE FORMULA:
#   The Poisson PMF for k events with rate λ is:
#       P(X = k | λ) = (e^{-λ} * λ^k) / k!
#
#   Log-likelihood over one observation k:
#       ℓ(λ) = -λ + k·ln(λ) - ln(k!)
#
#   Taking the derivative and setting to zero:
#       dℓ/dλ = -1 + k/λ = 0  →  λ̂ = k
#
#   Because λ is the expected count PER 100,000 residents (not raw count),
#   we scale:
#       λ̂ = (od_deaths / population) × 100,000
#
# INPUTS:
#   od_deaths   — int or float: observed overdose deaths in the county
#                 for the most recent available year
#   population  — int or float: county resident population (same year)
#
# OUTPUT:
#   Returns λ̂ as a float: estimated overdose deaths per 100,000 residents.
#   Higher λ̂ means higher baseline risk; sites placed there reduce more
#   expected deaths per dollar spent.
#
# WORKED EXAMPLE (Cabell County, WV):
#   od_deaths  = 178          (approximate annual count at ~185/100k rate)
#   population = 96,319
#
#   Step 1 — raw rate:
#       178 / 96,319 = 0.001848 deaths per resident
#
#   Step 2 — scale to per-100k:
#       0.001848 × 100,000 = 184.9 ≈ 185.2 deaths per 100,000
#
#   Interpretation: A planner allocating a single site should seriously
#   consider Cabell County — its λ̂ ≈ 185 is roughly 10× the national
#   average, signaling extremely elevated baseline risk.
# ════════════════════════════════════════════════════════
def poisson_mle(od_deaths: float, population: float) -> float:
    """
    Compute the Poisson MLE rate estimate λ̂ (deaths per 100,000 residents).

    Args:
        od_deaths:  Observed overdose death count.
        population: County resident population.

    Returns:
        λ̂ = (od_deaths / population) × 100,000
    """
    if population <= 0:
        raise ValueError("population must be positive")
    return (od_deaths / population) * 100_000


# ════════════════════════════════════════════════════════
# CONCEPT: Combinatorics — Counting Unordered Selections
# ════════════════════════════════════════════════════════
# WHAT IT IS:
#   A combination C(n, k) counts the number of ways to choose k items
#   from n distinct items when ORDER DOES NOT MATTER.  It differs from
#   a permutation, which counts ordered arrangements.  The formula is
#   C(n,k) = n! / (k! × (n−k)!), sometimes written "n choose k."
#
# WHY WE USE IT HERE:
#   Placing a site in County A and one in County B is identical to
#   placing one in B then one in A — the physical result is the same.
#   So we count placements as *combinations*, not permutations.  This
#   tells the optimizer exactly how large the search space is and
#   confirms that brute-force enumeration is feasible (C(15,5) = 3,003)
#   without the need for heuristic shortcuts.
#
# THE FORMULA:
#   C(n, k) = n! / (k! × (n−k)!)
#
#   Derivation sketch:
#     n! orders all n items.
#     Divide by k! to collapse orderings within the chosen group.
#     Divide by (n−k)! to collapse orderings within the unchosen group.
#
# INPUTS:
#   n — int: total number of candidate counties (up to 15 in this tool)
#   k — int: budget (number of sites to place)
#
# OUTPUT:
#   Returns C(n, k) as an integer — the exact count of distinct placement
#   combinations the optimizer must evaluate.
#
# WORKED EXAMPLE — search-space growth table:
#   With n = 15 candidate counties:
#
#   Budget k | C(15, k)  | Meaning
#   ---------|-----------|----------------------------------------
#       3    |     455   | 455 distinct 3-site placements
#       5    |   3,003   | brute-force scans 3,003 combos (~instant)
#       8    |   6,435   | still fully enumerable in milliseconds
#
#   If n were 50 counties and k = 10:
#       C(50, 10) = 10,272,278,170 — no longer brute-forceable!
#   Capping candidates at 15 is deliberate: it keeps the search tractable.
# ════════════════════════════════════════════════════════
def combinatorics(n: int, k: int) -> int:
    """
    Compute the binomial coefficient C(n, k) — unordered selections.

    Args:
        n: Pool size (number of candidate counties).
        k: Selection size (budget / number of sites to place).

    Returns:
        C(n, k) as an integer.
    """
    return math.comb(n, k)


# ════════════════════════════════════════════════════════
# CONCEPT: Conditional Probability
# ════════════════════════════════════════════════════════
# WHAT IT IS:
#   Conditional probability P(A | B) is the probability of event A
#   given that event B has already occurred.  The definition is:
#       P(A | B) = P(A ∩ B) / P(B)
#   Here we use a structured model: A = "overdose death" and B = "a
#   harm-reduction site exists at distance d miles from the county."
#
# WHY WE USE IT HERE:
#   A site's presence changes the probability of a fatal overdose.
#   The closer the site, the more people can reach it for clean supplies,
#   naloxone, and treatment referrals — so P(death | site nearby) <
#   P(death | no site).  Modeling this as conditional probability lets
#   us quantify exactly how much risk drops as a function of distance.
#
# THE FORMULA:
#   effectiveness(d):
#       0.40                         if d ≤ 5   (full effect, very close)
#       0.40 × (20 − d) / 15        if 5 < d ≤ 20  (linear decay)
#       0                            if d > 20   (beyond service radius)
#
#   P(death | site at distance d) = baseline_p × (1 − effectiveness(d))
#
# INPUTS:
#   baseline_p     — float: P(death) with NO nearby site (raw OD rate ÷ 100k).
#                    Typical value: 0.0003 for a mid-risk county.
#   distance_miles — float: straight-line distance from county centroid
#                    to nearest site.
#
# OUTPUT:
#   Returns the conditional probability of a fatal overdose per person
#   given a site at distance_miles.  Subtract from baseline_p to obtain
#   the per-person lives-saved estimate.
#
# WORKED EXAMPLE:
#   baseline_p = 0.00185  (Cabell County WV, λ̂ ≈ 185.2 per 100k)
#
#   At d = 0 miles  (site IS in the county):
#       eff = 0.40
#       P(death | 0 mi) = 0.00185 × (1 − 0.40) = 0.00111
#       Risk reduced by 40% → ~74 lives saved per 100,000 residents/yr
#
#   At d = 10 miles (site one county over):
#       eff = 0.40 × (20 − 10) / 15 = 0.40 × 0.667 = 0.267
#       P(death | 10 mi) = 0.00185 × (1 − 0.267) = 0.001356
#       Risk reduced by ~27%
#
#   At d = 25 miles (outside service radius):
#       eff = 0  (beyond 20-mile cap)
#       P(death | 25 mi) = 0.00185 × 1.0 = 0.00185  (no benefit)
# ════════════════════════════════════════════════════════
def _effectiveness(distance_miles: float) -> float:
    """
    Compute the effectiveness of a site at a given distance.
    Internal helper used by conditional_prob().
    """
    if distance_miles <= 5:
        return 0.40
    elif distance_miles <= 20:
        return 0.40 * (20 - distance_miles) / 15
    else:
        return 0.0


def conditional_prob(baseline_p: float, distance_miles: float) -> float:
    """
    Compute P(death | site at distance_miles) using the effectiveness model.

    Args:
        baseline_p:     Baseline probability of overdose death per person
                        (OD rate / 100,000).
        distance_miles: Distance from county centroid to nearest site.

    Returns:
        Conditional probability of death given site at that distance.
    """
    eff = _effectiveness(distance_miles)
    return baseline_p * (1.0 - eff)


# ════════════════════════════════════════════════════════
# CONCEPT: Bootstrap Resampling & the p-value
# ════════════════════════════════════════════════════════
# WHAT IT IS:
#   Bootstrapping is a non-parametric method that constructs an empirical
#   sampling distribution by repeatedly resampling from observed data (or
#   generating random alternatives) rather than assuming a theoretical
#   distribution.  It is especially useful when the underlying distribution
#   is unknown or the sample size is too small to invoke the CLT directly.
#
# WHY WE USE IT HERE:
#   We want to know: "Could a random placement have done as well as the
#   optimizer's chosen placement by chance?"  Because the distribution of
#   random-placement scores has no closed-form formula (it depends on the
#   particular counties selected), we generate 1,000 random placements,
#   score each one, and ask what fraction scored as well as or better than
#   the optimal — that fraction IS the p-value.
#
# THE FORMULA:
#   p = #{s ∈ random_scores : s ≤ optimal_score} / N
#
#   where N = len(random_scores).
#
#   Note: "≤ optimal_score" because our score is EXPECTED DEATHS (lower =
#   better); a random placement "as good as" ours means its score is also
#   low, i.e. ≤ optimal_score.
#
# INPUTS:
#   optimal_score  — float: the expected-deaths score for the chosen
#                    placement (lower is better).
#   random_scores  — list of float: 1,000 scores from randomly drawn
#                    k-county placements.
#
# OUTPUT:
#   Returns p as a float in [0, 1].  Small p means few random placements
#   are as good — the optimizer's result is not due to luck.
#
# WORKED EXAMPLE:
#   Suppose:
#     optimal_score  = 142.7   (optimizer found this placement)
#     random_scores  = [list of 1,000 values, mean ≈ 281.4]
#     #{s ≤ 142.7}   = 12
#
#   p = 12 / 1000 = 0.012
#
#   Interpretation: Only 1.2% of random placements matched or beat the
#   optimizer.  Since p < 0.05 we REJECT the null hypothesis that the
#   optimizer's result could have been achieved by chance, confirming
#   that the structured site-selection algorithm adds real value.
# ════════════════════════════════════════════════════════
def bootstrap_pvalue(optimal_score: float, random_scores: list) -> float:
    """
    Compute the bootstrap p-value for the optimizer's placement.

    Args:
        optimal_score:  Expected-deaths score of the chosen placement
                        (lower = better).
        random_scores:  List of scores from 1,000 random placements.

    Returns:
        p-value: fraction of random scores ≤ optimal_score.
    """
    if not random_scores:
        raise ValueError("random_scores must be a non-empty list")
    count_as_good = sum(1 for s in random_scores if s <= optimal_score)
    return count_as_good / len(random_scores)


# ════════════════════════════════════════════════════════
# CONCEPT: Central Limit Theorem & Confidence Intervals
# ════════════════════════════════════════════════════════
# WHAT IT IS:
#   The Central Limit Theorem (CLT) states that the sum (or mean) of a
#   large number of independent, identically distributed random variables
#   converges in distribution to a Normal distribution, regardless of the
#   original distribution's shape.  For a Poisson(λ) variable, E[X] = λ
#   and Var(X) = λ — a unique property where mean equals variance.
#
# WHY WE USE IT HERE:
#   If we place k sites, the total expected overdose deaths across those k
#   counties is the SUM of k independent Poisson(λᵢ) random variables.
#   The sum of independent Poissons is itself Poisson(Σλᵢ) — and when
#   Σλᵢ is large (typically > 30), the CLT guarantees that sum is well
#   approximated by Normal(μ = Σλᵢ, σ² = Σλᵢ).  This lets us build a
#   95% confidence interval around the placement's total expected deaths.
#
# THE FORMULA:
#   Given k counties with rates λ₁, λ₂, …, λₖ (per 100k per year):
#
#   μ     = Σλᵢ           (total Poisson rate = mean of the sum)
#   σ     = √μ            (std dev, because variance = mean for Poisson)
#   CI₉₅  = (μ − 1.96·σ,  μ + 1.96·σ)
#
# INPUTS:
#   lambda_list — list of float: per-county λ̂ values for all selected
#                 counties (output of poisson_mle for each county).
#   z           — float: z-score for desired confidence level (default
#                 1.96 for 95%; use 2.576 for 99%).
#
# OUTPUT:
#   Returns a tuple (mu, sigma, ci_low, ci_high) where:
#     mu       — total expected deaths across selected counties
#     sigma    — standard deviation of that total
#     ci_low   — lower bound of 95% CI (can be clipped to 0 in practice)
#     ci_high  — upper bound of 95% CI
#
# WORKED EXAMPLE (5-county placement in WV):
#   lambda_list = [185.2, 94.3, 67.8, 112.5, 43.1]
#     (Cabell, Wayne, Mingo, McDowell, Logan counties — illustrative)
#
#   μ    = 185.2 + 94.3 + 67.8 + 112.5 + 43.1 = 502.9
#   σ    = √502.9 ≈ 22.43
#   CI₉₅ = (502.9 − 1.96 × 22.43,  502.9 + 1.96 × 22.43)
#         = (502.9 − 43.96,  502.9 + 43.96)
#         = (458.9,  546.9)
#
#   A planner can report: "Our 5-site placement is expected to see
#   ~503 overdose deaths this year (95% CI: 459–547)."
# ════════════════════════════════════════════════════════
def clt_confidence_interval(lambda_list: list, z: float = 1.96) -> tuple:
    """
    Compute sum-of-Poissons confidence interval via the CLT.

    Args:
        lambda_list: Per-county λ̂ values (deaths per 100k).
        z:           Z-score for the desired confidence level (default 1.96).

    Returns:
        (mu, sigma, ci_low, ci_high) as floats.
    """
    if not lambda_list:
        raise ValueError("lambda_list must be non-empty")
    mu = sum(lambda_list)
    sigma = math.sqrt(mu)
    ci_low = mu - z * sigma
    ci_high = mu + z * sigma
    return (mu, sigma, ci_low, ci_high)


# ════════════════════════════════════════════════════════
# Standalone verification
# ════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("math_engine.py — Standalone Verification")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Poisson MLE
    # ------------------------------------------------------------------
    print("\n[1] POISSON MLE — Cabell County, WV")
    lam = poisson_mle(od_deaths=178.5, population=96_319)
    print(f"    od_deaths=178.5, population=96,319")
    print(f"    λ̂ = {lam:.2f} deaths per 100,000 residents/year")
    # Second example for variety
    lam2 = poisson_mle(od_deaths=45, population=250_000)
    print(f"    [also] od_deaths=45, population=250,000 → λ̂ = {lam2:.2f}")

    # ------------------------------------------------------------------
    # 2. Combinatorics
    # ------------------------------------------------------------------
    print("\n[2] COMBINATORICS — search-space size")
    for k in [3, 5, 8]:
        c = combinatorics(15, k)
        print(f"    C(15, {k}) = {c:,}")
    print(f"    C(15, 5) = {combinatorics(15, 5):,}  ← brute-force feasible")

    # ------------------------------------------------------------------
    # 3. Conditional Probability
    # ------------------------------------------------------------------
    print("\n[3] CONDITIONAL PROBABILITY — Cabell County baseline")
    bp = 0.00185   # λ̂ ≈ 185.2 / 100,000
    for d in [0, 5, 10, 15, 20, 25]:
        p = conditional_prob(bp, d)
        eff = _effectiveness(d)
        print(f"    d={d:>2} mi → eff={eff:.3f}, P(death|site)={p:.6f}")

    # ------------------------------------------------------------------
    # 4. Bootstrap p-value
    # ------------------------------------------------------------------
    print("\n[4] BOOTSTRAP p-VALUE")
    rng = random.Random(42)
    # Simulate 1,000 random placements with scores centered around 280
    random_scores = [rng.gauss(281.4, 30) for _ in range(1_000)]
    optimal = 142.7
    pval = bootstrap_pvalue(optimal, random_scores)
    print(f"    optimal_score = {optimal}")
    print(f"    mean(random_scores) ≈ {sum(random_scores)/len(random_scores):.1f}")
    print(f"    p = {pval:.4f}  {'(significant, p < 0.05)' if pval < 0.05 else '(not significant)'}")

    # ------------------------------------------------------------------
    # 5. CLT Confidence Interval
    # ------------------------------------------------------------------
    print("\n[5] CLT CONFIDENCE INTERVAL — 5-county WV placement")
    lambdas = [185.2, 94.3, 67.8, 112.5, 43.1]   # illustrative WV counties
    mu, sigma, lo, hi = clt_confidence_interval(lambdas)
    print(f"    λ list = {lambdas}")
    print(f"    μ      = {mu:.1f}")
    print(f"    σ      = {sigma:.2f}")
    print(f"    95% CI = ({lo:.1f}, {hi:.1f})")

    print("\n" + "=" * 60)
    print("All 5 functions executed without errors. ✓")
    print("=" * 60)