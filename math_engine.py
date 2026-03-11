"""
math_engine.py
==============
Python backend for the Needle Exchange Site Placement Optimizer.
"""

import math
import random

# Poisson Maximum Likelihood Estimation
# 
# What it is:
#   A Poisson process models the number of independent events that
#   occur in a fixed time window given a constant average rate λ (lambda).
#   Maximum Likelihood Estimation (MLE) finds the parameter value that
#   makes the observed data most probable; for the Poisson distribution
#   that estimator is the sample mean.
#
# Why use it here:
#   Lambda-hat gives us a normalized, comparable "risk rate" for every county.
#
# Formula:
#   The Poisson PMF for k events with rate λ is: P(X = k | λ) = (e^{-λ} * λ^k) / k!
#   Log-likelihood over one observation k: LL(λ) = -λ + k·log(λ) - log(k!)
#   Taking the derivative and setting to zero: = -1 + k/λ = 0  ->  λ̂ = k
#   Because λ is the expected count PER 100,000 residents (not raw count), we scale:
#   λ̂ = (od_deaths / population) × 100,000
#
# Example (Cabell County, WV):
#   od_deaths  = 178 (approximate annual count at ~185/100k rate)
#   population = 96,319
#   Step 1 raw rate: 178 / 96,319 = 0.001848 deaths per resident
#   Step 2 scale to per-100k: 0.001848 × 100,000 = 184.9 ≈ 185.2 deaths per 100,000
#
def poisson_mle(od_deaths: float, population: float) -> float:
    if population <= 0:
        raise ValueError("population must be positive")
    return (od_deaths / population) * 100_000


# Combinatorics
# 
# What it is:
#   A combination C(n, k) counts the number of ways to choose k items
#   from n distinct items when order does not matter, written "n choose k."
#
# Why we use it here:
#   Placing a site in County A and one in County B is identical to
#   placing one in B then one in A, so we count placements as combinations, 
#   not permutations. This tells the optimizer exactly how large the search 
#   space is and confirms that brute-force enumeration is feasible.
#
# Formula:
#   C(n, k) = n! / (k! × (n−k)!)
#       n! orders all n items.
#       Divide by k! to collapse orderings within the chosen group.
#       Divide by (n−k)! to collapse orderings within the unchosen group.
#
# Inputs:
#   n: total number of candidate counties (up to 15 in this tool)
#   k: budget (number of sites to place)
#
# Returns:
#   The exact count of distinct placement combinations the optimizer must evaluate.
#
def combinatorics(n: int, k: int) -> int:
    return math.comb(n, k)


# Conditional Probability
# 
# What it is:
#   Conditional probability P(A | B) is the probability of event A
#   given that event B has already occurred.  The definition is:
#   P(A | B) = P(A ∩ B) / P(B)
#   Here: A = "overdose death" and B = "a harm-reduction site exists 
#   at distance d miles from the county."
#
# Why we use it here:
#   A site available changes the probability of a fatal overdose.
#   The closer the site, the more people can reach it so 
#   P(death | site nearby) < P(death | no site).  Modeling this as 
#   conditional probability lets us quantify exactly how much risk 
#   drops as a function of distance.
#
# Formula:
#   effectiveness(d):
#       0.40                         if d ≤ 5   (full effect, very close)
#       0.40 × (20 − d) / 15        if 5 < d ≤ 20  (linear decay)
#       0                            if d > 20   (beyond service radius)
#
#   P(death | site at distance d) = baseline_p × (1 − effectiveness(d))
#
# Inputs:
#   baseline_p: P(death) with no nearby site (raw OD rate ÷ 100k). 
#   distance_miles: straight-line distance from county centroid to nearest site.
#
# Returns:
#   Returns the conditional probability of a fatal overdose per person
#   given a site at distance_miles.  Subtract from baseline_p to obtain
#   the per-person lives-saved estimate.
#
# Example (Cabell County, WV):
#   baseline_p = 0.00185  (λ̂ ≈ 185.2 per 100k)
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
# 
def _effectiveness(distance_miles: float) -> float:
    if distance_miles <= 5:
        return 0.40
    elif distance_miles <= 20:
        return 0.40 * (20 - distance_miles) / 15
    else:
        return 0.0


def conditional_prob(baseline_p: float, distance_miles: float) -> float:
    eff = _effectiveness(distance_miles)
    return baseline_p * (1.0 - eff)


# Bootstrap Resampling & the p-value
# 
# What it is:
#   Bootstrapping is a method that constructs a sampling distribution 
#   by repeatedly resampling from observed data (or generating random 
#   alternatives). It is especially useful when the underlying distribution
#   is unknown or the sample size is too small to invoke the CLT.
#
# Why we use it here:
#   We want to know: "Could a random placement have done as well as the
#   optimizer's chosen placement by chance?" Because the distribution of
#   random-placement scores has no closed-form formula (it depends on the
#   particular counties selected), we generate 1,000 random placements,
#   score each one, and ask what fraction scored as well as or better than
#   the optimal which is the p-value.
#
# Formula:
#   p = #{s ∈ random_scores : s ≤ optimal_score} / N
#   where N = len(random_scores).
#
# Inputs:
#   optimal_score: the expected-deaths score for the chosen placement (lower is better).
#   random_scores: 1,000 scores from randomly drawn k-county placements.
#
# Returns:
#   Returns p as a float in [0, 1].  Small p means few random placements
#   are as good and the optimizer's result is not due to luck.
#
# Example:
#   Suppose:
#     optimal_score  = 142.7   (optimizer found this placement)
#     random_scores  = [list of 1,000 values, mean ≈ 281.4]
#     #{s ≤ 142.7}   = 12
#     p = 12 / 1000 = 0.012
#   
#   Interpretation: Only 1.2% of random placements matched or beat the
#   optimizer. Since p < 0.05 we reject the null hypothesis that the
#   optimizer's result could have been achieved by chance.
# 
def bootstrap_pvalue(optimal_score: float, random_scores: list) -> float:
    if not random_scores:
        raise ValueError("random_scores must be a non-empty list")
    
    count_better_or_equal = 0
    
    for score in random_scores:
        if score <= optimal_score:
            count_better_or_equal = count_better_or_equal + 1
            
    total_scores = len(random_scores)
    p_value = count_better_or_equal / total_scores
    
    return p_value


# Central Limit Theorem & Confidence Intervals
# 
# What it is:
#   The Central Limit Theorem (CLT) states that the sum (or mean) of a
#   large number of independent, identically distributed random variables
#   converges in distribution to a Normal distribution, regardless of the
#   original distribution's shape.  For a Poisson(λ) variable, E[X] = λ
#   and Var(X) = λ.
#
# Why we use it here:
#   If we place k sites, the total expected overdose deaths across those k
#   counties is the sum of k independent Poisson(λᵢ) random variables.
#   The sum of independent Poissons is itself Poisson(Σλᵢ), and when
#   Σλᵢ is large (typically > 30), the CLT guarantees that sum is well
#   approximated by Normal(μ = Σλᵢ, σ² = Σλᵢ).  This lets us build a
#   95% confidence interval around the placement's total expected deaths.
#
# Formula:
#   Given k counties with rates λ₁, λ₂, …, λₖ (per 100k per year):
#   μ = Σλᵢ (total Poisson rate = mean of the sum)
#   σ = √μ (std dev, because variance = mean for Poisson)
#   CI95 = (μ − 1.96·σ,  μ + 1.96·σ)
#
# Inputs:
#   lambda_list: per-county λ̂ values for all selected counties (output of poisson_mle for each county).
#   z: z-score for desired confidence level (default 1.96 for 95%; use 2.576 for 99%).
#
# Returns:
#   Returns a tuple (mu, sigma, ci_low, ci_high) where:
#     mu: total expected deaths across selected counties
#     sigma: standard deviation of that total
#     ci_low: lower bound of 95% CI (can be clipped to 0 in practice)
#     ci_high: upper bound of 95% CI
#
def clt_confidence_interval(lambda_list: list, z: float = 1.96) -> tuple:
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