# ai_module.py
import os

# Try to use sklearn for a small DecisionTree classifier (optional)
try:
    from sklearn.tree import DecisionTreeClassifier
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

# Denominations available (canonical Indian set)
DENOMINATIONS_FULL = [2000, 500, 200, 100, 50, 20, 10, 5, 2, 1]

# Simple synthetic data to "train" a strategy predictor if sklearn present
# features: [amount, use_case_index] -> labels: 0 (few_large), 1 (balanced), 2 (many_small)
SYNTHETIC_X = [
    [100, 0], [200, 0], [500, 0], [1200, 0],  # ATM -> few_large
    [150, 1], [600, 1], [2500, 1],            # E-commerce -> balanced
    [50, 2], [120, 2], [500, 2], [700, 2]     # Donation -> many_small
]
SYNTHETIC_Y = [0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2]

# Map use_case string to index
USE_CASE_MAP = {
    "atm": 0,
    "ecommerce": 1,
    "donation": 2
}

# Train a tiny DecisionTree if sklearn installed
clf = None
if SKLEARN_AVAILABLE:
    try:
        clf = DecisionTreeClassifier(max_depth=4, random_state=42)
        clf.fit(SYNTHETIC_X, SYNTHETIC_Y)
    except Exception:
        clf = None


def _coin_change_dp(amount, denoms, max_amount_safe=200000):
    """
    Return a combination (denom -> count) that exactly sums to amount using
    the provided denoms (list of ints) using DP minimizing number of coins.
    Returns None if no exact combination or amount too large for DP (safety).
    """
    if amount < 0:
        return None
    if amount == 0:
        return {}

    if amount > max_amount_safe:
        # DP would be heavy — skip and return None (caller will fallback)
        return None

    denoms = sorted(set(int(d) for d in denoms if int(d) > 0), reverse=False)
    if not denoms:
        return None

    INF = 10**9
    dp = [INF] * (amount + 1)
    parent = [-1] * (amount + 1)
    dp[0] = 0

    for i in range(1, amount + 1):
        for coin in denoms:
            if coin <= i and dp[i - coin] + 1 < dp[i]:
                dp[i] = dp[i - coin] + 1
                parent[i] = coin

    if dp[amount] == INF:
        return None

    # reconstruct
    cur = amount
    counts = {}
    while cur > 0:
        coin = parent[cur]
        if coin == -1:
            return None
        counts[coin] = counts.get(coin, 0) + 1
        cur -= coin

    return counts


def _greedy_allocation(amount, ordered_denoms):
    """Greedy allocation in the provided order. Returns (result_dict, remainder)"""
    remaining = amount
    res = {}
    for d in ordered_denoms:
        if remaining <= 0:
            break
        if d <= 0:
            continue
        c = remaining // d
        if c > 0:
            res[int(d)] = int(c)
            remaining -= d * c
    return res, remaining


def smart_denomination(amount, selected_denoms=None, use_case='atm', allow_approximate=True):
    """
    Smart denomination optimizer:
    - amount: int
    - selected_denoms: list/iterable of available denominations (ints). If None, uses full set.
    - use_case: 'atm' | 'ecommerce' | 'donation' (influence allocation strategy)
    - allow_approximate: if True, returns greedy approx when exact match impossible.
    Returns:
      dict: {denom: count}
      OR if error: {'error': '...'}
      OR if approximate: {'denominations': {...}, 'note': 'approximate'}  (we use plain dict but include 'note' key)
    """
    # Validate amount
    try:
        amount = int(amount)
    except (TypeError, ValueError):
        return {"error": "Invalid amount. Please enter a number."}

    if amount <= 0:
        return {"error": "Amount must be greater than zero."}

    # prepare selected denominations
    if not selected_denoms:
        denoms = DENOMINATIONS_FULL.copy()
    else:
        try:
            denoms = sorted({int(x) for x in selected_denoms if int(x) > 0}, reverse=True)
            if not denoms:
                denoms = DENOMINATIONS_FULL.copy()
        except Exception:
            denoms = DENOMINATIONS_FULL.copy()

    # map use_case to index
    use_case_key = use_case.lower() if isinstance(use_case, str) else 'atm'
    use_case_idx = USE_CASE_MAP.get(use_case_key, 0)

    # If sklearn available and classifier trained, use it to pick strategy
    strategy = None
    if clf is not None:
        try:
            label = clf.predict([[amount, use_case_idx]])[0]
            if label == 0:
                strategy = 'few_large'
            elif label == 1:
                strategy = 'balanced'
            else:
                strategy = 'many_small'
        except Exception:
            strategy = None

    # fallback deterministic mapping if no clf
    if strategy is None:
        if use_case_idx == 0:
            strategy = 'few_large'
        elif use_case_idx == 1:
            strategy = 'balanced'
        else:
            strategy = 'many_small'

    # Build an ordered list based on strategy and available denoms
    denoms_desc = sorted(denoms, reverse=True)
    denoms_asc = sorted(denoms)

    if strategy == 'few_large':
        order = denoms_desc
    elif strategy == 'many_small':
        order = denoms_asc
    else:  # balanced -> interleave large and small to create a mixed allocation
        order = []
        ld = denoms_desc[:]
        sd = denoms_asc[:]
        while ld or sd:
            if ld:
                order.append(ld.pop(0))
            if sd:
                order.append(sd.pop(0))
        # remove duplicates preserving order
        seen = set()
        order = [x for x in order if not (x in seen or seen.add(x))]

    # Try greedy according to order
    greedy_res, rem = _greedy_allocation(amount, order)

    if rem == 0:
        # exact by greedy — return plain dict
        return {int(k): int(v) for k, v in greedy_res.items()}

    # If greedy didn't give exact, try DP using full selected denom set (best-effort)
    dp_denoms = sorted(denoms, reverse=True)
    dp_out = _coin_change_dp(amount, dp_denoms)
    if dp_out is not None:
        # success exact
        return {int(k): int(v) for k, v in dp_out.items()}

    # If DP failed (maybe amount too big or impossible with chosen denoms)
    if allow_approximate:
        # return greedy result and a note that it's approximate
        res = {int(k): int(v) for k, v in greedy_res.items()}
        # include remainder as special "Remainder" pseudo-denom if > 0
        if rem > 0:
            res['remainder'] = int(rem)
        # attach a note string to signal approximate (caller can check)
        # We return a dict with denominations and a reserved key '__note__' to indicate approximate.
        res_with_note = {"denominations": res, "note": "approximate - exact match not possible with selected denominations"}
        return res_with_note

    # else exact required and not possible
    return {"error": "Cannot create exact denomination breakdown with the selected denominations."}
# ai_module.py
"""
Smart denomination optimizer with use-case aware strategy and exact coin-change solver.

Usage:
    from ai_module import smart_denomination
    smart_denomination(amount, selected_denoms, use_case)

- amount: int
- selected_denoms: iterable of denominations (strings or ints) from the form; if None or empty uses full set
- use_case: 'atm'|'ecommerce'|'donation' (case-insensitive). Defaults to 'atm'

The function always tries to return an exact breakdown. If selected_denoms cannot produce an exact
sum it will fall back to the full denomination set (including 1) to guarantee exact solution.
"""

from typing import Iterable, Dict, Optional
import math

# Canonical denominations (Indian) - highest to smallest
DENOMINATIONS_FULL = [2000, 1000, 500, 200, 100, 50, 20, 10, 5, 2, 1]


def _parse_denoms(selected_denoms: Optional[Iterable]) -> list:
    """Parse input denominations list (strings/ints) -> sorted unique list descending."""
    if not selected_denoms:
        return sorted(DENOMINATIONS_FULL, reverse=True)
    parsed = []
    for d in selected_denoms:
        try:
            di = int(d)
            if di > 0:
                parsed.append(di)
        except Exception:
            continue
    parsed = sorted(set(parsed), reverse=True)
    if not parsed:
        return sorted(DENOMINATIONS_FULL, reverse=True)
    return parsed


def _cost_for_denom(denom: int, strategy: str, max_den: int) -> float:
    """
    Return a cost associated to using one coin/note of `denom`.
    DP will minimize total cost. Lower cost -> preferred.
    - For 'atm' we prefer large notes -> lower cost for larger denoms.
    - For 'donation' prefer smaller notes -> lower cost for smaller denoms.
    - For 'ecommerce' balanced -> moderate preference.
    """
    # Normalized size: larger denom -> value close to 1, small denom -> close to 0.
    size_score = denom / max_den if max_den > 0 else 1.0

    if strategy == 'atm':
        # prefer large: cost decreases as size_score increases
        cost = 1.0 - 0.7 * size_score  # large denom -> much lower cost
    elif strategy == 'donation':
        # prefer small: cost decreases as size_score decreases
        cost = 1.0 - 0.7 * (1.0 - size_score)  # small denom -> lower cost
    else:  # balanced / ecommerce
        # moderate preference: slight bias toward medium denominations
        # cost = 1 - 0.4 * (size_score - 0.5)^2  (lower for mid-range)
        cost = 1.0 - 0.6 * (1.0 - abs(size_score - 0.5))
    # ensure cost positive
    return max(0.01, float(cost))


def _coin_change_min_cost(amount: int, denoms: list, max_amount_safe: int = 200000):
    """
    Dynamic programming to compute exact change minimizing total weighted cost.
    - amount: target sum
    - denoms: list of denominations (ints), can be unsorted
    Returns: dict {denom: count} if exact solution found, else None

    Safety: if amount > max_amount_safe, return None to avoid heavy DP.
    """
    if amount < 0:
        return None
    if amount == 0:
        return {}

    if amount > max_amount_safe:
        return None

    denoms = sorted(set(int(d) for d in denoms if int(d) > 0))
    if not denoms:
        return None

    max_den = max(denoms)
    # Prepare cost per denom according to strategy encoded in denoms order by outside logic;
    # Here we assume cost_map already aligned; but we will pass strategy separately in wrapper.

    # We'll compute a DP minimizing total cost (float). Use list of length amount+1.
    INF = float('inf')
    dp_cost = [INF] * (amount + 1)
    dp_last = [-1] * (amount + 1)
    dp_cost[0] = 0.0

    # Precompute cost per denom externally — but we cannot here (lack of strategy param).
    # So callers should supply adjusted denoms_with_cost (list of tuples (denom, cost)).
    return None  # this helper is replaced by wrapper below


def _coin_change_with_cost(amount: int, denoms_with_cost: list, max_amount_safe: int = 200000):
    """
    DP over denoms_with_cost: list of (denom:int, cost:float)
    Minimizes sum(cost) for exact amount. Returns dict denom->count or None.
    """
    if amount < 0:
        return None
    if amount == 0:
        return {}
    if amount > max_amount_safe:
        return None

    denoms = sorted(set(int(d) for d, _ in denoms_with_cost if int(d) > 0))
    if not denoms:
        return None

    # Map denom -> cost (use smallest cost if duplicate supplied)
    cost_map = {}
    for d, c in denoms_with_cost:
        di = int(d)
        cost_map[di] = min(cost_map.get(di, float('inf')), float(c))

    denoms = sorted(cost_map.keys())
    max_denom = max(denoms)

    INF = float('inf')
    dp_cost = [INF] * (amount + 1)
    dp_choice = [-1] * (amount + 1)
    dp_cost[0] = 0.0

    for s in range(1, amount + 1):
        best_cost = INF
        best_coin = -1
        for coin in denoms:
            if coin <= s:
                prev = dp_cost[s - coin]
                if prev != INF:
                    cur_cost = prev + cost_map[coin]
                    # break ties by favoring fewer coins: add tiny epsilon proportional to coin value negative
                    # but dp minimizes cost only
                    if cur_cost < best_cost:
                        best_cost = cur_cost
                        best_coin = coin
        dp_cost[s] = best_cost
        dp_choice[s] = best_coin

    if dp_cost[amount] == INF:
        return None

    # reconstruct counts
    s = amount
    counts = {}
    while s > 0:
        coin = dp_choice[s]
        if coin <= 0:
            return None
        counts[coin] = counts.get(coin, 0) + 1
        s -= coin
    return counts


def smart_denomination(amount,
                       selected_denoms: Optional[Iterable] = None,
                       use_case: str = 'atm',
                       allow_approximate: bool = True) -> Dict:
    """
    Smart denomination optimizer.
    - amount: int
    - selected_denoms: iterable of denoms (strings/ints) selected by user
    - use_case: 'atm'|'ecommerce'|'donation'
    - allow_approximate: if True and exact not possible, returns best approximate with 'remainder' key

    Returns:
      - plain dict {denom: count} OR
      - {"error": "..."} on invalid input OR
      - {"denominations": {...}, "note": "..."} for approximate wrapper
    """

    # Validate amount
    try:
        amount = int(amount)
    except (TypeError, ValueError):
        return {"error": "Invalid amount. Please enter a numeric amount."}
    if amount < 0:
        return {"error": "Amount must be non-negative."}
    if amount == 0:
        return {}

    # Parse selected denominations; if empty -> use full set
    denoms = _parse_denoms(selected_denoms)
    # If the user supplied just one denomination via older code path (e.g., 'denomination' key), it may not be a list
    # _parse_denoms handles that.

    # Normalize use_case
    strategy = (use_case or 'atm').strip().lower()
    if strategy not in ('atm', 'ecommerce', 'donation'):
        strategy = 'atm'

    # Prepare denoms_with_cost according to strategy
    max_d = max(denoms) if denoms else max(DENOMINATIONS_FULL)
    denoms_with_cost = []
    for d in denoms:
        c = _cost_for_denom(d, strategy, max_d)
        denoms_with_cost.append((d, c))

    # 1) Try exact DP with selected denoms + strategy cost
    dp_result = _coin_change_with_cost(amount, denoms_with_cost)
    if dp_result is not None:
        # success
        return {int(k): int(v) for k, v in sorted(dp_result.items(), reverse=True)}

    # 2) If exact failed, try expanding to the full canonical set (including 1) so exact solution is always possible
    # Build cost mapping for full set but still respect strategy preference
    full_with_cost = []
    full_max = max(DENOMINATIONS_FULL)
    for d in sorted(set(DENOMINATIONS_FULL), reverse=False):
        c = _cost_for_denom(d, strategy, full_max)
        full_with_cost.append((d, c))

    dp_full = _coin_change_with_cost(amount, full_with_cost)
    if dp_full is not None:
        return {int(k): int(v) for k, v in sorted(dp_full.items(), reverse=True)}

    # 3) If still no exact result (very large amount beyond DP safety), fallback to greedy approximate
    # greedy over strategy-ordered denoms (desc or asc or interleaved)
    if strategy == 'atm':
        order = sorted(set(denoms), reverse=True)
    elif strategy == 'donation':
        order = sorted(set(denoms), reverse=False)
    else:  # ecommerce balanced
        # create a balanced order interleaving big and small
        desc = sorted(set(denoms), reverse=True)
        asc = sorted(set(denoms))
        order = []
        i = 0
        while i < max(len(desc), len(asc)):
            if i < len(desc):
                order.append(desc[i])
            if i < len(asc):
                order.append(asc[-1 - i])
            i += 1
        # remove duplicates while keeping order
        seen = set()
        ordered = []
        for x in order:
            if x not in seen:
                seen.add(x)
                ordered.append(x)
        order = ordered

    remaining = amount
    res = {}
    for d in order:
        if remaining <= 0:
            break
        cnt = remaining // d
        if cnt > 0:
            res[d] = cnt
            remaining -= d * cnt

    # If remainder > 0 and 1 is not in selected denoms, include 1 (from full set) to cover remainder
    if remaining > 0:
        if 1 not in denoms:
            # use 1s to cover remainder
            res[1] = res.get(1, 0) + remaining
            remaining = 0
        else:
            # if 1 already included, greedy above would have used 1; just put remainder as remainder
            pass

    if remaining == 0:
        # approximate (greedy) result but exact (because we added 1) — return plain dict
        return {int(k): int(v) for k, v in sorted(res.items(), reverse=True)}

    # If we still couldn't match exactly, return an approximate wrapper with remainder
    approx = {int(k): int(v) for k, v in sorted(res.items(), reverse=True)}
    if remaining > 0:
        approx['remainder'] = int(remaining)
    return {"denominations": approx, "note": "approximate - exact DP not feasible for very large amount"}
