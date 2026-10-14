# LCTL Bug Report

## Summary

Code review of LCTL core modules identified **16 bugs and issues** across 3 files. All issues have been fixed.

---

## lctl/core/events.py (5 issues fixed)

### 1. Event.from_dict - Missing required field validation
**Severity:** High
**Issue:** No validation for required fields (seq, type, timestamp, agent). Missing fields would cause KeyError with unclear error message.
**Fix:** Added validation that raises `ValueError` with clear message listing missing fields.

### 2. Event.from_dict - Poor timestamp error handling
**Severity:** Medium
**Issue:** Invalid timestamp format would raise generic `ValueError` without context.
**Fix:** Added try/catch with descriptive error message including the invalid value.

### 3. Chain.load - No exception handling
**Severity:** High
**Issue:** No handling for:
- File not found
- Permission errors
- Empty files
- Invalid JSON/YAML syntax
- Non-dict content (e.g., arrays, null)

**Fix:** Added comprehensive error handling with clear error messages for each case.

### 4. Chain.load - File existence check redundant with click.Path
**Severity:** Low
**Issue:** Click's `exists=True` doesn't provide useful error messages.
**Fix:** Removed reliance on click validation, added explicit checks with better messages.

### 5. ReplayEngine.replay_to - Shallow copy state mutation bug
**Severity:** High
**Issue:** State cache was copied using shallow `dict()`, causing mutation of cached states when replaying.
**Fix:** Changed to `copy.deepcopy()` for facts, metrics, and errors.

---

## lctl/core/session.py (5 issues fixed)

### 6. LCTLSession.__exit__ - Silent failure on error recording
**Severity:** Medium
**Issue:** If `self.error()` raised an exception, the original exception could be lost.
**Fix:** Wrapped error recording in try/except and explicitly return False.

### 7. LCTLSession._now - Deprecated datetime.utcnow()
**Severity:** Low
**Issue:** `datetime.utcnow()` is deprecated since Python 3.12.
**Fix:** Changed to `datetime.now(timezone.utc)`.

### 8. LCTLSession.export - No error handling
**Severity:** Medium
**Issue:** No handling for:
- Parent directory doesn't exist
- Permission denied
- Disk full/IO errors

**Fix:** Added validation and error handling with clear messages.

### 9. traced_step - Tracing failures suppress user exceptions
**Severity:** High
**Issue:** If `step_start()` or `step_end()` threw an exception, user code exceptions could be lost.
**Fix:** Wrapped all tracing calls in try/except to ensure user exceptions are always propagated.

### 10. traced_step - Missing cleanup on partial trace
**Severity:** Medium
**Issue:** If tracing started but failed mid-way, inconsistent state could result.
**Fix:** Gracefully degrade - if tracing fails, continue without tracing.

---

## lctl/cli/main.py (6 issues fixed)

### 11. All commands - Inconsistent error handling
**Severity:** High
**Issue:** Commands used `click.Path(exists=True)` but underlying `Chain.load()` exceptions were not caught, leading to stack traces instead of user-friendly errors.
**Fix:** Created `_load_chain_safely()` helper that handles all exceptions and prints user-friendly error messages. Changed all `click.Path(exists=True)` to `click.Path()` and use the helper.

### 12. replay command - None text crash
**Severity:** Medium
**Issue:** `fact['text'][:60]` would crash if text is None.
**Fix:** Added safe handling: `text = fact.get('text', '') or ''`

### 13. bottleneck command - Empty list access
**Severity:** Medium
**Issue:** `bottlenecks[0]` accessed without checking if list is empty after slicing.
**Fix:** Added `if bottlenecks and` before accessing `bottlenecks[0]`.

### 14. _print_event - None data handling
**Severity:** Medium
**Issue:** Event data fields could be None, causing string operations to fail.
**Fix:** Added safe defaults: `event.data.get('text', '') or ''`

### 15. _estimate_cost - No input validation
**Severity:** Low
**Issue:** None or negative token counts would cause incorrect results.
**Fix:** Added `tokens_in = max(0, tokens_in or 0)` handling.

### 16. _print_event, _estimate_cost - Missing type hints
**Severity:** Low
**Issue:** Helper functions lacked type hints.
**Fix:** Added `event: Event` and `-> None`, `-> float` type hints.

---

## Files Modified

1. `/home/kim/projects/llm-context-transfer/lctl/core/events.py`
2. `/home/kim/projects/llm-context-transfer/lctl/core/session.py`
3. `/home/kim/projects/llm-context-transfer/lctl/cli/main.py`

---

## Testing Recommendations

1. Test `Event.from_dict` with missing fields
2. Test `Chain.load` with non-existent, empty, malformed files
3. Test `traced_step` when tracing itself throws
4. Test CLI commands with invalid file paths
5. Test replay with facts containing None text
6. Test bottleneck with chains that have no step timing data
