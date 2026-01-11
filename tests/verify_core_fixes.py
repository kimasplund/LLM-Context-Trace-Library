
import pytest
from pathlib import Path
from lctl.core.events import Chain

def test_chain_validation():
    # Test 1: Missing 'chain' key BUT has 'events' key (Should PASS as legacy/fallback)
    c = Chain.from_dict({"lctl": "4.0", "events": []})
    assert c.id == "unknown"

    # Test 1b: Missing BOTH 'chain' and 'events' (Should FAIL)
    with pytest.raises(ValueError, match="missing 'chain' or 'events' keys"):
        Chain.from_dict({"lctl": "4.0"})

    # Test 2: 'chain' is not a dict
    with pytest.raises(ValueError, match="missing 'chain' or 'events' keys"):
        Chain.from_dict({"chain": "not-a-dict"})

    # Test 3: Valid modern format
    c = Chain.from_dict({"chain": {"id": "test"}, "events": []})
    assert c.id == "test"

    # Test 4: Valid legacy format (has events, no chain dict)
    c = Chain.from_dict({"events": []})
    assert c.id == "unknown"
    
    # Test 5: Load empty file (should fail)
    bad_file = Path("empty.json")
    bad_file.write_text("")
    with pytest.raises(ValueError, match="Chain file is empty"):
        Chain.load(bad_file)
    bad_file.unlink()

    # Test 6: Load non-dict json (should fail)
    bad_file = Path("list.json")
    bad_file.write_text("[]")
    with pytest.raises(ValueError, match="Chain file must contain a dictionary"):
        Chain.load(bad_file)
    bad_file.unlink()

    print("All core validation tests passed!")

if __name__ == "__main__":
    # We can run this with python directly if pytest is installed, or just import it in a pytest run
    # Let's just run the function
    try:
        test_chain_validation()
    except Exception as e:
        print(f"FAILED: {e}")
        exit(1)
