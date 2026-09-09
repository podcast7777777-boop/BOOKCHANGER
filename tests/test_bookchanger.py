"""Test suite for bookchanger.py.

Covers:
  1. The INI helpers (parse/add/remove/replace entries, name candidates,
     duplicate detection, name spacing, phone validation, cutoff parsing,
     fuzzy name scoring).
  2. The documented read= directive builders (single-digit menus and the
     keyed phone-entry step).
  3. The keyed-phone flow end-to-end through the real endpoint handler
     (State 13 Add-save, State 37 Edit-save, re-key, garbage input,
     State 36 routing).
  4. The DTMF confirm states 12/35 (approve / re-record / replay), including
     a regression test that the confirm read= param is lowercase
     (name_key / newname_key) to match what the states read.
  5. Two full call simulations (Add and Edit) turn by turn.

Runs under pytest:
    python3 -m pytest tests/ -v

...or with ZERO installs via the fallback runner at the bottom:
    python3 tests/test_bookchanger.py

External dependencies (fastapi/pydub/speech_recognition/httpx) are stubbed
by tests/yhm_stubs.py when not installed; YHM REST calls are replaced by
tests/fake_yhm.py's in-memory double.
"""
from __future__ import annotations

import asyncio
import importlib.util
import inspect
import sys
from pathlib import Path

_TESTS_DIR = Path(__file__).resolve().parent
for _p in (str(_TESTS_DIR), str(_TESTS_DIR.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import yhm_stubs  # noqa: E402

STUB_STATUS = yhm_stubs.install()

from fake_yhm import FakeYHM  # noqa: E402


def _load_bookchanger():
    """Import bookchanger.py by file path (it has no package structure)."""
    module_path = _TESTS_DIR.parent / "bookchanger.py"
    spec = importlib.util.spec_from_file_location("bookchanger_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Under pytest, conftest.py already loaded the module; reuse THAT instance so
# the yhm fixture and these tests patch the same module object. Under the
# fallback runner (no conftest), load it here.
try:
    import bookchanger_under_test as bc  # noqa: E402  (provided by conftest)
except ModuleNotFoundError:
    bc = _load_bookchanger()

PlainTextResponse = bc.PlainTextResponse

# ─── Shared constants ───────────────────────────────────────────────────────

EXT = "100"
CALLER = "0500000000"
INI = "ivr2:/100/ListAllInformation.ini"

SAMPLE_INI = (
    "יוני כהן=0501234567\n"
    "מאיר יהודה הבר=0525551234\n"
    "0653969443=משפחת לוי\n"
)

PHONE_READ_TAIL = "=phone_entry,no,10,9,7,Phone"
PHONE_READ_PROMPT = "read=t-אנא הקישו את מספר הטלפון"


def run(coro):
    return asyncio.run(coro)


class FakeRequest:
    """Duck-typed stand-in for fastapi Request (GET with query params only)."""

    def __init__(self, params: dict):
        self.method = "GET"
        self.query_params = params

    async def form(self):  # pragma: no cover - only called on POST
        return {}


def base_params(**over) -> dict:
    params = {
        "ApiPhone": CALLER,
        "ApiCallId": "CALL1",
        "ApiExtension": EXT,
        "token": "TOK",
        "ini_path": "ListAllInformation.ini",
    }
    params.update(over)
    return params


def call_endpoint(bc, params: dict):
    """Invoke the /ivr handler directly (no HTTP layer)."""
    return asyncio.run(bc.yemot_ivr(FakeRequest(params)))


def make_capture(text: str):
    """Replacement for capture_latest_recording returning a fixed STT text."""

    async def _fake(ext, token):
        return (text, "000.wav")

    return _fake


def seed_state(yhm: FakeYHM, state: int, call_id: str = "CALL1") -> None:
    yhm.put(f"ivr2:/{EXT}/state_{CALLER}.tts", f"{call_id}\n{state}\n")


# ═══ 1. INI helpers ══════════════════════════════════════════════════════════

def test_parse_ini_line_canonical(bc):
    assert bc.parse_ini_line("יוני כהן=0501234567") == ("יוני כהן", "0501234567")


def test_parse_ini_line_reversed(bc):
    assert bc.parse_ini_line("0653969443=משפחת לוי") == ("משפחת לוי", "0653969443")


def test_parse_ini_line_garbage(bc):
    assert bc.parse_ini_line("no equals sign") is None
    assert bc.parse_ini_line("") is None
    assert bc.parse_ini_line("; comment") is None
    assert bc.parse_ini_line("left= ") is None
    assert bc.parse_ini_line(" =right") is None


def test_get_name_candidates_names_only(bc):
    names = bc.get_name_candidates(SAMPLE_INI)
    assert names == ["יוני כהן", "מאיר יהודה הבר", "משפחת לוי"]


def test_find_existing_entry_by_name(bc):
    assert bc.find_existing_entry_by_name(SAMPLE_INI, "יוני כהן") == ("יוני כהן", "0501234567")
    # whitespace-normalized exact match
    assert bc.find_existing_entry_by_name(SAMPLE_INI, "יוני  כהן") == ("יוני כהן", "0501234567")
    assert bc.find_existing_entry_by_name(SAMPLE_INI, "לא קיים") is None


def test_add_ini_entry(bc):
    out = bc.add_ini_entry(SAMPLE_INI, "שם חדש", "0531112223")
    assert out.endswith("שם חדש=0531112223\n")
    assert out.startswith(SAMPLE_INI)


def test_add_ini_entry_no_trailing_newline(bc):
    out = bc.add_ini_entry("a=0501234567", "b", "0521234567")
    assert out == "a=0501234567\nb=0521234567\n"


def test_remove_ini_entry_canonical_and_reversed(bc):
    out, removed = bc.remove_ini_entry(SAMPLE_INI, "יוני כהן", "0501234567")
    assert removed and "יוני כהן" not in out
    out2, removed2 = bc.remove_ini_entry(SAMPLE_INI, "משפחת לוי", "0653969443")
    assert removed2 and "0653969443" not in out2


def test_remove_ini_entry_missing(bc):
    out, removed = bc.remove_ini_entry(SAMPLE_INI, "אין כזה", "0500000000")
    assert not removed and out == SAMPLE_INI


def test_replace_ini_entry(bc):
    out, found = bc.replace_ini_entry(
        SAMPLE_INI, "יוני כהן", "0501234567", "יוני כהן חדש", "0539998887"
    )
    assert found
    assert "יוני כהן חדש=0539998887" in out
    assert "0501234567" not in out


def test_autocorrect_name_spacing(bc):
    assert bc.autocorrect_name_spacing("Yoni1") == "Yoni 1"
    assert bc.autocorrect_name_spacing("  מאיר   הבר ") == "מאיר הבר"


def test_normalize_and_validate_phone(bc):
    assert bc.normalize_and_validate_phone("0501234567") == "0501234567"
    assert bc.normalize_and_validate_phone("021234567") == "021234567"
    assert bc.normalize_and_validate_phone("05-0123-4567") == "0501234567"
    assert bc.normalize_and_validate_phone("12345") is None
    assert bc.normalize_and_validate_phone("05012345678") is None
    assert bc.normalize_and_validate_phone("אפס חמש שתיים") is None


def test_parse_percentage_param(bc):
    assert bc.parse_percentage_param("80", 60) == 80
    assert bc.parse_percentage_param("0.8", 60) == 80
    assert bc.parse_percentage_param("80,5", 60) == 80.5
    assert bc.parse_percentage_param("80%", 60) == 80
    assert bc.parse_percentage_param(None, 60) == 60
    assert bc.parse_percentage_param("abc", 60) == 60
    assert bc.parse_percentage_param("150", 60) == 100
    assert bc.parse_percentage_param("-5", 60) == 0


def test_compute_name_match_score_partial_search(bc):
    assert bc.compute_name_match_score("הבר", "מאיר יהודה הבר") >= 80
    full = bc.compute_name_match_score("מאיר הבר", "מאיר יהודה הבר")
    partial = bc.compute_name_match_score("מאיר כהן", "דב כהן")
    assert full > partial


def test_find_best_matches_filters_and_orders(bc):
    matches = bc.find_best_matches("הבר", ["מאיר יהודה הבר", "יוני כהן"], 80)
    assert matches == ["מאיר יהודה הבר"]


# ═══ 2. read= builders ═══════════════════════════════════════════════════════

def test_build_digit_read_menu_schema(bc):
    assert bc.build_digit_read("בחירה. סיום", "menu_key") == "read=t-בחירה, סיום=menu_key,no,1,1"


def test_build_digit_read_multi_digit(bc):
    out = bc.build_digit_read("הקש ספרות", "code", max_digits=4, min_digits=2)
    assert out.endswith("=code,no,4,2")


def test_old_tap_syntax_gone(bc):
    import re
    src = open(_TESTS_DIR.parent / "bookchanger.py", encoding="utf-8").read()
    assert re.search(r"read=t-[^=]+=[\w]+,tap", src) is None


def test_build_phone_entry_read(bc):
    assert bc.build_phone_entry_read("phone_entry") == (
        PHONE_READ_PROMPT
        + ", לאישור הקישו אחת, להקשה מחודשת הקישו שתיים"
        + PHONE_READ_TAIL
    )


def test_sanitize_tts_text_strips_dot_and_dash(bc):
    assert bc.sanitize_tts_text("א. ב") == "א, ב"
    assert bc.sanitize_tts_text("א-ב") == "א ב"


# ═══ 3. Keyed-phone flow (States 13 / 36 / 37) ═══════════════════════════════

def test_state13_add_save_writes_ini(bc, yhm):
    seed_state(yhm, 13)
    yhm.put(f"ivr2:/{EXT}/NAME_{CALLER}.tts", "שם חדש")
    resp = call_endpoint(bc, base_params(phone_entry="0539998887"))
    assert "נוספה בהצלחה" in resp.body
    new_ini = yhm.files[INI]
    assert "שם חדש=0539998887" in new_ini
    assert "יוני כהן=0501234567" in new_ini  # existing entries preserved
    assert yhm.files[f"ivr2:/{EXT}/state_{CALLER}.tts"].endswith("\n2\n")


def test_state13_rekey_digit_replays_read(bc, yhm):
    seed_state(yhm, 13)
    resp = call_endpoint(bc, base_params(phone_entry="2"))
    assert resp.body.startswith(PHONE_READ_PROMPT)
    assert resp.body.endswith(PHONE_READ_TAIL)
    assert yhm.files[f"ivr2:/{EXT}/state_{CALLER}.tts"].endswith("\n13\n")


def test_state13_garbage_input_asks_again(bc, yhm):
    seed_state(yhm, 13)
    resp = call_endpoint(bc, base_params(phone_entry="12345"))
    assert resp.body.startswith(PHONE_READ_PROMPT)


def test_state36_new_phone_routes_to_key_entry(bc, yhm):
    seed_state(yhm, 36)
    resp = call_endpoint(bc, base_params(edit_key3="2"))
    assert resp.body.startswith(PHONE_READ_PROMPT)
    assert resp.body.endswith(PHONE_READ_TAIL)
    assert yhm.files[f"ivr2:/{EXT}/state_{CALLER}.tts"].endswith("\n37\n")


def test_state37_edit_save_replaces_line(bc, yhm):
    seed_state(yhm, 37)
    yhm.put(f"ivr2:/{EXT}/ORIGINAL_NAME_{CALLER}.tts", "יוני כהן")
    yhm.put(f"ivr2:/{EXT}/ORIGINAL_PHONE_{CALLER}.tts", "0501234567")
    yhm.put(f"ivr2:/{EXT}/CURRENT_NAME_{CALLER}.tts", "יוני כהן")
    resp = call_endpoint(bc, base_params(phone_entry="0539998887"))
    assert "עודכנה בהצלחה" in resp.body
    new_ini = yhm.files[INI]
    assert "יוני כהן=0539998887" in new_ini
    assert "0501234567" not in new_ini


# ═══ 4. DTMF confirmations (States 12 / 35) ══════════════════════════════════

def test_ask_dtmf_confirm_emits_lowercase_param(bc, yhm):
    # REGRESSION: the emitter and the state reader must agree on the param
    # name -- ask_dtmf_confirm lowercases the tts_key ("NAME" -> "name_key").
    resp = run(bc.ask_dtmf_confirm(EXT, CALLER, "CALL1", "TOK", "NAME", "יוני כהן", 12))
    assert resp.body.startswith(f"id_list_message=s-/{EXT}/NAME_{CALLER}&read=t-")
    assert resp.body.endswith("=name_key,no,1,1")
    assert yhm.files[f"ivr2:/{EXT}/NAME_{CALLER}.tts"] == "יוני כהן"


def test_state12_approve_advances_to_keyed_phone(bc, yhm):
    seed_state(yhm, 12)
    yhm.put(f"ivr2:/{EXT}/NAME_{CALLER}.tts", "שם חדש")
    resp = call_endpoint(bc, base_params(name_key="1"))
    assert resp.body.startswith(PHONE_READ_PROMPT)
    assert resp.body.endswith(PHONE_READ_TAIL)
    assert yhm.files[f"ivr2:/{EXT}/state_{CALLER}.tts"].endswith("\n13\n")


def test_state12_duplicate_name_rejected(bc, yhm):
    seed_state(yhm, 12)
    yhm.put(f"ivr2:/{EXT}/NAME_{CALLER}.tts", "יוני כהן")  # already in sample ini
    resp = call_endpoint(bc, base_params(name_key="1"))
    assert "השם הזה כבר קיים ברשימה" in resp.body


def test_state12_reject_resets_to_name_recording(bc, yhm):
    seed_state(yhm, 12)
    yhm.put(f"ivr2:/{EXT}/NAME_{CALLER}.tts", "שם גרוע")
    resp = call_endpoint(bc, base_params(name_key="2"))
    assert "go_to_folder=/100/1" in resp.body
    assert "אנא אמרו את השם להוספה" in resp.body
    assert any(p.endswith(f"NAME_{CALLER}.tts") for p in yhm.deleted)


def test_state12_other_digit_replays_without_discarding(bc, yhm):
    seed_state(yhm, 12)
    yhm.put(f"ivr2:/{EXT}/NAME_{CALLER}.tts", "שם טוב")
    resp = call_endpoint(bc, base_params(name_key="5"))
    assert resp.body.endswith("=name_key,no,1,1")
    assert f"NAME_{CALLER}" in resp.body  # announcement still played
    assert yhm.files[f"ivr2:/{EXT}/state_{CALLER}.tts"].endswith("\n12\n")


def test_state35_approve_updates_current_name(bc, yhm):
    seed_state(yhm, 35)
    yhm.put(f"ivr2:/{EXT}/NEWNAME_{CALLER}.tts", "מאיר הבר חדש")
    yhm.put(f"ivr2:/{EXT}/ORIGINAL_NAME_{CALLER}.tts", "משפחת לוי")
    yhm.put(f"ivr2:/{EXT}/ORIGINAL_PHONE_{CALLER}.tts", "0653969443")
    resp = call_endpoint(bc, base_params(newname_key="1"))
    assert yhm.files[f"ivr2:/{EXT}/CURRENT_NAME_{CALLER}.tts"] == "מאיר הבר חדש"
    assert resp.body.endswith("=edit_key3,no,1,1")
    assert yhm.files[f"ivr2:/{EXT}/state_{CALLER}.tts"].endswith("\n36\n")


def test_state35_duplicate_name_rejected(bc, yhm):
    seed_state(yhm, 35)
    yhm.put(f"ivr2:/{EXT}/NEWNAME_{CALLER}.tts", "יוני כהן")  # exists, != original
    yhm.put(f"ivr2:/{EXT}/ORIGINAL_NAME_{CALLER}.tts", "משפחת לוי")
    yhm.put(f"ivr2:/{EXT}/ORIGINAL_PHONE_{CALLER}.tts", "0653969443")
    resp = call_endpoint(bc, base_params(newname_key="1"))
    assert "השם הזה כבר קיים ברשימה" in resp.body


def test_state35_reject_resets_to_new_name_recording(bc, yhm):
    seed_state(yhm, 35)
    resp = call_endpoint(bc, base_params(newname_key="2"))
    assert "go_to_folder=/100/1" in resp.body
    assert "הקליטו את השם החדש" in resp.body


def test_state35_other_digit_replays(bc, yhm):
    seed_state(yhm, 35)
    resp = call_endpoint(bc, base_params(newname_key="9"))
    assert resp.body.endswith("=newname_key,no,1,1")


# ═══ 5. Endpoint basics + full call simulations ══════════════════════════════

def test_hangup_notification_ignored(bc, yhm):
    resp = call_endpoint(bc, base_params(hangup="yes"))
    assert resp.body == ""


def test_main_menu_read_on_new_call(bc, yhm):
    resp = call_endpoint(bc, base_params())
    assert resp.body.endswith("=menu_key,no,1,1")


def test_new_call_resets_stale_state(bc, yhm):
    yhm.put(f"ivr2:/{EXT}/state_{CALLER}.tts", "OLDCALL\n13\n")
    resp = call_endpoint(bc, base_params())
    assert resp.body.endswith("=menu_key,no,1,1")


def test_main_menu_invalid_digit_replays(bc, yhm):
    # The "invalid choice" prefix belongs to State 2 (menu digit processing);
    # a fresh call with no state file always replays a clean menu.
    seed_state(yhm, 2)
    resp = call_endpoint(bc, base_params(menu_key="9"))
    assert "בחירה לא תקינה" in resp.body
    assert resp.body.endswith("=menu_key,no,1,1")


def test_full_add_flow_via_keypad(bc, yhm):
    yhm.put(INI, SAMPLE_INI)
    original_capture = bc.capture_latest_recording
    try:
        # Turn 1: caller dials in -> main menu read=
        resp = call_endpoint(bc, base_params())
        assert resp.body.endswith("=menu_key,no,1,1")
        # Turn 2: presses 1 (add) -> routed to recording folder for the name
        resp = call_endpoint(bc, base_params(menu_key="1"))
        assert "go_to_folder=/100/1" in resp.body
        # Turn 3: recording landed; STT recognizes the name -> DTMF confirm
        bc.capture_latest_recording = make_capture("יוני כהן 2")
        resp = call_endpoint(bc, base_params())
        assert resp.body.endswith("=name_key,no,1,1")
        # Turn 4: presses 1 (approve) -> keyed phone entry read=
        resp = call_endpoint(bc, base_params(name_key="1"))
        assert resp.body.endswith(PHONE_READ_TAIL)
        # Turn 5: keys the digits -> entry saved, back to main menu
        resp = call_endpoint(bc, base_params(phone_entry="0539998887"))
        assert "נוספה בהצלחה" in resp.body
        assert "יוני כהן 2=0539998887" in yhm.files[INI]
    finally:
        bc.capture_latest_recording = original_capture


def test_full_edit_flow_via_keypad(bc, yhm):
    yhm.put(INI, SAMPLE_INI)
    original_capture = bc.capture_latest_recording
    try:
        # Turn 1: dial in -> menu
        resp = call_endpoint(bc, base_params())
        assert resp.body.endswith("=menu_key,no,1,1")
        # Turn 2: presses 3 (edit) -> record the search name
        resp = call_endpoint(bc, base_params(menu_key="3"))
        assert "go_to_folder=/100/1" in resp.body
        # Turn 3: says "הבר" -> found record menu
        bc.capture_latest_recording = make_capture("הבר")
        resp = call_endpoint(bc, base_params())
        assert resp.body.endswith("=edit_key1,no,1,1")
        # Turn 4: presses 1 (edit this record) -> name keep/new menu
        resp = call_endpoint(bc, base_params(edit_key1="1"))
        assert resp.body.endswith("=edit_key2,no,1,1")
        # Turn 5: presses 2 (record a new name) -> routed to recording folder
        resp = call_endpoint(bc, base_params(edit_key2="2"))
        assert "go_to_folder=/100/1" in resp.body
        # Turn 6: new name recognized -> DTMF confirm
        bc.capture_latest_recording = make_capture("מאיר הבר חדש")
        resp = call_endpoint(bc, base_params())
        assert resp.body.endswith("=newname_key,no,1,1")
        # Turn 7: approves -> phone keep/new menu
        resp = call_endpoint(bc, base_params(newname_key="1"))
        assert resp.body.endswith("=edit_key3,no,1,1")
        # Turn 8: presses 2 (key a new phone) -> keyed phone entry
        resp = call_endpoint(bc, base_params(edit_key3="2"))
        assert resp.body.endswith(PHONE_READ_TAIL)
        # Turn 9: keys the digits -> record updated in the ini
        resp = call_endpoint(bc, base_params(phone_entry="0539998887"))
        assert "עודכנה בהצלחה" in resp.body
        new_ini = yhm.files[INI]
        assert "מאיר הבר חדש=0539998887" in new_ini
        assert "מאיר יהודה הבר=0525551234" not in new_ini
    finally:
        bc.capture_latest_recording = original_capture


# ═══ Zero-install fallback runner ════════════════════════════════════════════
# Runs every test_* function without pytest, supplying the same fixtures
# conftest.py would (bc + a fresh, wired FakeYHM per test).

if __name__ == "__main__":
    if "bookchanger_under_test" not in sys.modules:
        # Not run under pytest: conftest never loaded, so load the module here.
        bc = _load_bookchanger()

    def _fresh_fake():
        fake = FakeYHM()
        fake.put(INI, SAMPLE_INI)
        bc.yhm_read_text_file = fake.read_text_file
        bc.yhm_write_text_file = fake.write_text_file
        bc.yhm_delete_files = fake.delete_files
        return fake

    failures = []
    ran = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        kwargs = {}
        sig = inspect.signature(fn)
        if "bc" in sig.parameters:
            kwargs["bc"] = bc
        if "yhm" in sig.parameters:
            kwargs["yhm"] = _fresh_fake()
        if "ini_path" in sig.parameters:
            kwargs["ini_path"] = INI
        try:
            fn(**kwargs)
            ran += 1
            print(f"PASS {name}")
        except Exception as exc:  # noqa: BLE001
            failures.append(name)
            print(f"FAIL {name}: {exc!r}")
    print(f"\n{ran} passed, {len(failures)} failed | stubs: {STUB_STATUS}")
    sys.exit(1 if failures else 0)
