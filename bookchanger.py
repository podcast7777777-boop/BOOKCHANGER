# ═══════════════════════════════════════════════════════════════════════════
# bookchanger.py — Voice/DTMF Phonebook Manager for Yemot Hamashiach (YHM)
# ═══════════════════════════════════════════════════════════════════════════
#
# WHAT THIS IS
# ------------
# A `type=api` webhook that lets a caller add, delete, or edit entries in an
# INI phonebook (Name=Phone) entirely by phone: a DTMF main menu, voice
# recording + Google STT for NAMES, and DTMF keypad entry for PHONE NUMBERS.
# Every confirmation in the system is DTMF (1 = approve / 2 = re-enter):
# names via ask_dtmf_confirm's own read= directive, keyed phone numbers via
# YHM's built-in approve/re-enter menu. State is stateless-HTTP-safe: it is
# NOT kept in memory or on local disk (Render's filesystem doesn't persist
# across requests/deploys) -- it lives entirely on the YHM server itself, as
# small .tts text files addressed by phone number, read/written through the
# same call2all.co.il REST API main.py already uses.
#
# WHAT'S REUSED VERBATIM FROM main.py (imports through find_best_matches,
# i.e. everything up to the "REUSED SECTION ENDS HERE" marker below):
# path helpers, the word/token-level fuzzy name-matching engine
# (compute_name_match_score / find_best_matches), percentage-param parsing,
# every YHM REST call (GetTextFile/UploadTextFile/GetIVR2Dir/DownloadFile/
# FileAction), the Google STT pipeline (1s silence padding, he-IL), the
# "find the highest numbered NNN.wav" + "delete numeric WAVs immediately and
# unconditionally after download" discipline (this is the fix your Bug #1
# changelog describes -- it's followed just as strictly here), and
# save_caller_state's CallId/State/last_processed_file file format.
# main.py's own changelog comments are left in place below, since they
# explain *why* several of these functions are written the way they are.
#
# WHAT'S NEW (added after the marker): DTMF menu support (main.py is 100%
# voice-driven and never reads a keypress), ini path/token resolution via
# ext.ini's api_add_0/api_add_1 -- see resolve_ini_path_and_token()'s
# docstring: YHM unpacks api_add_N=paramName=value into an actual request
# parameter named paramName, so this reads `ini_path`/`token` directly
# (matching the paired ext.ini files), with the old literal api_add_N scan
# kept only as a defensive fallback -- name-spacing autocorrection, Israeli
# phone validation, ini-line parsing that's tolerant of the Name=Phone /
# Phone=Name inconsistency seen in the sample data, and INI add/remove/
# replace helpers.
#
# build_digit_read() constructs the `read=` response directive used for
# every DTMF prompt (Main Menu, every "press 1/2/3" in Delete/Edit, and the
# keyed phone-number entry via build_phone_entry_read). The syntax follows
# the official YHM API module docs (read=<prompt>=param,useExisting,maxDigits,
# minDigits,...). Every DTMF prompt in this file goes through that one
# function, so if a digit ever comes back unexpectedly on a real call,
# that's the one place to fix.
#
# STATE MAP
# ---------
#   1      Main Menu: ask (plays menu, reads 1 digit)
#   2      Main Menu: process digit -> 11 / 21 / 31
#   11-13  Add:    record name -> confirm by keypad (1=approve / 2=re-record)
#                  -> KEY the phone (built-in approve/re-enter menu) -> save
#   21-22  Delete: record search name -> found menu (delete/search again/menu)
#   31-37  Edit:   record search name -> found menu -> name keep-or-new
#                  (-> record new name -> confirm by keypad) -> phone
#                  keep-or-new (-> KEY the new phone) -> save
# Every state number is "the state that will process whatever the caller
# does next", exactly like main.py's own State 1/2/3/4 -- see main.py's Bug
# #2 changelog note on _reset_to_name_recording for why resets always jump
# straight to the "process" state rather than re-running the "ask" step as
# its own saved state.
# ═══════════════════════════════════════════════════════════════════════════

import os
import re
import tempfile
import logging
import httpx
import difflib
import json
from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse
from pydub import AudioSegment
import speech_recognition as sr

try:
    from rapidfuzz import fuzz
    HAS_RAPIDFUZZ = True
except ImportError:
    HAS_RAPIDFUZZ = False

# הגדרת לוגר קריא ומפורט לשרת Render
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(filename)s:%(lineno)d | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# CHANGELOG — v1.7.0 -> v1.8.0: confirmations by DTMF instead of voice
# ═══════════════════════════════════════════════════════════════════════════
# The recorded "כן/לא" confirmation is gone -- the caller now confirms every
# step by keypad (1 = approve / 2 = re-enter):
#   - ask_voice_confirm/redo_voice_confirm replaced by ask_dtmf_confirm/
#     replay_dtmf_confirm: the recognized name is still played back from its
#     {tts_key}_{phone}.tts file, but instead of routing to the recording
#     sub-folder for a spoken yes/no, the same response carries a documented
#     read= directive collecting ONE digit into {tts_key}_key. States 12
#     (Add) and 35 (Edit) are now plain DTMF-menu states: "1" approves,
#     "2" re-records the name, anything else replays the confirm without
#     discarding the field.
#   - State 12's handoff to the phone step changed from routing to the
#     recording folder to the inline keyed-phone read= (v1.7.0's mechanism).
#   - Removed with the voice-confirm flow: evaluate_confirmation,
#     process_confirm_recording, fuzzy_match, and the confirm_cutoff
#     parameter (no longer read). CONFIRM_/PHONE_ prefixes stay in
#     cleanup_temp_files' list so legacy files from older deployments are
#     still purged on reset.
#   - Net effect for the caller: after saying a name they hear it back and
#     press 1 to approve or 2 to say it again -- no recording of a spoken
#     confirmation, no STT round-trip on the confirm step.
# ═══════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════
# CHANGELOG — v1.6.0 -> v1.7.0: phone entry by DTMF instead of voice
# ═══════════════════════════════════════════════════════════════════════════
# The caller now KEYS phone numbers on the keypad instead of recording them
# aloud (names are still voice + Google STT). What changed:
#   - Add flow: old States 13+14 (record phone -> confirm yes/no) merged
#     into a single State 13. Edit flow: old States 37+38 merged into a
#     single State 37. The new number is collected inline by a documented
#     YHM `read=` directive using the built-in `Phone` input type, which
#     accepts only a valid Israeli number (9 digits starting 02/03/04/08/09
#     or 10 digits starting 05/07), speaks the keyed digits back digit by
#     digit, and then plays YHM's own "לאישור הקישו אחת, להקשה מחודשת הקישו
#     שתיים" menu -- so capture AND confirmation happen inside that one
#     read=, with re-entry handled by YHM ("2" comes back as the value and
#     simply replays the prompt). No trip to the recording sub-folder, no
#     WAV handling, no STT, no CONFIRM/PHONE/NEWPHONE temp files on this
#     path.
#   - build_digit_read() rewritten to the official documented schema
#     (read=<prompt>=param,useExisting,maxDigits,minDigits,...): the old
#     "param,tap,max,min" form was exactly the ⚠️ unproven syntax the file
#     header used to flag, and is NOT part of the documented schema. All
#     existing DTMF menus route through that one function, so this also
#     resolves the header's flagged assumption.
#   - Python-side normalize_and_validate_phone kept on the keyed path as a
#     defense-in-depth net (an unexpected value can never reach the ini).
#   - State map and flow comments updated accordingly; 1_ext.ini is still
#     required -- names remain voice-recorded.
# ═══════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════
# CHANGELOG — Production Bug-fix Refactor (v1.5.0 -> v1.6.0)
# ═══════════════════════════════════════════════════════════════════════════
# Bug #1 — Confirmation word ("Yes"/"No") erroneously recorded as the target name
#   Root causes found in the original code:
#     (a) State 3 downloaded and ran STT on the confirmation WAV file, but NEVER
#         deleted it afterwards (unlike State 2 and State 4, which did delete
#         their WAV files). The leftover confirmation recording could sit in the
#         extension folder and later be picked up by State 2 (e.g. on the next
#         call) and mistaken for a freshly recorded NAME.
#     (b) Worse, the "confirmation audio too short/corrupt" branch AND the
#         "confirmation score below threshold" branch inside State 3 both saved
#         `state = 2` ("next input is a NAME") even though the prompt played
#         back to the caller explicitly asked them to reconfirm. A short "כן"
#         recording can easily trip the "< 1000 bytes" corruption check, so this
#         mismatch between the spoken prompt and the saved state is precisely
#         what let a "yes" utterance be swallowed by the name-search pipeline.
#   Fixes applied:
#     - Every state (2/3/4) now deletes the numeric WAV file(s) it just listed
#       IMMEDIATELY after downloading them into memory — unconditionally, i.e.
#       even if the subsequent STT/TTS-write step later fails. See
#       `_delete_numeric_wav_files()`. This is the fix that actually matters:
#       an initial draft also added a "skip this exact filename" safety net in
#       `extract_latest_sequential_wav()`, but integration testing showed that
#       was actively harmful — Yemot reuses filenames like "000.wav" for the
#       very next recording once the old one is deleted, so a by-name skip
#       would wrongly ignore a brand-new, legitimate recording. That approach
#       was reverted; see the design note directly on the function.
#     - Fixed both mis-transitioning branches in State 3 to stay in State 3
#       (or, for the "not confirmed" branch, to fully reset — see Bug #2)
#       instead of incorrectly arming State 2.
#
# Bug #2 — Negative confirmation ("No") does not reset the flow
#   Fix applied:
#     - State 3 now explicitly scores the confirmation text against a list of
#       NEGATIVE keywords (["לא", "טעות", "מחדש", "לא מאשר"]), in addition to
#       the existing POSITIVE ("כן" / "מאשר" / ...) keyword list.
#     - Any outcome that is not a clear, confident, positive confirmation
#       (an explicit "no", OR simply failing to clear the confirm_cutoff
#       threshold) now calls the new `_reset_to_name_recording()` helper, which
#       discards the previously recorded name/confirmation artifacts and sends
#       the caller back to State 1's "please say the desired name" prompt —
#       instead of looping on "please reconfirm the same (wrong) name".
#
# Bug #3 — Missing user instruction prompt on multiple matches (State 4 UX)
#   Fix applied:
#     - The multi-match announcement text (written to MATCH_{phone}.tts) now
#       ends with an explicit spoken instruction telling the caller to speak:
#       "אנא אמור מחדש את השם המבוקש מתוך הרשימה".
#     - Enhancement: the up-to-5 announced names are now also persisted to a
#       small CANDIDATES_{phone}.tts file, so State 4 matches the caller's
#       re-spoken choice against exactly what they just heard, rather than
#       silently re-searching the entire INI file. State 4 still resolves
#       directly to RESULT.tts, with no second confirmation step.
#
# Bug #4 — Weak search algorithm & ignored `cutoff` parameter
#   Fix applied:
#     - `find_best_matches()` now scores each candidate with
#       `compute_name_match_score()`, the MAX of two complementary signals:
#         (a) `rapidfuzz.fuzz.token_set_ratio` (or an equivalent pure-Python
#             fallback when rapidfuzz isn't installed) on punctuation-stripped
#             tokens — this correctly scores a partial/subset search like
#             "הבר" against "מאיר יהודה הבר" near 100, instead of the ~35% a
#             whole-string character ratio produces (verified empirically).
#         (b) the AVERAGE of each search-word's single best per-word match
#             against the candidate's words — a secondary signal that keeps
#             tolerating minor STT transcription variance on individual words
#             (e.g. a slightly mis-heard syllable) without over-inflating the
#             score of a merely partial multi-word overlap.
#     - `cutoff` and `confirm_cutoff` are now both parsed by the shared
#       `parse_percentage_param()` helper: strips "%", treats "," as a decimal
#       separator, treats values <= 1.0 as a fraction (0.8 -> 80), clamps the
#       result to [0, 100], and safely falls back to the caller-supplied
#       default on any unparseable input — instead of three separate, slightly
#       inconsistent try/except blocks.
#
# Per the request, LOG.TTS-related behavior is left untouched in this pass
# (no references to it were found in this file).
# ═══════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="bookchanger.py — YHM Phonebook Manager",
    description="ניהול ספר טלפונים קולי/הקשות עבור מערכות ימות המשיח",
    version="1.8.0"
)

# ─── Robust Environment Variable Parsing ──────────────────────────────────────

_yhm_env = os.environ.get("YHM_API_BASE", "").strip()
if not _yhm_env:
    YHM_API_BASE = "https://www.call2all.co.il/ym/api"
else:
    if not _yhm_env.startswith(("http://", "https://")):
        _yhm_env = "https://" + _yhm_env
    YHM_API_BASE = _yhm_env.rstrip("/")

logger.info("YHM API Base URL configured to: %s", YHM_API_BASE)

# ─── פונקציות עזר: נתיבים בימות המשיח ────────────────────────────────────────
# NOTE: main.py also has a get_parent_folder() here, used because its search
# tool exits back to whatever extension invoked it. Dropped in this file —
# bookchanger.py is a closed loop that always returns to its own Main Menu,
# never "up" to a caller.

def get_yhm_path(ext: str, file_name: str) -> str:
    clean_ext = ext.strip("/")
    if clean_ext:
        return f"ivr2:/{clean_ext}/{file_name}"
    return f"ivr2:/{file_name}"


# ─── BUG #4 FIX: התאמת שמות ברמת-מילה (token-level) ──────────────────────────
# הערה: main.py הגדיר כאן גם את fuzzy_match (יחס-דמיון ברמת-תו על מחרוזת שלמה)
# עבור מילות אישור/שלילה קוליות קצרות ("כן"/"לא"). הוא הוסר יחד עם זרימת ה-
# אישור הקולי: ב-bookchanger.py כל האישורים מתבצעים בהקשות (1 = אישור /
# 2 = הקלטה מחודשת) -- בין ב-read= של ask_dtmf_confirm ובין בתפריט המובנה של
# ימות בהקשת מספר הטלפון. ההערה ההיסטורית להלן נשארת על המקור של הבאג:
# יחס-דמיון ברמת-תו על המחרוזת כולה (כפי ש-fuzzy_match היה עושה) מוטה חזק
# על ידי הפרש-אורכים, ולכן חיפוש חלקי כמו "הבר" מול המועמד הארוך "מאיר יהודה
# הבר" מקבל ציון נמוך (נמדד בפועל: ~35%), הרבה מתחת לכל סף (cutoff) סביר --
# זה בדיוק התיאור בדוח התקלה. הפתרון: להשוות ברמת קבוצת-מילים.

_PUNCTUATION_PATTERN = re.compile(r"[\"'\u05F3\u05F4\-.,;:()\[\]{}]+")


def _tokenize(s: str) -> list[str]:
    """מנקה פיסוק נפוץ (כולל גרש/גרשיים עבריים ומקפים בשמות משפחה מחוברים
    כמו 'כהן-לוי') ומפצל למילים, לצורך השוואה ברמת-מילה ולא ברמת-תו גולמי."""
    cleaned = _PUNCTUATION_PATTERN.sub(" ", s.lower().strip())
    return [w for w in cleaned.split() if w]


def _pairwise_ratio(a: str, b: str) -> float:
    """יחס דמיון גולמי בין שתי מילים בודדות (עם rapidfuzz אם קיים, אחרת difflib)."""
    if HAS_RAPIDFUZZ:
        return float(fuzz.ratio(a, b))
    return difflib.SequenceMatcher(None, a, b).ratio() * 100.0


def _token_set_ratio(target_tokens: list[str], candidate_tokens: list[str]) -> float:
    """
    Signal A: משווה את שתי המחרוזות ברמת *קבוצת המילים*, לא ברמת-תו.
    כאשר כל מילות החיפוש מופיעות בתוך מועמד ארוך יותר (למשל "מאיר הבר" בתוך
    "מאיר יהודה הבר"), הציון קרוב ל-100 ללא תלות במילים הנוספות שיש למועמד.
    כשמדובר בחפיפה חלקית בלבד (למשל "מאיר כהן" מול "דב כהן" -- רק "כהן" משותף),
    הציון יורד משמעותית מ-100, כך שהתאמות מלאות עדיין מדורגות גבוה יותר
    מהתאמות חלקיות (אומת בפועל: ~67% במקום 100%).
    """
    if HAS_RAPIDFUZZ:
        return float(fuzz.token_set_ratio(" ".join(target_tokens), " ".join(candidate_tokens)))

    # נפילה בטוחה (ללא rapidfuzz): מימוש ידני שקול באמצעות difflib,
    # לפי אותו עיקרון (השוואת החיתוך מול כל צד).
    t_set, c_set = set(target_tokens), set(candidate_tokens)
    if not t_set or not c_set:
        return _pairwise_ratio(" ".join(target_tokens), " ".join(candidate_tokens))
    intersection = t_set & c_set
    diff1 = t_set - c_set
    diff2 = c_set - t_set
    sorted_intersection = " ".join(sorted(intersection))
    combined1 = (sorted_intersection + " " + " ".join(sorted(diff1))).strip()
    combined2 = (sorted_intersection + " " + " ".join(sorted(diff2))).strip()
    return max(
        _pairwise_ratio(sorted_intersection, combined1),
        _pairwise_ratio(sorted_intersection, combined2),
        _pairwise_ratio(combined1, combined2),
    )


def _average_best_word_match(target_tokens: list[str], candidate_tokens: list[str]) -> float:
    """
    Signal B ("רשת ביטחון" משלימה ל-Signal A): לכל מילה במחרוזת החיפוש,
    מוצאים את המילה הכי דומה לה במועמד, ואז ממוצעים את כל הציונים.
    זה תופס מקרים בהם ה-STT תמלל מילה בודדת קצת אחרת (למשל "כהן" -> "כוהן"),
    כך שההתאמה המדויקת של Signal A לא הייתה תופסת אותה כזהה במאה אחוז
    (אומת בפועל: מעלה ציון כזה מ-~55% ל-~86%).
    שימוש ב"ממוצע" ולא ב"מקסימום גורף" מונע ממקרה של חיפוש דו-מילים לקבל ציון
    מלא רק בגלל שמילה *אחת* (למשל שם משפחה) תואמת במדויק למועמד שגוי.
    """
    if not target_tokens or not candidate_tokens:
        return 0.0
    per_word_best = [
        max(_pairwise_ratio(tw, cw) for cw in candidate_tokens)
        for tw in target_tokens
    ]
    return sum(per_word_best) / len(per_word_best)


def compute_name_match_score(target: str, candidate: str) -> float:
    """
    BUG #4 FIX: ציון ההתאמה הסופי בין טקסט חיפוש (מהתמלול) לבין שם מועמד מה-INI.
    לוקח את המקסימום בין שני האותות המשלימים לעיל, כך שגם חיפוש במילה בודדת
    ("הבר") וגם חיפוש בכמה מילים ("מאיר הבר") מדורגים נכון מול מועמדים ארוכים
    יותר המכילים את כל מילות החיפוש (למשל "מאיר יהודה הבר"), תוך שמירה על
    יכולת אבחנה בין התאמה מלאה להתאמה חלקית-בלבד.
    """
    target_tokens = _tokenize(target)
    candidate_tokens = _tokenize(candidate)
    if not target_tokens or not candidate_tokens:
        return 0.0
    return max(
        _token_set_ratio(target_tokens, candidate_tokens),
        _average_best_word_match(target_tokens, candidate_tokens),
    )


def parse_percentage_param(raw_value: str | None, default: float) -> float:
    """
    BUG #4 FIX: פענוח עמיד של פרמטר סף (cutoff) שמגיע מה-HTTP
    request (query params או form data), ללא קשר לאופן שבו ext.ini/הבקשה שולחים
    אותו בפועל:
      - תומך בפורמטים כמו "80", "0.8", " 80 ", "80%", "80,5" (פסיק כנקודה עשרונית).
      - ערכים <= 1.0 מתפרשים כשבר (0.8 => 80%), עקבי עם ההתנהגות הקודמת.
      - התוצאה תמיד "נצבטת" (clamped) לטווח 0-100 כדי למנוע ערכים לא הגיוניים
        (למשל "150" או "-5") מלשבש את השוואת הסף.
      - כל כשל פענוח (None, מחרוזת ריקה, טקסט לא-מספרי) חוזר בבטחה לברירת
        המחדל שסופקה על ידי הקורא, במקום לזרוק חריגה.
    """
    if raw_value is None:
        return default
    cleaned = str(raw_value).strip().replace("%", "").replace(",", ".")
    if not cleaned:
        return default
    try:
        value = float(cleaned)
    except (ValueError, TypeError):
        logger.warning(
            "פרמטר cutoff לא תקין: '%s'. משתמש בברירת מחדל %.2f%%.", raw_value, default
        )
        return default
    if value <= 1.0:
        value *= 100.0
    return max(0.0, min(100.0, value))


# ─── פונקציות עזר: REST מול שרת ימות המשיח ──────────────────────────────────
# (ללא שינוי — פונקציות אלו אינן מעורבות באף אחת מ-4 התקלות)

async def yhm_read_text_file(path: str, token: str) -> str | None:
    url = f"{YHM_API_BASE}/GetTextFile"
    logger.info("מנסה לקרוא קובץ טקסט מהנתיב: %s", path)
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            resp = await client.get(url, params={"token": token, "what": path})
            if resp.status_code != 200:
                logger.warning("GetTextFile HTTP %d עבור %s", resp.status_code, path)
                return None
            
            payload = resp.json()
            if payload.get("responseStatus") == "OK":
                return payload.get("contents")
            logger.warning(
                "GetTextFile נכשל עבור %s: %s", path, payload.get("message", "")
            )
            return None
        except Exception as exc:
            logger.error("שגיאה בקריאת קובץ %s: %s", path, exc, exc_info=True)
            return None


async def yhm_write_text_file(path: str, contents: str, token: str) -> bool:
    url = f"{YHM_API_BASE}/UploadTextFile"
    logger.info("מנסה לכתוב קובץ טקסט לנתיב: %s (תוכן באורך %d תווים)", path, len(contents))

    async with httpx.AsyncClient(timeout=10.0) as client:
        for attempt, method in enumerate(["POST", "GET"], start=1):
            try:
                params = {"token": token, "what": path, "contents": contents}
                if method == "POST":
                    resp = await client.post(url, data=params)
                else:
                    resp = await client.get(url, params=params)

                if resp.status_code != 200:
                    logger.warning(
                        "UploadTextFile ניסיון %d/%s: HTTP %d עבור %s",
                        attempt, method, resp.status_code, path
                    )
                    continue

                payload = resp.json()
                if payload.get("responseStatus") == "OK":
                    logger.info("UploadTextFile הצליח (%s) עבור %s", method, path)
                    return True

                logger.error(
                    "UploadTextFile (%s) נכשל עבור %s | responseStatus=%s | message=%s",
                    method, path,
                    payload.get("responseStatus", "?"),
                    payload.get("message", "ללא הסבר"),
                )
            except Exception as exc:
                logger.error(
                    "UploadTextFile (%s) ניסיון %d שגיאת חיבור לקובץ %s: %s",
                    method, attempt, path, exc, exc_info=True
                )

    return False


async def yhm_list_dir(path: str, token: str) -> list[dict] | None:
    url = f"{YHM_API_BASE}/GetIVR2Dir"
    logger.info("מנסה לקבל רשימת קבצים בנתיב: %s", path)
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            resp = await client.get(url, params={"token": token, "path": path})
            if resp.status_code != 200:
                logger.error("GetIVR2Dir נכשל עם HTTP סטטוס: %d", resp.status_code)
                return None
            payload = resp.json()
            if payload.get("responseStatus") == "OK":
                files_list = payload.get("files", [])
                logger.info("התקבלו %d קבצים בתיקייה %s", len(files_list), path)
                return files_list
            logger.warning("GetIVR2Dir החזיר שגיאת מערכת: %s", payload.get("message", ""))
            return None
        except Exception as exc:
            logger.error("שגיאה בקבלת רשימת קבצים בנתיב %s: %s", path, exc, exc_info=True)
            return None


async def yhm_download_file(path: str, token: str) -> bytes | None:
    url = f"{YHM_API_BASE}/DownloadFile"
    logger.info("מוריד קובץ משרת ימות המשיח: %s", path)
    async with httpx.AsyncClient(timeout=15.0) as client:
        try:
            resp = await client.get(url, params={"token": token, "path": path})
            logger.info("תגובת הורדה: HTTP %d, אורך גוף התשובה: %d בתים", resp.status_code, len(resp.content))
            
            if resp.status_code == 200:
                if resp.content.strip().startswith(b'{"'):
                    try:
                        error_data = json.loads(resp.content.decode('utf-8'))
                        if "responseStatus" in error_data and error_data["responseStatus"] != "OK":
                            logger.critical(
                                "שגיאת API מימות המשיח בהורדת הקובץ %s | הודעה: %s", 
                                path, error_data.get("message", "לא צוין פירוט")
                            )
                            return None
                    except Exception:
                        pass
                
                return resp.content
            
            logger.error("הורדת הקובץ נכשלה ברמת ה-HTTP. קוד סטטוס: %d", resp.status_code)
            return None
        except Exception as exc:
            logger.error("שגיאה בהורדת קובץ %s: %s", path, exc, exc_info=True)
            return None


async def yhm_delete_files(paths: list[str], token: str) -> bool:
    """
    מוחקת קבצים משרת ימות המשיח באמצעות ה-API של FileAction.
    """
    if not paths:
        return True
    
    url = f"{YHM_API_BASE}/FileAction"
    params = {
        "token": token,
        "action": "delete"
    }
    
    for i, path in enumerate(paths):
        params[f"what{i}"] = path
        
    logger.info("מנסה למחוק את קבצי השמע הבאים מימות המשיח: %s", paths)
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            resp = await client.get(url, params=params)
            if resp.status_code == 200:
                payload = resp.json()
                if payload.get("responseStatus") == "OK":
                    logger.info("מחיקת הקבצים מהשלוחה הושלמה בהצלחה.")
                    return True
                logger.error("מחיקת קבצים נכשלה מימות המשיח: %s", payload.get("message", ""))
            else:
                logger.error("מחיקת קבצים נכשלה עם סטטוס HTTP %d", resp.status_code)
        except Exception as e:
            logger.error("שגיאה במהלך ביצוע מחיקת קבצים: %s", e, exc_info=True)
            
    return False


# ─── עיבוד שמע וזיהוי דיבור (STT) ───────────────────────────────────────────
# (ללא שינוי — לא מעורב באף אחת מ-4 התקלות. הערת המשתמש: להתעלם מ-LOG.TTS בשלב זה.)

def process_audio_stt(audio_bytes: bytes, file_name: str) -> str:
    recognizer = sr.Recognizer()
    logger.info("מתחיל עיבוד שמע ל-STT עבור קובץ: %s | גודל: %d בתים", file_name, len(audio_bytes))

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_in:
        temp_in.write(audio_bytes)
        temp_in.flush()
        try:
            logger.info("מנסה לפענח את קובץ השמע כפורמט WAV (מניעת תלות ב-ffmpeg)...")
            try:
                audio = AudioSegment.from_file(temp_in.name, format="wav")
            except Exception as wav_exc:
                logger.warning("פענוח ישיר כ-WAV נכשל (%s). מנסה פענוח אוטומטי...", wav_exc)
                audio = AudioSegment.from_file(temp_in.name)

            logger.info("מוסיף שניות שקט להקלטה לשיפור ביצועי ה-STT...")
            silence = AudioSegment.silent(duration=1000)
            processed = silence + audio + silence

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_out:
                processed.export(temp_out.name, format="wav")
                with sr.AudioFile(temp_out.name) as source:
                    data = recognizer.record(source)

            logger.info("שולח בקשה ל-Google Speech Recognition (עברית)...")
            text = recognizer.recognize_google(data, language="he-IL")
            logger.info("הטקסט שזוהה בהצלחה: '%s'", text)
            return text

        except sr.UnknownValueError:
            logger.warning("Google STT: לא זוהה דיבור ברור בהקלטה.")
            return ""
        except sr.RequestError as req_err:
            logger.error("שגיאה חמורה מול שרתי ה-STT של גוגל: %s", req_err)
            return ""
        except Exception as e:
            logger.error("שגיאה כללית בתהליך ה-STT: %s", e, exc_info=True)
            return ""


# ─── ניהול מצבי שיחה ─────────────────────────────────────────────────────────

def extract_latest_sequential_wav(files: list[dict]) -> dict | None:
    """
    מחלץ את קובץ ה-WAV הנומרי הגבוה ביותר (למשל 002.wav מתוך 000.wav, 001.wav, 002.wav).
    סדר עוקב זה מייצג את ההקלטה האחרונה ביותר שהתווספה לשלוחה.

    BUG #1 FIX — הערת עיצוב: גרסה מוקדמת של התיקון ניסתה להוסיף כאן גם פרמטר
    exclude_name (כדי "להתעלם במפורש" מהקובץ ששמו נשמר כ-last_processed_file),
    כרשת ביטחון נוספת מעבר למחיקה. בבדיקות אינטגרציה בפועל התברר שזו גישה
    שגויה: לאחר מחיקת 000.wav, ההקלטה *הבאה* (למשל תשובת האישור) נשמרת על
    ידי ימות המשיח כמעט תמיד תחת אותו שם "000.wav" (מספור מתחדש מ-0 באותה
    תיקיית הקלטה). סינון-לפי-שם היה אם כן פוסל בטעות הקלטה חדשה ולגיטימית
    רק בגלל ששיתפה שם עם קובץ שכבר נמחק — תקלה חמורה יותר מזו שניסה למנוע.
    לכן ההגנה האמיתית והמספיקה נגד Bug #1 היא אך ורק המחיקה המיידית והבלתי-
    מותנית מיד לאחר ההורדה (ראה _delete_numeric_wav_files, הנקראת בכל מצב
    מיד לאחר yhm_download_file) — לא סינון לפי שם קובץ.
    """
    wav_files = []
    for f in files:
        name = f.get("name", "")
        if name.lower().endswith(".wav"):
            base_name = name[:-4] # הסרת הסיומת .wav
            if base_name.isdigit():
                # שמירה של ערך המספר כאינטג'ר יחד עם המילון של הקובץ
                wav_files.append((int(base_name), f))
    
    if not wav_files:
        logger.warning("לא נמצאו קבצי WAV נומריים עוקבים (כדוגמת 000.wav) בשלוחה זו.")
        return None
    
    # מיון לפי הערך הנומרי בסדר עולה (0, 1, 2...)
    wav_files.sort(key=lambda x: x[0])
    
    # בחירת הקובץ עם המספר הגבוה ביותר
    highest_num, selected_file = wav_files[-1]
    logger.info(
        "נמצאו %d קבצי WAV נומריים עוקבים. הקובץ שנבחר הוא בעל המספר הגבוה ביותר: %s", 
        len(wav_files), selected_file.get("name")
    )
    return selected_file


async def _delete_numeric_wav_files(ext: str, files: list[dict], token: str) -> None:
    """
    BUG #1 FIX: מרכזת במקום אחד את לוגיקת מחיקת קובצי ה-WAV הנומריים
    (000.wav, 001.wav, ...), משותפת לכל המצבים (State 2/3/4).

    זו הפונקציה שנקראת כעת *מיד* לאחר שקובץ השמע יורד לזיכרון בכל מצב —
    ללא תלות בהצלחת כתיבת קובץ ה-TTS או פענוח ה-STT שמגיעים אחר-כך.
    בקוד המקורי, State 3 לא ביצע מחיקה כזו בכלל, ו-State 2 ביצע אותה רק
    "if write_success" (כלומר אם כתיבת ה-TTS נכשלה, קובץ השמע נשאר בתיקייה).
    שני הפערים האלה הם הסיבה השורשית ל-Bug #1: הקלטת אישור/שם שנשארת
    בתיקייה עלולה "לדלוף" ולהיתפס בטעות כהקלטה חדשה במצב אחר.
    """
    wav_to_delete = []
    for f in files:
        name = f.get("name", "")
        if name.lower().endswith(".wav"):
            base_name = name[:-4]
            if base_name.isdigit():  # מוחק אך ורק קבצים נומריים עוקבים!
                wav_to_delete.append(get_yhm_path(ext, name))

    if wav_to_delete:
        logger.info(
            "מבצע מחיקה מיידית של %d קובצי WAV נומריים משלוחה %s: %s",
            len(wav_to_delete), ext, wav_to_delete
        )
        await yhm_delete_files(wav_to_delete, token)


async def save_caller_state(
    ext: str, phone: str, api_call_id: str,
    state: int, last_processed_file: str, token: str
) -> bool:
    state_file_path = get_yhm_path(ext, f"state_{phone}.tts")
    content = f"{api_call_id}\n{state}\n{last_processed_file}"
    success = await yhm_write_text_file(state_file_path, content, token)
    if not success:
        logger.critical(
            "CRITICAL: שמירת קובץ מצב נכשלה! phone=%s ext=%s state=%d path=%s "
            "— המאזין עלול לחזור לשלב 1 בשיחה הבאה.",
            phone, ext, state, state_file_path
        )
    else:
        logger.info(
            "שמירת מצב השיחה הושלמה בהצלחה: phone=%s | ext=%s | state=%d", phone, ext, state
        )
    return success


# NOTE: main.py also has _reset_to_name_recording() and parse_ini_candidates()
# here. Both are dropped in this file: the same "reset + wipe stale TTS
# files" job is done more generally by ask_record() + cleanup_temp_files()
# below (bookchanger.py resets many different fields, not just one name),
# and parse_ini_candidates()'s both-sides-of-'=' candidate list is replaced
# by get_name_candidates() below, which is name-only — see that function's
# docstring for why.


def get_phone_by_name(ini_content: str, target_name: str) -> str:
    """
    סורק את קובץ ה-INI ומחפש את השורה שבה מופיע השם שנמצא, 
    ומחזיר את הצד השני של השוויון (הטלפון).
    """
    target_name_clean = target_name.strip()
    for line in ini_content.splitlines():
        line = line.strip()
        if not line or line.startswith(";"):
            continue
        if "=" in line:
            parts = line.split("=", 1)
            k, v = parts[0].strip(), parts[1].strip()
            if k == target_name_clean:
                return v
            elif v == target_name_clean:
                return k
    return ""


def find_best_matches(target: str, candidates: list[str], cutoff_ratio: float) -> list[str]:
    """
    BUG #4 FIX: משתמש כעת ב-compute_name_match_score (התאמה ברמת מילה/טוקן,
    ראה למעלה) במקום ביחס דמיון גולמי ברמת-תו על המחרוזת השלמה. כתוצאה מכך,
    ה-cutoff (המגיע דינמית מ-ext.ini דרך הבקשה) הופך "בעל משמעות" אמיתית:
    התאמות רלוונטיות (מילה שלמה מתוך החיפוש נמצאת אצל המועמד) מקבלות ציון
    גבוה ועקבי (~100), בעוד התאמות לא-רלוונטיות נשארות נמוכות משמעותית,
    כך שסף כמו 80% אכן מסנן כראוי במקום לדחות כמעט הכל (או לקבל הכל).
    """
    matches_with_scores = []
    for candidate in candidates:
        score = compute_name_match_score(target, candidate)
        if score >= cutoff_ratio:
            matches_with_scores.append((candidate, score))
    matches_with_scores.sort(key=lambda x: x[1], reverse=True)
    return [match[0] for match in matches_with_scores]


# ─── ה-Endpoint המרכזי ────────────────────────────────────────────────────────

# ═══════════════════════════════════════════════════════════════════════════
# REUSED SECTION ENDS HERE — everything above this line (imports through
# find_best_matches) is unchanged from main.py. Everything below is new,
# written specifically for bookchanger.py's add/delete/edit flows.
# ═══════════════════════════════════════════════════════════════════════════


# ─── Playback paths (NOT the same as get_yhm_path's "ivr2:/..." REST path) ──

def get_play_path(ext: str, base_name: str) -> str:
    """
    The relative path used inside an 's-' response segment to play back a
    .tts file that was already written to the extension folder. This is a
    different path *format* than get_yhm_path (which is for the REST API
    calls themselves) -- it's the exact inline pattern main.py repeats at
    every State 2/3/4 confirmation and match announcement:
        f"/{ext}/NAME_{phone}" if ext else f"/NAME_{phone}"
    Pulled into one function here since bookchanger.py needs it far more
    often than main.py did (every confirm step, every found-record
    announcement, every keep/new-menu announcement).
    """
    return f"/{ext}/{base_name}" if ext else f"/{base_name}"


# ─── Path + token resolution ────────────────────────────────────────────────
# CORRECTED after checking real ext.ini examples from the YHM developer
# community: api_add_N is a config-file-only device for working around
# ini's "no duplicate keys" limitation. Its VALUE is itself "paramName=
# paramValue", and YHM unpacks that before calling the webhook -- so
# `api_add_0=ini_path=ListAllInformation.ini` in ext.ini results in your
# webhook receiving an actual parameter named `ini_path`, NOT one literally
# named `api_add_0`. (Confirmed independently across three unrelated forum
# examples doing exactly this to pass a Gemini key/token/path trio, an
# ini-file token/action/path trio, and a queue token/path/say_tts trio to
# their own webhooks.) This file's ext.ini uses `ini_path` and `token` as
# the chosen parameter names, so that's what's read directly below; the old
# literal api_add_0/api_add_1 scan is kept only as a defensive fallback in
# case a particular setup ever surfaces them unparsed.

def _scan_api_add_params(params: dict) -> list[str]:
    values = []
    for i in range(10):
        val = params.get(f"api_add_{i}", "")
        if val:
            values.append(val.strip())
    return values


def resolve_ini_path_and_token(params: dict) -> tuple[str, str]:
    """
    Primary path: read the `ini_path` / `token` parameters directly -- what
    YHM actually sends once it unpacks this file's ext.ini
    `api_add_0=ini_path=...` / `api_add_1=token=...` directives.

    Fallback, only if either of those came back empty: scan literal
    api_add_N values and guess which is which by shape (a value containing
    '/' or ending '.ini' is the path; whichever other value doesn't look
    path-like is the token) -- defensive in case some setup passes them
    unparsed rather than via the paramName=paramValue convention above.
    """
    ini_path = params.get("ini_path", "").strip()
    token = params.get("token", "").strip()
    if ini_path and token:
        return ini_path, token

    values = _scan_api_add_params(params)

    def looks_like_path(v: str) -> bool:
        return "/" in v or v.lower().endswith(".ini")

    path_candidates = [v for v in values if looks_like_path(v)]
    other_candidates = [v for v in values if not looks_like_path(v)]

    if not ini_path:
        if path_candidates:
            ini_path = path_candidates[0]
        elif values:
            ini_path = values[0]
    if not token:
        if other_candidates:
            token = other_candidates[0]
        elif len(values) > 1:
            token = values[1]

    return ini_path, token


def resolve_ini_full_path(raw_path: str, ext: str) -> str:
    """
    Accepts either a bare filename / relative path (combined with the
    calling extension via get_yhm_path, the same way main.py hardcodes
    ListAllInformation.ini) or an already-fully-qualified 'ivr2:/...' path
    (used as-is). The latter lets the phonebook optionally live outside the
    calling extension's own folder -- e.g. shared across several
    extensions -- without any extra configuration surface.
    """
    raw_path = raw_path.strip()
    if raw_path.startswith("ivr2:"):
        return raw_path
    if raw_path.startswith("/"):
        # Root-relative path (e.g. "/3/2/ListAllInformation.ini"): resolve
        # from the IVR root, not from the calling extension's folder.
        # Without this, a leading "/" was silently stripped and the path
        # was re-rooted under the calling extension ("ivr2:/3/6/3/2/...").
        return f"ivr2:{raw_path}"
    return get_yhm_path(ext, raw_path)


# ─── Shared "capture the latest recording" pipeline ─────────────────────────
# main.py repeats the list-dir -> extract-latest -> download -> delete
# (Bug #1 discipline) -> validate-length -> STT sequence almost verbatim in
# States 2/3/4. bookchanger.py needs this same sequence at ten different
# points across Add/Delete/Edit, so it's factored into one function here --
# a deliberate DRY refactor versus main.py's copy-paste style, done
# specifically because of how much more often this file needs it.

async def capture_latest_recording(ext: str, token: str) -> tuple[str, str] | None:
    """
    Returns (recognized_text, filename) for the most recently recorded
    numeric WAV in `ext`, or None if there's no usable recording (missing,
    failed download, too short, or STT returned nothing). The numeric WAV
    is deleted immediately and unconditionally after download regardless of
    what happens next -- the same immediate-and-unconditional discipline
    main.py's Bug #1 fix is built around -- so a caller of this function
    never needs to think about WAV cleanup itself.
    """
    files = await yhm_list_dir(get_yhm_path(ext, ""), token)
    latest_audio = extract_latest_sequential_wav(files) if files else None
    if not latest_audio:
        logger.warning("capture_latest_recording: no numeric WAV found in %s", ext)
        return None

    filename = latest_audio["name"]
    download_path = get_yhm_path(ext, filename)
    audio_bytes = await yhm_download_file(download_path, token)

    # Bug #1 discipline: delete immediately and unconditionally, regardless
    # of what happens to audio_bytes / STT below.
    await _delete_numeric_wav_files(ext, files, token)

    if not audio_bytes or len(audio_bytes) < 1000:
        logger.warning("capture_latest_recording: download failed or file too short (%s)", filename)
        return None

    recognized_text = process_audio_stt(audio_bytes, filename)
    if not recognized_text:
        logger.warning("capture_latest_recording: STT returned no text (%s)", filename)
        return None

    return recognized_text, filename


# ─── Name spacing autocorrection ────────────────────────────────────────────
# NOTE: main.py's shared voice yes/no confirmation (State 3's scoring of a
# recorded "כן/לא" -- evaluate_confirmation + process_confirm_recording in
# the previous version of this file) is gone entirely: every confirmation
# in bookchanger.py is DTMF now (1 = approve / 2 = re-enter), either via
# ask_dtmf_confirm's own read= or YHM's built-in approve/re-enter menu on
# the keyed phone step. `confirm_cutoff` is therefore no longer read.




_LETTER_DIGIT_BOUNDARY = re.compile(r"(?<=[^\s\d])(?=\d)|(?<=\d)(?=[^\s\d])")


def autocorrect_name_spacing(name: str) -> str:
    """
    Enforces the "a number can't be glued directly onto a word" rule by
    inserting a space at any letter<->digit boundary that doesn't already
    have whitespace ("Yoni1" -> "Yoni 1"), then collapses any resulting
    double spaces. Chosen as autocorrect rather than outright rejection so
    a caller doesn't have to re-record just because STT ran two tokens
    together -- flagged as an assumption when this was proposed; flip the
    body of this function to a plain validity check + rejection message if
    you'd rather reject than autocorrect.
    """
    spaced = _LETTER_DIGIT_BOUNDARY.sub(" ", name)
    return re.sub(r"\s+", " ", spaced).strip()


# ─── Israeli phone validation ───────────────────────────────────────────────

_PHONE_DIGITS_ONLY = re.compile(r"\D+")
_VALID_IL_PHONE = re.compile(r"^(0[57]\d{8}|0[23489]\d{7})$")

# Bonus robustness, not strictly requested: if Google STT happens to return
# spoken-out digit WORDS ("אפס חמש שתיים...") instead of numerals -- which
# can happen with slowly-enunciated numbers -- convert each recognized word
# to its digit before validating. No-op when STT already returns numerals
# (the common case), so this only ever helps, never changes normal behavior.
_HEBREW_DIGIT_WORDS = {
    "אפס": "0", "אחד": "1", "אחת": "1", "שתיים": "2", "שניים": "2", "שתים": "2",
    "שלוש": "3", "שלושה": "3", "ארבע": "4", "ארבעה": "4", "חמש": "5", "חמישה": "5",
    "שש": "6", "שישה": "6", "שבע": "7", "שבעה": "7", "שמונה": "8",
    "תשע": "9", "תשעה": "9",
}


def normalize_and_validate_phone(raw_text: str) -> str | None:
    """
    Normalizes STT phone output to digits-only and validates against the
    rule as given: 10 digits starting 05/07, or 9 digits starting
    02/03/04/08/09. Returns the normalized digit string, or None if it
    doesn't validate.
    """
    words = raw_text.split()
    converted_tokens = [_HEBREW_DIGIT_WORDS.get(w, w) for w in words]
    combined = " ".join(converted_tokens)
    digits_only = _PHONE_DIGITS_ONLY.sub("", combined)
    if _VALID_IL_PHONE.match(digits_only):
        return digits_only
    return None


# ─── INI line parsing (bidirectional-aware) + mutation helpers ─────────────
# The sample ListAllInformation.ini has one line reversed (phone=name
# instead of name=phone) -- the same thing main.py's get_phone_by_name
# already tolerates by checking both sides of '='. These helpers apply the
# same tolerance on READ, using the phone regex above to figure out which
# side is which, while every WRITE from this file always produces a
# canonical Name=Phone line.

_LOOKS_LIKE_PHONE_SHAPE = re.compile(r"^[\d\s\-]+$")


def _looks_like_phone_shape(s: str) -> bool:
    """
    Lenient shape check used ONLY to decide which side of an ini line is
    the phone side, for tolerantly reading existing/legacy data. Separate
    on purpose from normalize_and_validate_phone's strict Israeli-plan
    check: a legacy number that wouldn't pass fresh validation (like the
    sample data's line starting '0653969443=', where '06' isn't an allowed
    prefix) still needs to be recognized as "the phone side" here, rather
    than being read backwards as a name -- validation happens separately,
    at write-time, only for NEW numbers a caller records.
    """
    if not _LOOKS_LIKE_PHONE_SHAPE.match(s):
        return False
    return len(_PHONE_DIGITS_ONLY.sub("", s)) >= 7


def parse_ini_line(line: str) -> tuple[str, str] | None:
    """Splits a "left=right" line into (name, phone). Returns None for
    blank/comment/malformed lines."""
    if "=" not in line:
        return None
    left, right = line.split("=", 1)
    left, right = left.strip(), right.strip()
    if not left or not right:
        return None
    left_is_phone = _looks_like_phone_shape(left)
    right_is_phone = _looks_like_phone_shape(right)
    if right_is_phone and not left_is_phone:
        return left, right
    if left_is_phone and not right_is_phone:
        return right, left
    # Ambiguous (both or neither look like a phone number): default to the
    # documented Name=Phone convention.
    return left, right


def get_name_candidates(ini_content: str) -> list[str]:
    """
    Name-only candidate list for voice search. Deliberately narrower than
    main.py's parse_ini_candidates, which returns both sides of every line
    (reasonable for its generic single-purpose search tool) -- here that
    would let a phone number get fuzzy-matched against a spoken name, which
    is never useful for Delete/Edit's "search by name" flows.
    """
    names = []
    for raw_line in ini_content.splitlines():
        line = raw_line.strip()
        if not line or line.startswith(";"):
            continue
        parsed = parse_ini_line(line)
        if parsed:
            names.append(parsed[0])
    return names


def normalize_name_for_comparison(name: str) -> str:
    return re.sub(r"\s+", " ", name.strip())


def find_existing_entry_by_name(ini_content: str, name: str) -> tuple[str, str] | None:
    """
    Exact (whitespace-normalized) name match, used for the Add-flow
    duplicate check. Deliberately NOT fuzzy: a fuzzy "already exists" check
    would risk false positives between legitimately different entries like
    "יוני 1" and "יוני 2" -- exactly the kind of name the spacing rule
    exists to support.
    """
    target = normalize_name_for_comparison(name)
    for raw_line in ini_content.splitlines():
        line = raw_line.strip()
        if not line or line.startswith(";"):
            continue
        parsed = parse_ini_line(line)
        if parsed and normalize_name_for_comparison(parsed[0]) == target:
            return parsed
    return None


def add_ini_entry(ini_content: str, name: str, phone: str) -> str:
    """Appends a canonical Name=Phone line."""
    body = ini_content or ""
    if body and not body.endswith("\n"):
        body += "\n"
    return body + f"{name}={phone}\n"


def remove_ini_entry(ini_content: str, name: str, phone: str) -> tuple[str, bool]:
    """
    Removes the first line matching name=phone (or the reversed
    phone=name form, for consistency with the tolerant read behavior
    above). Returns (new_content, removed).
    """
    target_forms = {f"{name}={phone}", f"{phone}={name}"}
    out_lines = []
    removed = False
    for raw_line in ini_content.splitlines():
        if not removed and raw_line.strip() in target_forms:
            removed = True
            continue
        out_lines.append(raw_line)
    new_content = "\n".join(out_lines)
    if out_lines:
        new_content += "\n"
    return new_content, removed


def replace_ini_entry(
    ini_content: str, old_name: str, old_phone: str, new_name: str, new_phone: str
) -> tuple[str, bool]:
    """
    Replaces the first line matching old_name=old_phone (or reversed) with
    a canonical new_name=new_phone line. Returns (new_content, found).
    """
    target_forms = {f"{old_name}={old_phone}", f"{old_phone}={old_name}"}
    out_lines = []
    found = False
    for raw_line in ini_content.splitlines():
        if not found and raw_line.strip() in target_forms:
            out_lines.append(f"{new_name}={new_phone}")
            found = True
        else:
            out_lines.append(raw_line)
    new_content = "\n".join(out_lines)
    if out_lines:
        new_content += "\n"
    return new_content, found


# ─── Temp-file cleanup discipline (Bug #1/#2 spirit, applied to ~a dozen files) ─
# main.py's own changelog is explicit that stale per-call TTS files leaking
# into the next turn was the root cause of two of its four bugs. bookchanger
# has far more of these files (one pair per field, per flow) than main.py
# ever did, so a single exhaustive cleanup helper is used at every full
# reset -- including a plain return to the Main Menu -- rather than
# trusting each call site to remember every filename.

_ALL_TEMP_PREFIXES = [
    "NAME_", "CONFIRM_", "PHONE_",
    "TARGET_NAME_", "TARGET_PHONE_",
    "ORIGINAL_NAME_", "ORIGINAL_PHONE_",
    "CURRENT_NAME_", "CURRENT_PHONE_",
    "NEWNAME_", "NEWPHONE_",
    "ANNOUNCE_",
]


async def cleanup_temp_files(
    ext: str, safe_phone: str, token: str, prefixes: list[str] | None = None
) -> None:
    keys = prefixes if prefixes is not None else _ALL_TEMP_PREFIXES
    paths = [get_yhm_path(ext, f"{p}{safe_phone}.tts") for p in keys]
    await yhm_delete_files(paths, token)


# ─── DTMF digit-read directive (see the ⚠️ note at the top of this file) ────

def sanitize_tts_text(text: str) -> str:
    """
    YHM's id_list_message docs forbid the period ('.') and dash ('-')
    characters in spoken text: the dot is the segment separator between
    t-/s-/f- directives, and a stray one makes the whole response invalid
    (YHM plays M1607 "אין מענה משרת API" and drops the call immediately --
    the exact symptom seen in the Render logs). Applied to every prompt
    text before it's embedded in a response, so prompts can be authored
    naturally (with punctuation) while this stays a one-place fix.
    """
    return text.replace(". ", ", ").replace(".", ", ").replace("-", " ")


def build_digit_read(prompt_text: str, param_name: str, max_digits: int = 1, min_digits: int = 1) -> str:
    """
    Builds the 'read' fragment of a YHM API response, asking the caller to
    press digits and attaching the result to the *next* request under
    `param_name`.

    Uses the schema documented in YHM's official API module docs (f2.freeivr
    topic 56, "read" section -- previously this was an unproven guess; the
    'tap' type this function used to emit is NOT part of the documented
    schema):

        read=<prompt>=<paramName>,<useExisting>,<maxDigits>,<minDigits>,
             <waitSeconds>,<sayAsFormat>,<blockStar>,<blockZero>,...

    Here only the first four data fields are set: useExisting=no (always
    collect fresh input), then max/min digits; the rest fall back to YHM's
    defaults (7s wait, NO spoken playback of what was keyed). Every DTMF
    prompt in this file calls this one function rather than building read=
    strings inline, so any future correction stays a one-place edit.
    """
    return f"read=t-{sanitize_tts_text(prompt_text)}={param_name},no,{max_digits},{min_digits}"


# ─── Keyed phone-number entry (DTMF instead of voice recording) ─────────────

_PHONE_ENTRY_PROMPT = (
    "אנא הקישו את מספר הטלפון. לאישור הקישו אחת, להקשה מחודשת הקישו שתיים"
)


def build_phone_entry_read(param_name: str) -> str:
    """
    The documented `read=` directive that collects a *keyed* phone number.

    Uses YHM's built-in `Phone` input type (same official schema as
    build_digit_read, with the say-format field set):
      - accepts only a valid Israeli number (9 digits starting 02/03/04/08/09,
        or 10 digits starting 05/07),
      - speaks the keyed digits back digit-by-digit,
      - then plays YHM's built-in menu "לאישור הקישו אחת, להקשה מחודשת
        הקישו שתיים" -- so re-entry, and the yes/no style confirmation the
        old voice flow needed, are handled by YHM itself without any extra
        state or recording.
    `param_name` receives exactly the digits (or the re-entry key "2").
    """
    return f"read=t-{sanitize_tts_text(_PHONE_ENTRY_PROMPT)}={param_name},no,10,9,7,Phone"


async def ask_phone_entry(
    ext: str, safe_phone: str, api_call_id: str, token: str, next_state: int
) -> PlainTextResponse:
    """
    Shared "key the phone number" step for Add (State 13) and Edit (State
    37): saves the state that will process the keyed digits, then returns
    the read= directive. No routing to the recording sub-folder happens on
    this path -- YHM collects the digits inline and re-calls the webhook.
    """
    saved = await save_caller_state(ext, safe_phone, api_call_id, next_state, "", token)
    if not saved:
        return PlainTextResponse("id_list_message=t-שגיאה פנימית במערכת אנא נסה שוב")
    return PlainTextResponse(build_phone_entry_read("phone_entry"))


async def process_keyed_phone(
    ext: str, safe_phone: str, api_call_id: str, token: str, params: dict, state: int
) -> str | PlainTextResponse:
    """
    Processes what YHM sends back after the built-in approve/re-enter menu:
      - "2" means the caller asked to re-key the number -> replay the read=
        step (same state).
      - Anything else is the digits themselves. YHM's Phone input type
        already rejected malformed numbers, and the approve menu means a
        confirmed number -- this one state therefore replaces what used to
        be two separate capture + voice-confirm states.

    Returns the validated digit string on success, or a PlainTextResponse
    (a re-key prompt) when the caller chose to re-enter or the Python-side
    validation -- kept as a defense-in-depth safety net so an unexpected
    input shape can never be written into the ini -- rejects the value.
    Caller contract: `if isinstance(result, PlainTextResponse): return result`.
    """
    raw = params.get("phone_entry", "").strip()
    digits = _PHONE_DIGITS_ONLY.sub("", raw)

    if digits == "2":
        logger.info("State %d: caller chose to re-key the phone number.", state)
        return await ask_phone_entry(ext, safe_phone, api_call_id, token, state)

    valid_phone = normalize_and_validate_phone(digits)
    if not valid_phone:
        logger.info(
            "State %d: keyed input '%s' failed the Python-side validation -- "
            "asking to key again (YHM's Phone type should normally prevent this).",
            state, raw,
        )
        return await ask_phone_entry(ext, safe_phone, api_call_id, token, state)

    return valid_phone


# ─── Response-building helpers (ask-and-route / ask-and-read patterns) ─────
# These mirror main.py's own State 1 / _reset_to_name_recording shape:
# perform the action (play a prompt, route somewhere), save the state that
# will process whatever the caller does next, return the response.

async def ask_record(
    ext: str, safe_phone: str, api_call_id: str, token: str,
    routing_path: str, prompt_text: str, next_state: int
) -> PlainTextResponse:
    saved = await save_caller_state(ext, safe_phone, api_call_id, next_state, "", token)
    if not saved:
        return PlainTextResponse("id_list_message=t-שגיאה פנימית במערכת אנא נסה שוב")
    return PlainTextResponse(f"id_list_message=t-{sanitize_tts_text(prompt_text)}&go_to_folder={routing_path}")


async def ask_dtmf_confirm(
    ext: str, safe_phone: str, api_call_id: str, token: str,
    tts_key: str, recognized_text: str, next_state: int
) -> PlainTextResponse:
    """
    Writes `recognized_text` to `{tts_key}_{phone}.tts`, plays it back (via
    a safe 's-' file reference, not interpolated directly into the response
    text -- see the note on ask_menu_with_announcement below) followed by
    the approve/re-record menu, then reads ONE digit via a documented read=
    directive. The webhook is re-called with `{tts_key.lower()}_key` --
    lowercased to match how States 12/35 read the digit (and how every
    other read= param in this file is named):
      "1" = approve  -> the caller moves on to `next_state`'s handler
      "2" = re-record the same field
      anything else  -> the confirm step replays (field not discarded)
    Used for NAME (Add, State 12) and NEWNAME (Edit, State 35). The caller
    no longer has to record a spoken "כן/לא" -- confirmation is DTMF.
    """
    tts_path = get_yhm_path(ext, f"{tts_key}_{safe_phone}.tts")
    await yhm_write_text_file(tts_path, recognized_text, token)
    play_path = get_play_path(ext, f"{tts_key}_{safe_phone}")

    saved = await save_caller_state(ext, safe_phone, api_call_id, next_state, "", token)
    if not saved:
        return PlainTextResponse("id_list_message=t-שגיאה פנימית במערכת אנא נסה שוב")

    confirm_prompt = (
        "האם אתה מאשר את השם. לאישור הקישו אחת, "
        "להקלטה מחודשת הקישו שתיים"
    )
    key_param = f"{tts_key.lower()}_key"
    return PlainTextResponse(
        f"id_list_message=s-{play_path}&{build_digit_read(confirm_prompt, key_param)}"
    )


async def replay_dtmf_confirm(
    ext: str, safe_phone: str, api_call_id: str, token: str,
    tts_key: str, same_state: int
) -> PlainTextResponse:
    """
    Replays the same DTMF confirm (announcement + 1-digit read=) without
    resetting the field -- the DTMF analogue of main.py's State 3 staying
    on State 3 when its confirmation recording was missing/too short.
    Reached when the digit wasn't 1 (approve) or 2 (re-record).
    """
    play_path = get_play_path(ext, f"{tts_key}_{safe_phone}")
    await save_caller_state(ext, safe_phone, api_call_id, same_state, "", token)
    confirm_prompt = (
        "בחירה לא תקינה. האם אתה מאשר את השם. לאישור הקישו אחת, "
        "להקלטה מחודשת הקישו שתיים"
    )
    key_param = f"{tts_key.lower()}_key"
    return PlainTextResponse(
        f"id_list_message=s-{play_path}&{build_digit_read(confirm_prompt, key_param)}"
    )


async def ask_menu(
    ext: str, safe_phone: str, api_call_id: str, token: str,
    prompt_text: str, param_name: str, next_state: int
) -> PlainTextResponse:
    saved = await save_caller_state(ext, safe_phone, api_call_id, next_state, "", token)
    if not saved:
        return PlainTextResponse("id_list_message=t-שגיאה פנימית במערכת אנא נסה שוב")
    return PlainTextResponse(build_digit_read(prompt_text, param_name))


async def ask_menu_with_announcement(
    ext: str, safe_phone: str, api_call_id: str, token: str,
    announcement_text: str, menu_text: str, param_name: str, next_state: int
) -> PlainTextResponse:
    """
    Writes `announcement_text` -- which may contain a dynamic, recognized
    value (a found name/phone, or the record's current name/phone) -- to
    ANNOUNCE_{phone}.tts and plays it via a safe 's-' reference, immediately
    followed by a *static* digit-menu read directive. Keeping dynamic text
    out of the response string itself mirrors why main.py writes NAME_/
    MATCH_ files instead of interpolating recognized text directly: a
    transcribed value could contain characters ('&', '=', '.') that would
    corrupt the response format if embedded inline. `menu_text` is always
    static text authored in this file, so it's safe to embed directly.
    """
    announce_path = get_yhm_path(ext, f"ANNOUNCE_{safe_phone}.tts")
    await yhm_write_text_file(announce_path, announcement_text, token)
    announce_play_path = get_play_path(ext, f"ANNOUNCE_{safe_phone}")

    saved = await save_caller_state(ext, safe_phone, api_call_id, next_state, "", token)
    if not saved:
        return PlainTextResponse("id_list_message=t-שגיאה פנימית במערכת אנא נסה שוב")

    return PlainTextResponse(
        f"id_list_message=s-{announce_play_path}&{build_digit_read(menu_text, param_name)}"
    )


MAIN_MENU_PROMPT = (
    "לתפריט ניהול ספר הטלפונים. להוספת רשומה חדשה הקישו 1. "
    "למחיקת רשומה הקישו 2. לעריכת רשומה קיימת הקישו 3"
)


async def ask_main_menu(
    ext: str, safe_phone: str, api_call_id: str, token: str, prefix: str = ""
) -> PlainTextResponse:
    text = f"{prefix}. {MAIN_MENU_PROMPT}" if prefix else MAIN_MENU_PROMPT
    return await ask_menu(ext, safe_phone, api_call_id, token, text, "menu_key", 2)


async def finalize_edit(
    ext: str, safe_phone: str, api_call_id: str, token: str, ini_path: str
) -> PlainTextResponse:
    """
    Shared final-save step for the Edit flow, reached either from State 36
    (phone kept) or State 38 (new phone confirmed) -- both leave CURRENT_*
    fully resolved, so there's one save path regardless of which fields
    actually changed.
    """
    original_name = await yhm_read_text_file(get_yhm_path(ext, f"ORIGINAL_NAME_{safe_phone}.tts"), token)
    original_phone = await yhm_read_text_file(get_yhm_path(ext, f"ORIGINAL_PHONE_{safe_phone}.tts"), token)
    current_name = await yhm_read_text_file(get_yhm_path(ext, f"CURRENT_NAME_{safe_phone}.tts"), token)
    current_phone = await yhm_read_text_file(get_yhm_path(ext, f"CURRENT_PHONE_{safe_phone}.tts"), token)

    if not all([original_name, original_phone, current_name, current_phone]):
        logger.error("finalize_edit: missing an ORIGINAL/CURRENT temp file for phone=%s", safe_phone)
        await cleanup_temp_files(ext, safe_phone, token)
        return await ask_main_menu(ext, safe_phone, api_call_id, token, prefix="אירעה שגיאה, אנא נסו שוב")

    ini_content = await yhm_read_text_file(ini_path, token) or ""
    new_content, found = replace_ini_entry(ini_content, original_name, original_phone, current_name, current_phone)

    if not found:
        # The record may have changed (or been deleted) via another call in
        # the meantime -- don't silently create a stray line, tell the
        # caller plainly and bail out rather than guessing.
        logger.warning(
            "finalize_edit: original line '%s=%s' no longer found in %s.",
            original_name, original_phone, ini_path
        )
        await cleanup_temp_files(ext, safe_phone, token)
        return await ask_main_menu(ext, safe_phone, api_call_id, token, prefix="הרשומה השתנתה בינתיים, אנא נסו שוב")

    write_success = await yhm_write_text_file(ini_path, new_content, token)
    if not write_success:
        logger.error("finalize_edit: failed to write updated ini to %s", ini_path)
        await cleanup_temp_files(ext, safe_phone, token)
        return await ask_main_menu(ext, safe_phone, api_call_id, token, prefix="שמירת הרשומה נכשלה")

    logger.info(
        "finalize_edit: replaced '%s=%s' with '%s=%s' in %s",
        original_name, original_phone, current_name, current_phone, ini_path
    )
    await cleanup_temp_files(ext, safe_phone, token)
    return await ask_main_menu(ext, safe_phone, api_call_id, token, prefix="הרשומה עודכנה בהצלחה")


# ═══════════════════════════════════════════════════════════════════════════
# The Main Endpoint
# ═══════════════════════════════════════════════════════════════════════════

@app.api_route("/ivr", methods=["GET", "POST"])
async def yemot_ivr(request: Request):
    query_params = dict(request.query_params)
    form_params = {}
    if request.method == "POST":
        try:
            form_params = dict(await request.form())
        except Exception as exc:
            # Same bonus observation main.py flags: request.form() needs
            # python-multipart installed to parse even a plain urlencoded
            # POST body. Logged loudly here so a missing dependency shows up
            # immediately instead of manifesting as a confusing "token
            # missing" error further down.
            logger.error(
                "Form-data parsing failed (check python-multipart is installed): %s",
                exc, exc_info=True
            )
    params = {**query_params, **form_params}

    # YHM reports the caller hanging up with a final request carrying
    # `hangup=yes`. That's a notification, not a live interaction -- answer
    # it immediately without touching state files or playing any prompt
    # (previously the code replayed the menu and wrote state for a call
    # that was already dead).
    if params.get("hangup", "").strip().lower() == "yes":
        logger.info("Hangup notification for call %s -- ignoring.", params.get("ApiCallId", ""))
        return PlainTextResponse("")

    api_phone = params.get("ApiPhone", "").strip()
    api_call_id = params.get("ApiCallId", "").strip()
    api_extension = params.get("ApiExtension", "").strip()

    ini_raw_path, token = resolve_ini_path_and_token(params)

    if not token:
        logger.error("Could not resolve a YHM token from `token` (or a fallback api_add_* scan).")
        return PlainTextResponse("id_list_message=t-שגיאה בטוקן")
    if not api_phone:
        logger.error("Missing ApiPhone parameter.")
        return PlainTextResponse("id_list_message=t-שגיאה במספר טלפון")
    if not api_call_id:
        logger.error("Missing ApiCallId parameter.")
        return PlainTextResponse("id_list_message=t-שגיאה במזהה שיחה")

    safe_phone = "".join(ch for ch in api_phone if ch.isalnum())
    ext = api_extension.strip("/")
    routing_path = f"/{ext}/1" if ext else "/1"

    ini_path = (
        resolve_ini_full_path(ini_raw_path, ext) if ini_raw_path
        else get_yhm_path(ext, "ListAllInformation.ini")
    )

    # ── Read current call state (identical convention to main.py) ─────────
    state_file_path = get_yhm_path(ext, f"state_{safe_phone}.tts")
    state_content = await yhm_read_text_file(state_file_path, token)

    current_state = 1
    if state_content:
        lines = state_content.splitlines()
        if len(lines) >= 2:
            stored_call_id = lines[0].strip()
            if stored_call_id == api_call_id:
                try:
                    current_state = int(lines[1].strip())
                except ValueError:
                    current_state = 1
            else:
                logger.info(
                    "New call detected (previous id=%s, current=%s) -- resetting to Main Menu.",
                    stored_call_id, api_call_id
                )

    logger.info(
        "--- Incoming request --- call=%s phone=%s ext=%s state=%d ini_path=%s",
        api_call_id, safe_phone, ext, current_state, ini_path
    )

    cutoff = parse_percentage_param(params.get("cutoff"), default=80.0)

    # ═══════════════════════════════════════════════════════════════════
    # MAIN MENU (States 1-2)
    # ═══════════════════════════════════════════════════════════════════
    if current_state == 1:
        logger.info("State 1: playing Main Menu, reading 1 digit.")
        return await ask_main_menu(ext, safe_phone, api_call_id, token)

    elif current_state == 2:
        digit = params.get("menu_key", "").strip()
        logger.info("State 2: Main Menu digit pressed: '%s'", digit)

        if digit == "1":
            return await ask_record(
                ext, safe_phone, api_call_id, token, routing_path,
                "אנא אמרו את השם להוספה", 11
            )
        elif digit == "2":
            return await ask_record(
                ext, safe_phone, api_call_id, token, routing_path,
                "אנא אמרו את השם של הרשומה למחיקה", 21
            )
        elif digit == "3":
            return await ask_record(
                ext, safe_phone, api_call_id, token, routing_path,
                "אנא אמרו את השם של הרשומה לעריכה", 31
            )
        else:
            logger.warning("State 2: unrecognized menu digit '%s'. Replaying menu.", digit)
            return await ask_main_menu(ext, safe_phone, api_call_id, token, prefix="בחירה לא תקינה")

    # ═══════════════════════════════════════════════════════════════════
    # ADD FLOW (States 11-13): record name -> confirm by keypad (1=approve
    # / 2=re-record) -> KEY the phone -> append to ini (the phone step is
    # DTMF entry with YHM's built-in approve/re-enter menu, not a recording
    # -- see build_phone_entry_read)
    # ═══════════════════════════════════════════════════════════════════
    elif current_state == 11:
        captured = await capture_latest_recording(ext, token)
        if not captured:
            return await ask_record(
                ext, safe_phone, api_call_id, token, routing_path,
                "דיבור לא ברור. אנא אמרו את השם להוספה", 11
            )
        recognized_text, _ = captured
        corrected_name = autocorrect_name_spacing(recognized_text)
        logger.info("State 11: recognized name '%s' -> corrected '%s'", recognized_text, corrected_name)
        return await ask_dtmf_confirm(
            ext, safe_phone, api_call_id, token,
            "NAME", corrected_name, 12
        )

    elif current_state == 12:
        # The name confirmation is now DTMF: 1 = approve, 2 = re-record.
        digit = params.get("name_key", "").strip()
        logger.info("State 12: name confirm digit: '%s'", digit)

        if digit == "1":  # approved -> move on to the KEYED phone step
            name_path = get_yhm_path(ext, f"NAME_{safe_phone}.tts")
            confirmed_name = await yhm_read_text_file(name_path, token)
            if not confirmed_name:
                logger.error("State 12: NAME_%s.tts missing after confirmation.", safe_phone)
                await cleanup_temp_files(ext, safe_phone, token, ["NAME_"])
                return await ask_record(
                    ext, safe_phone, api_call_id, token, routing_path,
                    "אנא אמרו את השם להוספה", 11
                )

            ini_content = await yhm_read_text_file(ini_path, token) or ""
            if find_existing_entry_by_name(ini_content, confirmed_name):
                logger.info("State 12: '%s' already exists. Asking for a different name.", confirmed_name)
                await cleanup_temp_files(ext, safe_phone, token, ["NAME_"])
                return await ask_record(
                    ext, safe_phone, api_call_id, token, routing_path,
                    "השם הזה כבר קיים ברשימה. אנא אמרו שם אחר", 11
                )

            return await ask_phone_entry(ext, safe_phone, api_call_id, token, 13)

        elif digit == "2":  # re-record the name
            logger.info("State 12: name not confirmed. Resetting to name recording.")
            await cleanup_temp_files(ext, safe_phone, token, ["NAME_"])
            return await ask_record(
                ext, safe_phone, api_call_id, token, routing_path,
                "אנא אמרו את השם להוספה", 11
            )

        else:
            return await replay_dtmf_confirm(ext, safe_phone, api_call_id, token, "NAME", 12)

    elif current_state == 13:
        # Phone step: the caller KEYS the number (DTMF) and YHM itself
        # speaks the digits back and offers approve (1) / re-enter (2)
        # before the webhook is re-called -- so this one state replaces
        # the old capture (13) + voice-confirm (14) pair.
        result = await process_keyed_phone(ext, safe_phone, api_call_id, token, params, 13)
        if isinstance(result, PlainTextResponse):
            return result
        valid_phone = result

        final_name = await yhm_read_text_file(get_yhm_path(ext, f"NAME_{safe_phone}.tts"), token)
        if not final_name:
            logger.error("State 13: NAME_%s.tts missing at final save (phone=%s).", safe_phone)
            await cleanup_temp_files(ext, safe_phone, token)
            return await ask_main_menu(ext, safe_phone, api_call_id, token, prefix="אירעה שגיאה, אנא נסו שוב")

        ini_content = await yhm_read_text_file(ini_path, token) or ""
        # Re-check for a duplicate created in the gap since State 12 (very
        # unlikely for a phone IVR, but cheap to guard against).
        if find_existing_entry_by_name(ini_content, final_name):
            await cleanup_temp_files(ext, safe_phone, token)
            return await ask_record(
                ext, safe_phone, api_call_id, token, routing_path,
                "השם הזה כבר קיים ברשימה. אנא אמרו שם אחר", 11
            )

        new_content = add_ini_entry(ini_content, final_name, valid_phone)
        write_success = await yhm_write_text_file(ini_path, new_content, token)
        if not write_success:
            logger.error("State 13: failed to write updated ini to %s", ini_path)
            await cleanup_temp_files(ext, safe_phone, token)
            return await ask_main_menu(ext, safe_phone, api_call_id, token, prefix="שמירת הרשומה נכשלה")

        logger.info("State 13: added '%s=%s' to %s", final_name, valid_phone, ini_path)
        await cleanup_temp_files(ext, safe_phone, token)
        return await ask_main_menu(ext, safe_phone, api_call_id, token, prefix="הרשומה נוספה בהצלחה")

    # ═══════════════════════════════════════════════════════════════════
    # DELETE FLOW (States 21-22): record search name -> found menu
    # ═══════════════════════════════════════════════════════════════════
    elif current_state == 21:
        captured = await capture_latest_recording(ext, token)
        if not captured:
            return await ask_record(
                ext, safe_phone, api_call_id, token, routing_path,
                "דיבור לא ברור. אנא אמרו את השם של הרשומה למחיקה", 21
            )
        recognized_text, _ = captured

        ini_content = await yhm_read_text_file(ini_path, token)
        if not ini_content:
            logger.error("State 21: ini file missing/empty at %s", ini_path)
            return await ask_main_menu(ext, safe_phone, api_call_id, token, prefix="שגיאה בקריאת קובץ המידע")

        candidates = get_name_candidates(ini_content)
        matches = find_best_matches(recognized_text, candidates, cutoff)

        if not matches:
            logger.info("State 21: no match for '%s'.", recognized_text)
            return await ask_record(
                ext, safe_phone, api_call_id, token, routing_path,
                "לא נמצאה התאמה. אנא אמרו את השם של הרשומה למחיקה", 21
            )

        best_name = matches[0]
        best_phone = get_phone_by_name(ini_content, best_name)
        if not best_phone:
            logger.error("State 21: matched name '%s' but no phone found via get_phone_by_name.", best_name)
            return await ask_record(
                ext, safe_phone, api_call_id, token, routing_path,
                "לא נמצאה התאמה. אנא אמרו את השם של הרשומה למחיקה", 21
            )

        await yhm_write_text_file(get_yhm_path(ext, f"TARGET_NAME_{safe_phone}.tts"), best_name, token)
        await yhm_write_text_file(get_yhm_path(ext, f"TARGET_PHONE_{safe_phone}.tts"), best_phone, token)

        announce = f"נמצאה התאמה. {best_name}, מספר טלפון {best_phone}"
        menu_text = "למחיקת הרשומה הקישו 1. לחיפוש מחדש הקישו 2. לחזרה לתפריט הראשי הקישו 3"
        return await ask_menu_with_announcement(
            ext, safe_phone, api_call_id, token, announce, menu_text, "del_key", 22
        )

    elif current_state == 22:
        digit = params.get("del_key", "").strip()
        logger.info("State 22: found-record menu digit: '%s'", digit)

        if digit == "1":
            target_name = await yhm_read_text_file(get_yhm_path(ext, f"TARGET_NAME_{safe_phone}.tts"), token)
            target_phone = await yhm_read_text_file(get_yhm_path(ext, f"TARGET_PHONE_{safe_phone}.tts"), token)
            if not target_name or not target_phone:
                logger.error("State 22: missing TARGET_NAME/PHONE temp files (phone=%s).", safe_phone)
                await cleanup_temp_files(ext, safe_phone, token)
                return await ask_main_menu(ext, safe_phone, api_call_id, token, prefix="אירעה שגיאה, אנא נסו שוב")

            ini_content = await yhm_read_text_file(ini_path, token) or ""
            new_content, removed = remove_ini_entry(ini_content, target_name, target_phone)

            if not removed:
                logger.warning("State 22: target line '%s=%s' no longer found in ini.", target_name, target_phone)
                await cleanup_temp_files(ext, safe_phone, token)
                return await ask_main_menu(ext, safe_phone, api_call_id, token, prefix="הרשומה השתנתה בינתיים, אנא נסו שוב")

            write_success = await yhm_write_text_file(ini_path, new_content, token)
            if not write_success:
                logger.error("State 22: failed to write updated ini to %s", ini_path)
                await cleanup_temp_files(ext, safe_phone, token)
                return await ask_main_menu(ext, safe_phone, api_call_id, token, prefix="מחיקת הרשומה נכשלה")

            logger.info("State 22: removed '%s=%s' from %s", target_name, target_phone, ini_path)
            await cleanup_temp_files(ext, safe_phone, token)
            return await ask_main_menu(ext, safe_phone, api_call_id, token, prefix="הרשומה נמחקה בהצלחה")

        elif digit == "2":
            await cleanup_temp_files(ext, safe_phone, token, ["TARGET_NAME_", "TARGET_PHONE_", "ANNOUNCE_"])
            return await ask_record(
                ext, safe_phone, api_call_id, token, routing_path,
                "אנא אמרו את השם של הרשומה למחיקה", 21
            )
        else:
            await cleanup_temp_files(ext, safe_phone, token)
            return await ask_main_menu(ext, safe_phone, api_call_id, token)

    # ═══════════════════════════════════════════════════════════════════
    # EDIT FLOW (States 31-37): record search name -> found menu ->
    # name keep-or-new (-> record new name -> confirm by keypad) ->
    # phone keep-or-new (-> KEY the new phone, with YHM's built-in
    # approve/re-enter menu) -> save
    # ═══════════════════════════════════════════════════════════════════
    elif current_state == 31:
        captured = await capture_latest_recording(ext, token)
        if not captured:
            return await ask_record(
                ext, safe_phone, api_call_id, token, routing_path,
                "דיבור לא ברור. אנא אמרו את השם של הרשומה לעריכה", 31
            )
        recognized_text, _ = captured

        ini_content = await yhm_read_text_file(ini_path, token)
        if not ini_content:
            logger.error("State 31: ini file missing/empty at %s", ini_path)
            return await ask_main_menu(ext, safe_phone, api_call_id, token, prefix="שגיאה בקריאת קובץ המידע")

        candidates = get_name_candidates(ini_content)
        matches = find_best_matches(recognized_text, candidates, cutoff)

        if not matches:
            logger.info("State 31: no match for '%s'.", recognized_text)
            return await ask_record(
                ext, safe_phone, api_call_id, token, routing_path,
                "לא נמצאה התאמה. אנא אמרו את השם של הרשומה לעריכה", 31
            )

        best_name = matches[0]
        best_phone = get_phone_by_name(ini_content, best_name)
        if not best_phone:
            logger.error("State 31: matched name '%s' but no phone found.", best_name)
            return await ask_record(
                ext, safe_phone, api_call_id, token, routing_path,
                "לא נמצאה התאמה. אנא אמרו את השם של הרשומה לעריכה", 31
            )

        # ORIGINAL_* is the immutable key used later to locate/replace the
        # old ini line. CURRENT_* is the working copy that gets updated in
        # place as the caller decides what to keep vs. re-record.
        await yhm_write_text_file(get_yhm_path(ext, f"ORIGINAL_NAME_{safe_phone}.tts"), best_name, token)
        await yhm_write_text_file(get_yhm_path(ext, f"ORIGINAL_PHONE_{safe_phone}.tts"), best_phone, token)
        await yhm_write_text_file(get_yhm_path(ext, f"CURRENT_NAME_{safe_phone}.tts"), best_name, token)
        await yhm_write_text_file(get_yhm_path(ext, f"CURRENT_PHONE_{safe_phone}.tts"), best_phone, token)

        announce = f"נמצאה התאמה. {best_name}, מספר טלפון {best_phone}"
        menu_text = "לעריכת הרשומה הקישו 1. לחיפוש מחדש הקישו 2. לחזרה לתפריט הראשי הקישו 3"
        return await ask_menu_with_announcement(
            ext, safe_phone, api_call_id, token, announce, menu_text, "edit_key1", 32
        )

    elif current_state == 32:
        digit = params.get("edit_key1", "").strip()
        logger.info("State 32: found-record menu digit: '%s'", digit)

        if digit == "1":
            current_name = await yhm_read_text_file(get_yhm_path(ext, f"CURRENT_NAME_{safe_phone}.tts"), token) or ""
            announce = f"השם הנוכחי הוא {current_name}"
            menu_text = "להשארת השם הקישו 1. להקלטת שם חדש הקישו 2"
            return await ask_menu_with_announcement(
                ext, safe_phone, api_call_id, token, announce, menu_text, "edit_key2", 33
            )
        elif digit == "2":
            await cleanup_temp_files(ext, safe_phone, token, [
                "ORIGINAL_NAME_", "ORIGINAL_PHONE_", "CURRENT_NAME_", "CURRENT_PHONE_",
                "ANNOUNCE_",
            ])
            return await ask_record(
                ext, safe_phone, api_call_id, token, routing_path,
                "אנא אמרו את השם של הרשומה לעריכה", 31
            )
        else:
            await cleanup_temp_files(ext, safe_phone, token)
            return await ask_main_menu(ext, safe_phone, api_call_id, token)

    elif current_state == 33:
        digit = params.get("edit_key2", "").strip()
        logger.info("State 33: name keep/new menu digit: '%s'", digit)

        if digit == "1":  # keep current name -> move on to the phone step
            current_phone = await yhm_read_text_file(get_yhm_path(ext, f"CURRENT_PHONE_{safe_phone}.tts"), token) or ""
            announce = f"מספר הטלפון הנוכחי הוא {current_phone}"
            menu_text = "להשארת המספר הקישו 1. להקשת מספר חדש הקישו 2"
            return await ask_menu_with_announcement(
                ext, safe_phone, api_call_id, token, announce, menu_text, "edit_key3", 36
            )
        elif digit == "2":  # record a new name
            return await ask_record(
                ext, safe_phone, api_call_id, token, routing_path,
                "אנא הקליטו את השם החדש", 34
            )
        else:
            current_name = await yhm_read_text_file(get_yhm_path(ext, f"CURRENT_NAME_{safe_phone}.tts"), token) or ""
            announce = f"השם הנוכחי הוא {current_name}"
            menu_text = "בחירה לא תקינה. להשארת השם הקישו 1. להקלטת שם חדש הקישו 2"
            return await ask_menu_with_announcement(
                ext, safe_phone, api_call_id, token, announce, menu_text, "edit_key2", 33
            )

    elif current_state == 34:
        captured = await capture_latest_recording(ext, token)
        if not captured:
            return await ask_record(
                ext, safe_phone, api_call_id, token, routing_path,
                "דיבור לא ברור. אנא הקליטו את השם החדש", 34
            )
        recognized_text, _ = captured
        corrected_name = autocorrect_name_spacing(recognized_text)
        return await ask_dtmf_confirm(
            ext, safe_phone, api_call_id, token,
            "NEWNAME", corrected_name, 35
        )

    elif current_state == 35:
        # The new-name confirmation is now DTMF: 1 = approve, 2 = re-record.
        digit = params.get("newname_key", "").strip()
        logger.info("State 35: new-name confirm digit: '%s'", digit)

        if digit == "1":  # approved -> keep going with the edit
            new_name = await yhm_read_text_file(get_yhm_path(ext, f"NEWNAME_{safe_phone}.tts"), token)
            if not new_name:
                logger.error("State 35: NEWNAME_%s.tts missing after confirmation.", safe_phone)
                return await ask_record(
                    ext, safe_phone, api_call_id, token, routing_path,
                    "אנא הקליטו את השם החדש", 34
                )

            # Duplicate check against every OTHER entry -- not against the
            # record currently being edited itself.
            ini_content = await yhm_read_text_file(ini_path, token) or ""
            original_name = await yhm_read_text_file(get_yhm_path(ext, f"ORIGINAL_NAME_{safe_phone}.tts"), token) or ""
            existing = find_existing_entry_by_name(ini_content, new_name)
            if existing and normalize_name_for_comparison(new_name) != normalize_name_for_comparison(original_name):
                await cleanup_temp_files(ext, safe_phone, token, ["NEWNAME_"])
                return await ask_record(
                    ext, safe_phone, api_call_id, token, routing_path,
                    "השם הזה כבר קיים ברשימה. אנא הקליטו שם אחר", 34
                )

            await yhm_write_text_file(get_yhm_path(ext, f"CURRENT_NAME_{safe_phone}.tts"), new_name, token)
            current_phone = await yhm_read_text_file(get_yhm_path(ext, f"CURRENT_PHONE_{safe_phone}.tts"), token) or ""
            announce = f"מספר הטלפון הנוכחי הוא {current_phone}"
            menu_text = "להשארת המספר הקישו 1. להקשת מספר חדש הקישו 2"
            return await ask_menu_with_announcement(
                ext, safe_phone, api_call_id, token, announce, menu_text, "edit_key3", 36
            )

        elif digit == "2":  # re-record the new name
            await cleanup_temp_files(ext, safe_phone, token, ["NEWNAME_"])
            return await ask_record(
                ext, safe_phone, api_call_id, token, routing_path,
                "אנא הקליטו את השם החדש", 34
            )

        else:
            return await replay_dtmf_confirm(ext, safe_phone, api_call_id, token, "NEWNAME", 35)

    elif current_state == 36:
        digit = params.get("edit_key3", "").strip()
        logger.info("State 36: phone keep/new menu digit: '%s'", digit)

        if digit == "1":  # keep current phone -> both fields resolved, save now
            return await finalize_edit(ext, safe_phone, api_call_id, token, ini_path)
        elif digit == "2":
            # New phone is now KEYED (DTMF), not recorded -- route through
            # the read= phone-entry step instead of the recording folder.
            return await ask_phone_entry(ext, safe_phone, api_call_id, token, 37)
        else:
            current_phone = await yhm_read_text_file(get_yhm_path(ext, f"CURRENT_PHONE_{safe_phone}.tts"), token) or ""
            announce = f"מספר הטלפון הנוכחי הוא {current_phone}"
            menu_text = "בחירה לא תקינה. להשארת המספר הקישו 1. להקשת מספר חדש הקישו 2"
            return await ask_menu_with_announcement(
                ext, safe_phone, api_call_id, token, announce, menu_text, "edit_key3", 36
            )

    elif current_state == 37:
        # New-phone step for Edit: the caller KEYS the number (DTMF) and
        # YHM itself speaks the digits back and offers approve (1) /
        # re-enter (2) before the webhook is re-called -- one state
        # replaces the old capture (37) + voice-confirm (38) pair.
        result = await process_keyed_phone(ext, safe_phone, api_call_id, token, params, 37)
        if isinstance(result, PlainTextResponse):
            return result
        new_phone = result

        await yhm_write_text_file(get_yhm_path(ext, f"CURRENT_PHONE_{safe_phone}.tts"), new_phone, token)
        return await finalize_edit(ext, safe_phone, api_call_id, token, ini_path)

    # ═══════════════════════════════════════════════════════════════════
    # Unrecognized state -> reset to Main Menu
    # ═══════════════════════════════════════════════════════════════════
    else:
        logger.warning("Unrecognized state: %d. Resetting to Main Menu.", current_state)
        await cleanup_temp_files(ext, safe_phone, token)
        await save_caller_state(ext, safe_phone, api_call_id, 1, "", token)
        return PlainTextResponse(f"go_to_folder={routing_path}")


# ─── Health Check ────────────────────────────────────────────────────────────

@app.get("/health")
async def health_check():
    return {"status": "ok", "engine": "FastAPI bookchanger.py — YHM Phonebook Manager v1.8.0"}
