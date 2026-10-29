import os
import re
import queue
import time
import sqlite3
import logging
import threading
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

import telebot
from dotenv import load_dotenv
from google import genai
from google.genai import types

logging.basicConfig(level=logging.INFO)
# Hide the noisy "AFC is enabled" and "HTTP Request" lines from the libraries
logging.getLogger("google_genai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

# ---- Load settings from config.env ----
load_dotenv("config.env")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.8-flash").strip()

# Optional: only these Telegram user IDs may use the bot (comma-separated).
# Needed for /digest and /groups, which reveal what is said in your groups.
ALLOWED_USER_IDS = set()
for _part in os.getenv("ALLOWED_USER_IDS", "").replace(" ", "").split(","):
    if _part.isdigit():
        ALLOWED_USER_IDS.add(int(_part))

# Optional: how many days of group messages to keep (default 14).
try:
    RETENTION_DAYS = max(1, int(os.getenv("RETENTION_DAYS", "14")))
except ValueError:
    RETENTION_DAYS = 14

# Optional: put READ_PICTURES=no in config.env to stop the bot sending pictures to Gemini.
READ_PICTURES = os.getenv("READ_PICTURES", "yes").strip().lower() not in ("no", "false", "0", "off")

if not TELEGRAM_TOKEN or TELEGRAM_TOKEN.startswith("PASTE"):
    print("ERROR: Open config.env and paste your Telegram bot token.")
    raise SystemExit(1)
if not GEMINI_API_KEY or GEMINI_API_KEY.startswith("PASTE"):
    print("ERROR: Open config.env and paste your Gemini API key.")
    raise SystemExit(1)

# ---- Start the connections ----
# num_threads = how many messages can be handled at the same time. Gemini can take
# 15-20 seconds when busy, so extra workers stop one slow answer blocking everyone.
bot = telebot.TeleBot(TELEGRAM_TOKEN, num_threads=6)
client = genai.Client(api_key=GEMINI_API_KEY)

MAX_TELEGRAM_LEN = 4000     # Telegram's limit is 4096 characters per message
MAX_TURNS = 30              # reset a user's memory after this many messages
MAX_SAVED = 100             # max forwarded messages kept per chat
MAX_MSG_CHARS = 2000        # each saved message is cut to this length
MAX_DIGEST_MESSAGES = 400   # most messages read per group in one briefing
MAX_BLOCK_CHARS = 1500      # longest briefing shown per group
DIGEST_WORKERS = 3          # groups summarized at the same time
MAX_FILE_BYTES = 10 * 1024 * 1024   # biggest picture or PDF the bot will read
PICTURE_MIMES = ("image/jpeg", "image/png", "image/webp", "image/heic", "image/heif", "application/pdf")
GROUP_TYPES = ("group", "supergroup")

# Forwarded messages and conversations live in memory only (wiped on restart).
sessions = {}        # chat_id -> {"chat": Gemini chat, "turns": int}
forwards = {}        # chat_id -> list of forwarded messages ("Sender: text")
ack_timers = {}      # chat_id -> timer used to send one "saved" confirmation

# ------------------------------------------------------------------
# Group messages are saved to a file next to this script (messages.db),
# so they survive restarts. Old ones are deleted after RETENTION_DAYS.
# ------------------------------------------------------------------
DB_PATH = os.getenv("MESSAGES_DB") or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "messages.db"
)
db_lock = threading.Lock()
db = sqlite3.connect(DB_PATH, check_same_thread=False)
with db_lock:
    db.execute(
        """CREATE TABLE IF NOT EXISTS messages (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               chat_id INTEGER NOT NULL,
               chat_title TEXT,
               sender TEXT,
               ts INTEGER NOT NULL,
               text TEXT)"""
    )
    db.execute("CREATE INDEX IF NOT EXISTS idx_messages_chat_ts ON messages (chat_id, ts)")
    db.execute("CREATE TABLE IF NOT EXISTS state (user_id INTEGER PRIMARY KEY, last_digest INTEGER)")
    db.commit()

last_purge = 0.0


def purge_old():
    global last_purge
    last_purge = time.time()
    with db_lock:
        db.execute("DELETE FROM messages WHERE ts < ?", (int(time.time()) - RETENTION_DAYS * 86400,))
        db.commit()


purge_old()


# ------------------------------------------------------------------
# Gemini helpers
# ------------------------------------------------------------------
SYSTEM_INSTRUCTION = (
    "You are a Telegram assistant bot powered by Google Gemini. You chat with the "
    "user, and you can also summarize messages they forward to you and draft replies "
    "to messages. You keep a record of messages in the work groups you have been added "
    "to: the user can ask for a briefing of all groups since they last checked with "
    "/digest, or for one group with /summary (also /draft for a reply). You cannot read "
    "the user's other private chats on your own: the user has to forward messages to "
    "you. If asked to be added to a group, explain that the user adds this bot to the "
    "group themselves. Never claim abilities you don't have. Reply in the language the "
    "user writes in. Keep answers clear and reasonably short. Use plain text only: no "
    "markdown, no asterisks, no # headings, because Telegram will show those symbols "
    "literally."
)
if READ_PICTURES:
    SYSTEM_INSTRUCTION += (
        " You can also read pictures sent to you (photos, image files and PDFs), for "
        "example documents such as bills of lading."
    )
GEMINI_CONFIG = types.GenerateContentConfig(system_instruction=SYSTEM_INSTRUCTION)


def new_chat(chat_id):
    sessions[chat_id] = {
        "chat": client.chats.create(model=MODEL_NAME, config=GEMINI_CONFIG),
        "turns": 0,
    }
    return sessions[chat_id]


def get_chat(chat_id):
    session = sessions.get(chat_id)
    if session is None or session["turns"] >= MAX_TURNS:
        session = new_chat(chat_id)
    return session


RETRY_CODES = (429, 500, 503, 504)   # temporary errors worth trying again


def with_retry(fn, attempts=4):
    """Run fn(). If Google says it's busy, wait a few seconds and try again."""
    delay = 2
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as e:
            code = getattr(e, "code", None)
            if code in RETRY_CODES and attempt < attempts - 1:
                logging.warning("Gemini busy (error %s). Retrying in %s seconds...", code, delay)
                time.sleep(delay)
                delay *= 2
                continue
            raise


def error_text(e):
    """A friendly message to show in Telegram instead of a technical error."""
    code = getattr(e, "code", None)
    if code in (503, 500, 504):
        return "Gemini is very busy right now. Please try again in a minute."
    if code == 429:
        return "I've hit Gemini's usage limit for now. Please wait a bit and try again."
    return "Sorry, something went wrong. Please try again in a moment."


chat_locks = {}   # one lock per chat, so quick messages from one person stay in order


def ask_in_session(chat_id, prompt):
    """Ask Gemini inside this user's ongoing conversation (it remembers)."""
    with chat_locks.setdefault(chat_id, threading.Lock()):
        session = get_chat(chat_id)
        response = with_retry(lambda: session["chat"].send_message(prompt))
        session["turns"] += 1
    return response.text or "I couldn't generate a reply to that. Try rephrasing."


def ask_once(prompt, attempts=4):
    """Ask Gemini a one-off question with no memory (used for groups)."""
    response = with_retry(
        lambda: client.models.generate_content(
            model=MODEL_NAME, contents=prompt, config=GEMINI_CONFIG
        ),
        attempts=attempts,
    )
    return response.text or "I couldn't generate a reply to that."


# ------------------------------------------------------------------
# Small helpers
# ------------------------------------------------------------------
def send_long(message, text):
    """Telegram rejects very long messages, so send them in pieces."""
    for i in range(0, len(text), MAX_TELEGRAM_LEN):
        bot.reply_to(message, text[i:i + MAX_TELEGRAM_LEN])


def send_blocks(chat_id, blocks, limit=3800):
    """Send several text blocks, packing as many as fit into each Telegram message."""
    chunk = ""
    for block in blocks:
        block = block[:limit]
        if chunk and len(chunk) + len(block) + 2 > limit:
            bot.send_message(chat_id, chunk)
            chunk = ""
        chunk = f"{chunk}\n\n{block}" if chunk else block
    if chunk:
        bot.send_message(chat_id, chunk)


def display_name(user):
    if user is None:
        return "Unknown"
    name = " ".join(p for p in [user.first_name or "", user.last_name or ""] if p).strip()
    return name or user.username or "Unknown"


def is_forward(message):
    return (getattr(message, "forward_origin", None) is not None
            or getattr(message, "forward_date", None) is not None)


def forward_sender(message):
    """Work out who originally wrote a forwarded message."""
    origin = getattr(message, "forward_origin", None)
    if origin is not None:
        user = getattr(origin, "sender_user", None)
        if user is not None:
            return display_name(user)
        hidden_name = getattr(origin, "sender_user_name", None)
        if hidden_name:
            return hidden_name
        chat = getattr(origin, "chat", None) or getattr(origin, "sender_chat", None)
        if chat is not None and getattr(chat, "title", None):
            return chat.title
    old_user = getattr(message, "forward_from", None)
    if old_user is not None:
        return display_name(old_user)
    return getattr(message, "forward_sender_name", None) or "Unknown"


def format_messages(items):
    return "\n".join(items)


def blocked(message):
    """True if this user isn't allowed. Politely tells them in private chats."""
    if not ALLOWED_USER_IDS:
        return False
    user = message.from_user
    if user is not None and user.id in ALLOWED_USER_IDS:
        return False
    if message.chat.type == "private":
        bot.reply_to(message, "Sorry, this bot is private.")
    return True


def digest_allowed(message):
    """/digest and /groups show what is said in your groups, so they only work
    in a private chat and only once the bot is locked to allowed users."""
    if message.chat.type != "private":
        bot.reply_to(message, "Please send that command to me in a private chat, "
                              "so the briefing stays private.")
        return False
    if not ALLOWED_USER_IDS:
        bot.reply_to(
            message,
            "For safety, /digest and /groups only work once the bot is private. "
            "Send /id, add ALLOWED_USER_IDS=<your number> to config.env, and restart me.",
        )
        return False
    return True


# ------------------------------------------------------------------
# Saved group messages
# ------------------------------------------------------------------
def fmt_time(ts):
    return datetime.fromtimestamp(ts).strftime("%d %b %H:%M")   # this computer's local time


def group_body(message):
    """What to save for a group message: its text, or a label for photos, voice, etc."""
    if message.content_type == "text":
        return message.text or ""
    labels = {
        "photo": "[photo]", "voice": "[voice message]", "audio": "[audio]",
        "video": "[video]", "video_note": "[video message]", "location": "[location shared]",
    }
    if message.content_type == "document":
        name = getattr(getattr(message, "document", None), "file_name", None)
        label = f"[document: {name}]" if name else "[document]"
    else:
        label = labels.get(message.content_type, "[other]")
    return f"{label} {message.caption}" if getattr(message, "caption", None) else label


def save_group_message(message, body):
    """Save one group message and return its row number."""
    with db_lock:
        cursor = db.execute(
            "INSERT INTO messages (chat_id, chat_title, sender, ts, text) VALUES (?, ?, ?, ?, ?)",
            (message.chat.id, message.chat.title or str(message.chat.id),
             display_name(message.from_user), int(message.date), body[:MAX_MSG_CHARS]),
        )
        db.commit()
        row_id = cursor.lastrowid
    if time.time() - last_purge > 6 * 3600:
        purge_old()
    return row_id


def update_message_text(row_id, text):
    with db_lock:
        db.execute("UPDATE messages SET text = ? WHERE id = ?", (text[:MAX_MSG_CHARS], row_id))
        db.commit()


def load_group_messages(chat_id, since_ts, limit=MAX_DIGEST_MESSAGES):
    """Newest `limit` messages since since_ts, returned oldest first."""
    with db_lock:
        rows = db.execute(
            "SELECT ts, sender, text FROM messages WHERE chat_id = ? AND ts >= ? "
            "ORDER BY ts DESC LIMIT ?",
            (chat_id, since_ts, limit),
        ).fetchall()
    rows.reverse()
    return rows


def count_group_messages(chat_id, since_ts):
    with db_lock:
        return db.execute(
            "SELECT COUNT(*) FROM messages WHERE chat_id = ? AND ts >= ?", (chat_id, since_ts)
        ).fetchone()[0]


def delete_group_messages(chat_id):
    with db_lock:
        db.execute("DELETE FROM messages WHERE chat_id = ?", (chat_id,))
        db.commit()


def group_overview(since_ts):
    """Every group seen recently: (chat_id, title, last_message_ts, messages_since)."""
    with db_lock:
        return db.execute(
            "SELECT chat_id, chat_title, MAX(ts), SUM(ts >= ?) FROM messages GROUP BY chat_id",
            (since_ts,),
        ).fetchall()


def get_last_digest(user_id):
    with db_lock:
        row = db.execute("SELECT last_digest FROM state WHERE user_id = ?", (user_id,)).fetchone()
    return row[0] if row else None


def set_last_digest(user_id, ts):
    with db_lock:
        db.execute(
            "INSERT INTO state (user_id, last_digest) VALUES (?, ?) "
            "ON CONFLICT(user_id) DO UPDATE SET last_digest = excluded.last_digest",
            (user_id, ts),
        )
        db.commit()


def format_rows(rows):
    return "\n".join(f"[{fmt_time(ts)}] {sender}: {text}" for ts, sender, text in rows)


def parse_hours(message):
    """Reads the number in '/digest 12' or '/summary 8'. None if there isn't a valid one."""
    parts = (message.text or "").split()
    if len(parts) > 1:
        try:
            hours = float(parts[1])
        except ValueError:
            return None
        if 0 < hours <= RETENTION_DAYS * 24:
            return hours
    return None


# ------------------------------------------------------------------
# Prompts
# ------------------------------------------------------------------
SUMMARY_PROMPT = (
    "Below are Telegram messages. Summarize them clearly: start with a short "
    "overview, then list the key points, then list any questions or action items "
    "that need a response. Keep it concise and in plain text.\n\nMessages:\n{messages}"
)

DRAFT_PROMPT = (
    "Below are Telegram messages I received. Write one short, friendly, natural "
    "reply I could send back. Give only the reply text, in plain text. Match the "
    "language the messages are written in.\n\nMessages:\n{messages}"
)

DIGEST_PROMPT = (
    "You are helping a logistics manager catch up on a Telegram group between a truck "
    "driver and dispatchers. Group name: {title}. The messages below are oldest first, "
    "each with its time. Photos, documents and voice messages appear in [brackets]. For "
    "photos and documents the brackets may contain a description of what the picture shows, "
    "written automatically by an image reader: names and numbers in it can be misread, so "
    "treat them as likely but not certain. Voice messages, and files without a description, "
    "are only labels because you cannot hear or see them.\n\n"
    "Write a short briefing in plain text, with no markdown and no asterisks, in English, "
    "in exactly this layout:\n"
    "Line 1: only one word: URGENT, ACTION or OK.\n"
    "  URGENT = needs the manager now (breakdown, accident, big delay, missed appointment, "
    "cargo damage or shortage, safety problem, a conflict nobody has resolved).\n"
    "  ACTION = something is pending or someone is waiting for an answer or a decision "
    "(including a load that was sent to the driver and not yet confirmed).\n"
    "  OK = routine, nothing needed.\n"
    "Then: Now: one or two sentences on where things stand at the end of the messages "
    "(latest status, location or ETA if mentioned).\n"
    "Then: Loads: if a load was dispatched or discussed (look for picture descriptions that "
    "start with LOAD, or load details in the text), one line per load: ID, PU -> DEL, price, "
    "and its status if known (offered, accepted, picked up, delivered). Copy prices exactly. "
    "Leave this out if there is none.\n"
    "Then: What happened: up to 4 short lines starting with '-', in time order, with times.\n"
    "Then: Open items: who needs to do what, naming the person. Leave this out if there is none.\n"
    "Only use facts from the messages. If something is unclear, say it is unclear. Never "
    "invent names, times, places or numbers. The messages may be in any language.\n\n"
    "Messages:\n{messages}"
)

TAG_ICONS = {"URGENT": "\U0001F534", "ACTION": "\U0001F7E1", "OK": "\U0001F7E2", "ERROR": "⚠️"}
TAG_ORDER = {"URGENT": 0, "ACTION": 1, "OK": 2, "ERROR": 3}


def split_tag(text):
    """Split the model's reply into (URGENT/ACTION/OK, body). Unknown replies count as ACTION."""
    text = text.strip()
    first, _, rest = text.partition("\n")
    tag = re.sub(r"[^A-Za-z]", "", first).upper()
    if tag in ("URGENT", "ACTION", "OK"):
        return tag, rest.strip()
    return "ACTION", text


def render_block(title, count, tag, body):
    body = body.strip()
    if len(body) > MAX_BLOCK_CHARS:
        body = body[:MAX_BLOCK_CHARS].rstrip() + "..."
    return f"{TAG_ICONS.get(tag, '')} {tag}: {title} ({count} messages)\n{body}"


def summarize_group(chat_id, title, since_ts, count):
    """Briefing for one group. Returns (tag, body)."""
    rows = load_group_messages(chat_id, since_ts)
    if not rows:
        return "OK", "No messages."
    prompt = DIGEST_PROMPT.format(title=title, messages=format_rows(rows))
    tag, body = split_tag(ask_once(prompt, attempts=5))
    if count > len(rows):
        body += f"\n(Only the latest {len(rows)} of {count} messages were read.)"
    return tag, body


# ------------------------------------------------------------------
# Pictures: photos, image files and PDFs are read by Gemini.
# Only the description is kept; the picture itself is never saved.
# ------------------------------------------------------------------
image_queue = queue.Queue()   # pictures waiting to be read by 2 background workers


def _image_worker():
    while True:
        job = image_queue.get()
        try:
            job()
        except Exception:
            logging.exception("Could not read a picture")
        finally:
            image_queue.task_done()


for _ in range(2):
    threading.Thread(target=_image_worker, daemon=True).start()


def wait_for_pictures(timeout=45):
    """Give pictures that just arrived a moment to be read before we summarize."""
    deadline = time.time() + timeout
    while image_queue.unfinished_tasks and time.time() < deadline:
        time.sleep(0.2)


def media_info(message):
    """(file_id, mime_type, size, label) for a picture or PDF we can read, else None."""
    if not READ_PICTURES:
        return None
    if message.content_type == "photo" and message.photo:
        biggest = message.photo[-1]   # Telegram lists the sizes smallest to largest
        return biggest.file_id, "image/jpeg", getattr(biggest, "file_size", None), "photo"
    if message.content_type == "document" and message.document:
        doc = message.document
        if (doc.mime_type or "").lower() in PICTURE_MIMES:
            return doc.file_id, doc.mime_type.lower(), doc.file_size, f"document {doc.file_name or 'file'}"
    return None


def picture_part(file_id, mime, size):
    """Download the picture and wrap it for Gemini. None if it is too big."""
    if size and size > MAX_FILE_BYTES:
        return None
    data = bot.download_file(bot.get_file(file_id).file_path)
    if len(data) > MAX_FILE_BYTES:
        return None
    return types.Part.from_bytes(data=data, mime_type=mime)


IMAGE_PROMPT = (
    "This picture was posted in a Telegram work chat between a truck driver and "
    "dispatchers at a logistics company. Plain text only, stating only what you can see.\n\n"
    "1) If it shows a LOAD or TRIP (a dispatch screenshot, Amazon Relay trip details, load "
    "board, rate confirmation, tender), answer on ONE line in exactly this form:\n"
    "LOAD <load or trip ID>: PU <facility, city, state, date and time> -> DEL <facility, "
    "city, state, date and time>; <miles> mi; <equipment / trailer>; PAY $<total> "
    "(base $<amount> + fuel $<amount>)\n"
    "List any extra stops in order between PU and DEL. Times are local to each stop. Copy "
    "every name and number exactly as shown, never calculate or round. Write ? for anything "
    "you cannot read, and leave out parts that are not shown.\n\n"
    "2) If it is another document (bill of lading, proof of delivery, scale ticket, "
    "receipt, invoice), say which kind and give the key details: shipper, consignee, load / "
    "PO / BOL numbers, dates, piece counts, weights, signatures or stamps, and any notes "
    "such as shortage or damage.\n\n"
    "3) Otherwise describe it in at most 40 words (damage, truck or trailer, warning light, "
    "seal number, road, location). If it is blurry or unreadable, say that instead of guessing."
)


def describe_picture(file_id, mime, size, caption=None):
    """Ask Gemini what a picture shows. None if it can't be read."""
    part = picture_part(file_id, mime, size)
    if part is None:
        return None
    prompt = IMAGE_PROMPT
    if caption:
        prompt += f"\nCaption written by the sender: {caption}"
    response = with_retry(
        lambda: client.models.generate_content(
            model=MODEL_NAME, contents=[part, prompt], config=GEMINI_CONFIG
        )
    )
    return (response.text or "").strip()[:800] or None


def picture_text(label, description, caption=None):
    text = f"[{label} - description: {description}]"
    return f"{text} Caption: {caption}" if caption else text


def read_group_picture(row_id, info, caption):
    """Background job: read a picture from a group and add its description to the saved message."""
    file_id, mime, size, label = info
    description = describe_picture(file_id, mime, size, caption)
    if description:
        update_message_text(row_id, picture_text(label, description, caption))


def answer_picture(message, info):
    """A picture sent straight to the bot: answer the question in its caption."""
    chat_id = message.chat.id
    file_id, mime, size, _label = info
    bot.send_chat_action(chat_id, "typing")
    try:
        part = picture_part(file_id, mime, size)
        if part is None:
            bot.reply_to(message, "That file is too big for me to read (the limit is 10 MB).")
            return
        question = message.caption or "Describe this. If it is a document, list its key details."
        send_long(message, ask_in_session(chat_id, [part, question]))
    except Exception as e:
        logging.exception("Could not read a picture")
        bot.reply_to(message, error_text(e))


# ------------------------------------------------------------------
# Commands
# ------------------------------------------------------------------
@bot.message_handler(commands=["id"])
def show_id(message):
    """Works for everyone, so you can find your own number."""
    user_id = message.from_user.id
    bot.reply_to(
        message,
        f"Your Telegram user ID is: {user_id}\n\n"
        f"To make this bot private, add this line to config.env and restart the bot:\n"
        f"ALLOWED_USER_IDS={user_id}",
    )


@bot.message_handler(commands=["start"])
def start(message):
    if blocked(message):
        return
    new_chat(message.chat.id)
    bot.reply_to(
        message,
        "Hello! I'm connected to Google Gemini.\n\n"
        "- Send /digest for a briefing of all your groups since you last checked.\n"
        "- Forward me messages from any chat, then send /summary or /draft.\n"
        "- Or just send me a message and I'll reply.\n\n"
        "Send /help to see all commands.",
    )


@bot.message_handler(commands=["help"])
def help_cmd(message):
    if blocked(message):
        return
    bot.reply_to(
        message,
        "Commands:\n"
        "/digest - briefing of every group since you last checked (urgent first)\n"
        "/digest 12 - same, for the last 12 hours (doesn't move your 'last checked' time)\n"
        "/groups - list the groups I'm watching\n"
        "/summary - in a group: summarize the last 24 hours (/summary 8 = last 8 hours). "
        "In private: summarize your forwarded messages\n"
        "/draft - suggest a reply\n"
        "/clear - forget forwarded messages (in a group: forget its saved messages)\n"
        "/reset - forget our whole conversation\n"
        "/id - show your Telegram ID"
        + ("\n\nPictures: send me a photo, image file or PDF (write your question as the "
           "caption) and I'll read it. In groups I read them automatically for /digest."
           if READ_PICTURES else ""),
    )


@bot.message_handler(commands=["reset"])
def reset(message):
    if blocked(message):
        return
    new_chat(message.chat.id)
    forwards.pop(message.chat.id, None)
    bot.reply_to(message, "Conversation cleared.")


@bot.message_handler(commands=["clear"])
def clear(message):
    if blocked(message):
        return
    forwards.pop(message.chat.id, None)
    if message.chat.type in GROUP_TYPES:
        delete_group_messages(message.chat.id)
    bot.reply_to(message, "Saved messages cleared.")


@bot.message_handler(commands=["digest"])
def digest(message):
    if blocked(message) or not digest_allowed(message):
        return
    user_id = message.from_user.id
    chat_id = message.chat.id
    now = int(time.time())

    hours = parse_hours(message)
    if hours is not None:
        since, advance = now - int(hours * 3600), False
    else:
        last = get_last_digest(user_id)
        since, advance = (last if last else now - 24 * 3600), True
    since_label = fmt_time(since)

    wait_for_pictures()   # pictures that arrived just now may still be being read
    groups = group_overview(since)
    active =[g for g in groups if g[3]]
    quiet = sorted((g[1] or str(g[0])) for g in groups if not g[3])

    if not active:
        if advance:
            set_last_digest(user_id, now)
        text = f"No new messages in any group since {since_label}."
        if quiet:
            text += f" I'm watching {len(quiet)} group(s); /groups lists them."
        else:
            text += " I haven't saved any group messages yet: add me to your groups first."
        bot.reply_to(message, text)
        return

    bot.reply_to(message, f"Reading {len(active)} active group(s) since {since_label}. "
                          "This can take a minute.")
    bot.send_chat_action(chat_id, "typing")

    def work(group):
        gid, title, _last, n = group
        title = title or str(gid)
        try:
            tag, body = summarize_group(gid, title, since, n)
        except Exception:
            logging.exception("Digest failed for group %s", title)
            tag, body = "ERROR", "I couldn't summarize this group (Gemini error). Try /digest again in a minute."
        return title, n, tag, body

    with ThreadPoolExecutor(max_workers=DIGEST_WORKERS) as pool:
        results = list(pool.map(work, active))

    failed = any(r[2] == "ERROR" for r in results)
    results.sort(key=lambda r: (TAG_ORDER.get(r[2], 1), r[0].lower()))
    urgent = sum(1 for r in results if r[2] == "URGENT")
    action = sum(1 for r in results if r[2] == "ACTION")

    header = f"Digest since {since_label}: {len(active)} active group(s)"
    header += f", {urgent} urgent, {action} need action." if (urgent or action) else ", nothing needs you."
    blocks = [header] + [render_block(*r) for r in results]
    if quiet:
        blocks.append("Quiet (no new messages): " + ", ".join(quiet))
    send_blocks(chat_id, blocks)

    if advance:
        if failed:
            bot.send_message(chat_id, "Some groups failed, so I did not move your 'last checked' "
                                      "time. Send /digest again to retry.")
        else:
            set_last_digest(user_id, now)


@bot.message_handler(commands=["groups"])
def groups_cmd(message):
    if blocked(message) or not digest_allowed(message):
        return
    since = int(time.time()) - 24 * 3600
    rows = sorted(group_overview(since), key=lambda r: (r[1] or "").lower())
    if not rows:
        bot.reply_to(
            message,
            "I haven't saved messages from any group yet. Add me to your groups, turn off "
            "Group Privacy for me in @BotFather, then wait for someone to write.",
        )
        return
    lines = [f"- {title or gid}: {n} message(s) in the last 24h, last at {fmt_time(last)}"
             for gid, title, last, n in rows]
    send_long(message, f"Groups I'm watching (messages kept {RETENTION_DAYS} days):\n" + "\n".join(lines))


@bot.message_handler(commands=["summary"])
def summary(message):
    if blocked(message):
        return
    chat_id = message.chat.id

    if message.chat.type in GROUP_TYPES:
        hours = parse_hours(message) or 24
        since = int(time.time() - hours * 3600)
        wait_for_pictures()
        count = count_group_messages(chat_id, since)
        if not count:
            bot.reply_to(
                message,
                f"I haven't saved any messages from this group in the last {hours:g} hour(s). "
                "I only see messages sent while I'm running, and Group Privacy must be off "
                "for me in @BotFather.",
            )
            return
        bot.send_chat_action(chat_id, "typing")
        try:
            title = message.chat.title or "this group"
            tag, body = summarize_group(chat_id, title, since, count)
            send_long(message, render_block(title, count, tag, body))
        except Exception as e:
            logging.exception("Gemini request failed")
            bot.reply_to(message, error_text(e))
        return

    items = list(forwards.get(chat_id, []))
    if not items:
        bot.reply_to(message, "Forward me some messages first, then send /summary.")
        return
    bot.send_chat_action(chat_id, "typing")
    try:
        send_long(message, ask_in_session(chat_id, SUMMARY_PROMPT.format(messages=format_messages(items))))
    except Exception as e:
        logging.exception("Gemini request failed")
        bot.reply_to(message, error_text(e))


@bot.message_handler(commands=["draft"])
def draft(message):
    if blocked(message):
        return
    chat_id = message.chat.id
    in_group = message.chat.type in GROUP_TYPES

    if in_group:
        rows = load_group_messages(chat_id, int(time.time()) - 24 * 3600)[-20:]
        items = [f"{sender}: {text}" for _ts, sender, text in rows]
    else:
        items = list(forwards.get(chat_id, []))

    if not items:
        bot.reply_to(message, "Forward me some messages first, then send /draft.")
        return

    bot.send_chat_action(chat_id, "typing")
    try:
        prompt = DRAFT_PROMPT.format(messages=format_messages(items))
        text = ask_once(prompt) if in_group else ask_in_session(chat_id, prompt)
        send_long(message, text)
    except Exception as e:
        logging.exception("Gemini request failed")
        bot.reply_to(message, error_text(e))


# ------------------------------------------------------------------
# Forwarded messages (private chat with the bot)
# ------------------------------------------------------------------
def send_saved_confirmation(chat_id):
    count = len(forwards.get(chat_id, []))
    if count:
        bot.send_message(
            chat_id,
            f"Saved {count} forwarded message(s).\n"
            "Send /summary for a summary, /draft for a suggested reply, "
            "or just ask me a question about them. /clear forgets them.",
        )


def schedule_confirmation(chat_id):
    """Wait 2 seconds after the last forward, then confirm once."""
    old = ack_timers.get(chat_id)
    if old is not None:
        old.cancel()
    timer = threading.Timer(2.0, send_saved_confirmation, args=[chat_id])
    timer.daemon = True
    ack_timers[chat_id] = timer
    timer.start()


@bot.message_handler(
    func=lambda m: m.chat.type == "private" and is_forward(m),
    content_types=["text", "photo", "video", "document", "audio", "voice"],
)
def handle_forward(message):
    if blocked(message):
        return
    chat_id = message.chat.id
    body = message.text or message.caption or "[non-text message]"
    info = media_info(message)
    if info:
        try:
            description = describe_picture(info[0], info[1], info[2], message.caption)
        except Exception:
            logging.exception("Could not read a forwarded picture")
            description = None
        if description:
            body = picture_text(info[3], description, message.caption)
    saved = forwards.setdefault(chat_id, [])
    saved.append(f"{forward_sender(message)}: {body[:MAX_MSG_CHARS]}")
    del saved[:-MAX_SAVED]
    schedule_confirmation(chat_id)


# ------------------------------------------------------------------
# Groups: quietly save every message (from everyone) so /digest works
# ------------------------------------------------------------------
@bot.message_handler(
    func=lambda m: m.chat.type in GROUP_TYPES,
    content_types=["text", "photo", "voice", "audio", "video", "video_note", "document", "location"],
)
def handle_group_message(message):
    if message.content_type == "text" and (not message.text or message.text.startswith("/")):
        return
    try:
        row_id = save_group_message(message, group_body(message))
        info = media_info(message)
        if info:
            caption = getattr(message, "caption", None)
            image_queue.put(lambda: read_group_picture(row_id, info, caption))
    except Exception:
        logging.exception("Could not save a group message")


# ------------------------------------------------------------------
# Normal private chat
# ------------------------------------------------------------------
@bot.message_handler(func=lambda m: m.chat.type == "private", content_types=["text"])
def handle_text(message):
    if blocked(message):
        return
    chat_id = message.chat.id
    bot.send_chat_action(chat_id, "typing")
    try:
        prompt = message.text
        saved = forwards.pop(chat_id, None)
        if saved:
            prompt = (
                "The user forwarded these Telegram messages:\n"
                f"{format_messages(saved)}\n\n"
                f"The user now says: {message.text}"
            )
        send_long(message, ask_in_session(chat_id, prompt))
    except Exception as e:
        logging.exception("Gemini request failed")
        bot.reply_to(message, error_text(e))


@bot.message_handler(
    func=lambda m: m.chat.type == "private",
    content_types=["photo", "voice", "audio", "video", "document", "sticker"],
)
def handle_other(message):
    if blocked(message):
        return
    info = media_info(message)
    if info:
        answer_picture(message, info)
        return
    if READ_PICTURES:
        bot.reply_to(message, "I can only read text, photos, image files and PDFs for now.")
    else:
        bot.reply_to(message, "I can only read text messages for now.")


if __name__ == "__main__":
    print(f"Bot is running with model '{MODEL_NAME}'. Press Ctrl+C to stop.")
    print(f"Group messages are saved in {DB_PATH} (kept {RETENTION_DAYS} days).")
    print("Picture reading is ON (photos, image files, PDFs)." if READ_PICTURES
          else "Picture reading is OFF (READ_PICTURES=no).")
    if ALLOWED_USER_IDS:
        print(f"Private mode ON: {len(ALLOWED_USER_IDS)} allowed user(s).")
    else:
        print(
            "WARNING: anyone who finds this bot can use it, and /digest is switched off. "
            "Send /id to the bot, then add ALLOWED_USER_IDS=<your number> to config.env and restart."
        )
    bot.infinity_polling(timeout=10, long_polling_timeout=5)
