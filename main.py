#!/usr/bin/env python3
"""
VC Market Sentiment Agent

Searches X (Twitter) and LinkedIn for VC funding activity over the last 15 days,
then uses Claude Opus 4.6 to synthesize findings — including how sentiment has
evolved across the period — into a structured market intelligence report.

Setup:
    cp .env.example .env   # fill in your credentials
    pip install -r requirements.txt
    python main.py

Schedule (every 15 days via cron):
    Run `crontab -e` and add:
    0 8 1,16 * * cd /path/to/vc-sentiment-agent && /usr/bin/env python3 main.py
"""

import os
import sys
from datetime import datetime, timedelta, timezone

import httpx
from dotenv import load_dotenv

import anthropic
from anthropic import beta_tool

load_dotenv()

# ── Config ─────────────────────────────────────────────────────────────────────

# Find alternative LinkedIn actors at apify.com/store → search "LinkedIn posts"
LINKEDIN_ACTOR_ID = "apify/linkedin-post-search-scraper"

WINDOW_DAYS = 15  # Analysis window in days

TODAY = datetime.now().strftime("%B %d, %Y")
START_DATE = (datetime.now() - timedelta(days=WINDOW_DAYS)).strftime("%B %d")
MID_DATE = (datetime.now() - timedelta(days=WINDOW_DAYS // 2)).strftime("%B %d")

client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])


# ── Tools ──────────────────────────────────────────────────────────────────────

@beta_tool
def search_x_posts(query: str, max_results: int = 50) -> str:
    """Search recent X (Twitter) posts for VC funding sentiment and trends.

    Supports standard Twitter operators: #hashtag, "exact phrase", from:username.
    Do NOT include -is:retweet in your query — it is added automatically.

    NOTE: X API Basic tier caps recent search at 7 days regardless of start_time.
    For the full 15-day window on X, Pro tier ($5k/mo) is required.

    Args:
        query: Twitter search query string.
        max_results: Number of posts to return (10-100).
    """
    bearer_token = os.environ.get("X_BEARER_TOKEN", "")
    if not bearer_token:
        return "Error: X_BEARER_TOKEN not set."

    start_time = (
        datetime.now(timezone.utc) - timedelta(days=WINDOW_DAYS)
    ).strftime("%Y-%m-%dT%H:%M:%SZ")

    try:
        resp = httpx.get(
            "https://api.twitter.com/2/tweets/search/recent",
            headers={"Authorization": f"Bearer {bearer_token}"},
            params={
                "query": f"({query}) -is:retweet lang:en",
                "max_results": max(10, min(100, max_results)),
                "start_time": start_time,
                "tweet.fields": "created_at,public_metrics,text,author_id",
                "expansions": "author_id",
                "user.fields": "name,username,description,public_metrics",
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
    except httpx.HTTPStatusError as e:
        return f"X API error {e.response.status_code}: {e.response.text[:300]}"
    except Exception as e:
        return f"X API error: {e}"

    tweets = data.get("data", [])
    if not tweets:
        return f"No X posts found for: {query}"

    users = {u["id"]: u for u in data.get("includes", {}).get("users", [])}

    lines = [f"**X Posts** — `{query}` ({len(tweets)} results, last {WINDOW_DAYS} days)\n"]
    for tweet in tweets:
        author = users.get(tweet.get("author_id", ""), {})
        m = tweet.get("public_metrics", {})
        um = author.get("public_metrics", {})
        lines.append(
            f"@{author.get('username', 'unknown')} ({author.get('name', '')})\n"
            f"Followers: {um.get('followers_count', 0):,} · "
            f"Likes: {m.get('like_count', 0)} · RTs: {m.get('retweet_count', 0)}\n"
            f"{tweet['text']}\n"
            f"[{tweet.get('created_at', '')}]\n"
            "---"
        )
    return "\n".join(lines)


@beta_tool
def search_linkedin_posts(keywords: str, max_posts: int = 25) -> str:
    """Search LinkedIn for posts about VC funding and investment trends.

    Args:
        keywords: Search phrase (e.g. "venture capital AI 2025", "seed round closing").
        max_posts: Maximum posts to fetch (up to 50).
    """
    from apify_client import ApifyClient

    apify_token = os.environ.get("APIFY_API_TOKEN", "")
    if not apify_token:
        return "Error: APIFY_API_TOKEN not set."

    apify = ApifyClient(apify_token)

    try:
        run = apify.actor(LINKEDIN_ACTOR_ID).call(
            run_input={
                "keywords": [keywords],
                "maxResults": min(max_posts, 50),
                "proxy": {"useApifyProxy": True},
            },
            timeout_secs=180,
        )
        items = list(apify.dataset(run["defaultDatasetId"]).iterate_items())
    except Exception as e:
        return f"LinkedIn scrape error: {e}"

    if not items:
        return f"No LinkedIn posts found for: {keywords}"

    lines = [f"**LinkedIn Posts** — `{keywords}` ({len(items)} results)\n"]
    for item in items[:max_posts]:
        author = item.get("author", {})
        # Different Apify actors use different field names — handle both
        text = (item.get("text") or item.get("content") or item.get("commentary") or "")[:600]
        # Include post date when available to support temporal analysis
        post_date = item.get("postedAt") or item.get("date") or item.get("publishedAt") or ""
        lines.append(
            f"{author.get('name', 'Unknown')} — {author.get('headline', '')[:80]}\n"
            f"Likes: {item.get('likesCount', 0)} · Comments: {item.get('commentsCount', 0)}"
            + (f" · Posted: {post_date}" if post_date else "") + "\n"
            f"{text}\n"
            "---"
        )
    return "\n".join(lines)


# ── System Prompt ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = f"""You are a venture capital market intelligence analyst. Your job is to search \
X (Twitter) and LinkedIn for posts from VCs, investors, and founders over the last {WINDOW_DAYS} days, \
then synthesize your findings into a comprehensive report that captures both the current state \
AND how sentiment has evolved across the period.

Today: {TODAY}
Analysis window: {START_DATE} → {TODAY} ({WINDOW_DAYS} days)
Midpoint: ~{MID_DATE} (use to distinguish early-period vs recent signals)

## Research Strategy

Run at least 3-4 searches per platform (6-8 total) across different angles. Pay attention to \
post timestamps — you need enough data to identify shifts between the early half ({START_DATE}–{MID_DATE}) \
and the recent half ({MID_DATE}–{TODAY}) of the window.

**Suggested X searches:**
1. `#venturecapital OR #VCfunding OR #startupfunding` — broad VC activity
2. `"excited to invest" OR "thrilled to lead" OR "proud to announce" OR "portfolio company"` — deal signals
3. `"AI startup" OR "climate tech" OR "biotech" OR "fintech" funding` — sector pulse
4. `"Series A" OR "seed round" OR "pre-seed" raised 2025` — stage-level signals
5. `"down round" OR "valuation reset" OR "bridge round" OR "runway"` — bearish signals

**Suggested LinkedIn searches:**
1. `venture capital investment 2025` — general activity
2. `excited announce investment portfolio company` — deal announcements
3. `AI startup seed funding venture` — top sector
4. `climate tech venture capital investing` — growing sector
5. `venture capital market trends outlook` — macro sentiment

After gathering data, write the complete Markdown report as your final response.

## Output Format

Your final response must be the complete report in this exact structure:

# VC Market Sentiment Report — {TODAY}
*Analysis window: {START_DATE} – {TODAY}*

## Executive Summary
[3-4 sentences covering the current state AND the directional arc of the past {WINDOW_DAYS} days]

## Overall Market Sentiment
**Verdict:** [Bullish / Cautiously Bullish / Neutral / Cautiously Bearish / Bearish]
**Direction:** [Improving / Stable / Deteriorating] vs. the start of the window
[Evidence-backed explanation citing specific posts or patterns]

## Sentiment Evolution ({START_DATE} → {TODAY})
[This is the core temporal analysis. Split the window into two halves and describe what changed:
- **Early period ({START_DATE}–{MID_DATE}):** dominant tone, themes, and deal activity
- **Recent period ({MID_DATE}–{TODAY}):** how sentiment shifted, what accelerated or cooled
- **Trajectory:** is momentum building, stalling, or reversing on key themes?]

## Top Investment Themes
[Numbered list of 4-6 dominant themes. For each, note whether it is gaining, stable, or losing \
momentum across the 15-day window]

## Hottest Sectors & Technologies
[Which sectors are attracting the most capital and excitement. Flag any that surged or cooled \
specifically in the recent half of the window]

## Notable VC Perspectives
[Direct quotes or paraphrased views from identifiable investors, with their names/firms. \
Note if their tone shifted during the period]

## Emerging Narratives
[New themes that barely appeared in the early period but are gaining traction now]

## Fading Narratives
[Topics that were prominent early in the window but have cooled or disappeared recently]

## Cautionary Signals
[Headwinds, valuation concerns, down rounds, cooling sectors, skepticism. Flag any that are \
newly appearing vs. persistent throughout the window]

## Key Deals & Announcements
[Specific companies, rounds, or partnerships surfaced in posts. Note approximate timing]

## Platform Comparison
[Notable differences in what VCs discuss on X vs LinkedIn over this period]

## Methodology
[List exact search queries used, note that X Basic tier covers ~7 days not full 15, \
and any other data limitations]
"""


# ── Agent Runner ───────────────────────────────────────────────────────────────

def run_agent() -> str:
    """Run the VC sentiment agent and return the markdown report."""
    print(f"🚀  VC Sentiment Agent  |  {START_DATE} → {TODAY}\n" + "=" * 50)

    final_report = ""

    runner = client.beta.messages.tool_runner(
        model="claude-opus-4-6",
        max_tokens=16000,
        thinking={"type": "adaptive"},
        tools=[search_x_posts, search_linkedin_posts],
        system=SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": (
                f"Research VC market sentiment for the past {WINDOW_DAYS} days "
                f"({START_DATE} to {TODAY}). Search both X and LinkedIn (at least 6 searches "
                "total), paying attention to post timestamps so you can identify how sentiment "
                "evolved across the period. Then write the complete market intelligence report "
                "including the temporal evolution analysis."
            ),
        }],
    )

    for message in runner:
        for block in message.content:
            if block.type == "tool_use":
                first_arg = next(iter(block.input.values()), "")
                print(f"\n🔍  {block.name}  →  \"{str(first_arg)[:60]}\"")
            elif block.type == "thinking":
                print("   💭 [thinking...]", end="", flush=True)
            elif block.type == "text" and message.stop_reason == "end_turn":
                final_report = block.text
                print("\n\n✅  Analysis complete.\n")

    return final_report


# ── Entry Point ────────────────────────────────────────────────────────────────

def main():
    # Validate required environment variables
    required = {"ANTHROPIC_API_KEY", "X_BEARER_TOKEN", "APIFY_API_TOKEN"}
    missing = required - set(os.environ.keys())
    if missing:
        print(f"❌  Missing env vars: {', '.join(sorted(missing))}")
        print("    Copy .env.example → .env and fill in your credentials.")
        sys.exit(1)

    report = run_agent()

    if not report:
        print("❌  Agent produced no output. Check your API credentials and quotas.")
        sys.exit(1)

    # Save report with date range in filename
    start_slug = (datetime.now() - timedelta(days=WINDOW_DAYS)).strftime("%Y%m%d")
    today_slug = datetime.now().strftime("%Y%m%d")
    filename = f"vc_sentiment_{start_slug}_to_{today_slug}.md"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"📄  Report saved → {filename}\n" + "-" * 50)
    preview = report[:800]
    print(preview)
    if len(report) > 800:
        print(f"\n... [{len(report) - 800} more characters in {filename}]")


if __name__ == "__main__":
    main()
