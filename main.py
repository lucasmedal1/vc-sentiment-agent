#!/usr/bin/env python3
"""
VC Market Sentiment Agent

Searches X (Twitter) and LinkedIn for recent VC funding activity, then uses
Claude Opus 4.6 to synthesize findings into a structured market intelligence report.

Setup:
    cp .env.example .env   # fill in your credentials
    pip install -r requirements.txt
    python main.py
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

TODAY = datetime.now().strftime("%B %d, %Y")

client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])


# ── Tools ──────────────────────────────────────────────────────────────────────

@beta_tool
def search_x_posts(query: str, max_results: int = 50) -> str:
    """Search recent X (Twitter) posts for VC funding sentiment and trends.

    Supports standard Twitter operators: #hashtag, "exact phrase", from:username.
    Do NOT include -is:retweet in your query — it is added automatically.

    Args:
        query: Twitter search query string.
        max_results: Number of posts to return (10–100).
    """
    bearer_token = os.environ.get("X_BEARER_TOKEN", "")
    if not bearer_token:
        return "Error: X_BEARER_TOKEN not set."

    start_time = (
        datetime.now(timezone.utc) - timedelta(days=7)
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

    lines = [f"**X Posts** — `{query}` ({len(tweets)} results, last 7 days)\n"]
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
        lines.append(
            f"{author.get('name', 'Unknown')} — {author.get('headline', '')[:80]}\n"
            f"Likes: {item.get('likesCount', 0)} · Comments: {item.get('commentsCount', 0)}\n"
            f"{text}\n"
            "---"
        )
    return "\n".join(lines)


# ── System Prompt ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = f"""You are a venture capital market intelligence analyst. Your job is to search \
X (Twitter) and LinkedIn for recent posts from VCs, investors, and founders, then synthesize \
your findings into a comprehensive market sentiment report.

Today: {TODAY} | Data window: last 7 days

## Research Strategy

Run at least 3–4 searches per platform (6–8 total) across different angles before writing the report.

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

After gathering sufficient data, write the complete Markdown report as your final response.

## Output Format

Your final response must be the complete report in this structure:

# VC Market Sentiment Report — {TODAY}

## Executive Summary
[3–4 sentences: what's happening in VC right now]

## Overall Market Sentiment
**Verdict:** [Bullish / Cautiously Bullish / Neutral / Cautiously Bearish / Bearish]
[Evidence-backed explanation citing specific posts or patterns]

## Top Investment Themes
[Numbered list of 4–6 dominant themes with evidence from collected posts]

## Hottest Sectors & Technologies
[Which sectors are attracting the most capital and excitement, with evidence]

## Notable VC Perspectives
[Direct quotes or paraphrased views from identifiable investors, with their names/firms]

## Emerging Narratives
[New themes just starting to gain traction — things mentioned less but growing]

## Cautionary Signals
[Headwinds, valuation concerns, down rounds, cooling sectors, skepticism]

## Key Deals & Announcements Mentioned
[Specific companies, rounds, or partnerships surfaced in posts]

## Platform Comparison
[Notable differences in what VCs discuss on X vs LinkedIn]

## Methodology
[List the exact search queries used, note data limitations and caveats]
"""


# ── Agent Runner ───────────────────────────────────────────────────────────────

def run_agent() -> str:
    """Run the VC sentiment agent and return the markdown report."""
    print(f"🚀  VC Sentiment Agent  |  {TODAY}\n" + "=" * 50)

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
                "Please research current VC market sentiment by searching both X and LinkedIn "
                "(at least 6 searches total), then write the complete market intelligence report."
            ),
        }],
    )

    for message in runner:
        for block in message.content:
            if block.type == "tool_use":
                # Show first kwarg for a compact progress line
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

    # Save report
    filename = f"vc_sentiment_{datetime.now().strftime('%Y%m%d_%H%M')}.md"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"📄  Report saved → {filename}\n" + "-" * 50)
    # Preview first ~800 chars
    preview = report[:800]
    print(preview)
    if len(report) > 800:
        print(f"\n... [{len(report) - 800} more characters in {filename}]")


if __name__ == "__main__":
    main()
