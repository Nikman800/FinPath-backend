"""
FinPath AI Chatbot — FastAPI backend
POST /chat  →  streams a personalized response from Claude
"""

import os
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import anthropic

app = FastAPI(title="FinPath AI Backend", version="1.0.0")

# CORS — allow Next.js frontend
ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:3000,https://finpath.vercel.app",
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Models ────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    history: list = []          # [{role: "user"|"assistant", content: str}, ...]
    user_profile: dict = {}
    savings_accounts: list = []
    debt_accounts: list = []
    expenses: list = []
    investment_accounts: list = []
    assets: list = []


class SummaryRequest(BaseModel):
    health_score: int
    user_profile: dict = {}
    savings_accounts: list = []
    debt_accounts: list = []
    expenses: list = []
    investment_accounts: list = []
    assets: list = []


# ── Helpers ───────────────────────────────────────────────────────────────────

def build_financial_snapshot(profile: dict, debt_accounts: list, savings_accounts: list, expenses: list, investment_accounts: list, assets: list) -> dict:
    """Pre-compute key financial metrics from profile and account data."""
    income = float(profile.get("monthly_income") or 0)
    car_insurance = float(profile.get("car_insurance_monthly_cost") or 0)

    # Real debt totals from debt_accounts table
    total_debt_from_accounts = sum(float(d.get("balance") or 0) for d in debt_accounts)
    total_debt = total_debt_from_accounts or float(profile.get("total_debt") or 0)

    # Real minimum payments from debt_accounts
    real_min_payments = sum(float(d.get("min_payment") or 0) for d in debt_accounts)
    estimated_min_debt_payment = real_min_payments if real_min_payments > 0 else round(total_debt * 0.02) if total_debt > 0 else 0

    # Real savings from savings_accounts table
    total_savings = sum(float(s.get("balance") or 0) for s in savings_accounts)

    # Real monthly expenses from expenses table
    total_monthly_expenses = sum(float(e.get("amount") or 0) for e in expenses)

    # Investment totals
    total_investments = sum(float(i.get("balance") or 0) for i in investment_accounts)

    # Asset totals
    total_assets = sum(float(a.get("value") or 0) for a in assets)

    # Monthly obligations
    monthly_obligations = car_insurance + estimated_min_debt_payment + total_monthly_expenses

    # Debt-to-income ratio (annual)
    dti_pct = round((total_debt / (income * 12)) * 100) if income > 0 else None

    # Monthly surplus
    monthly_surplus = round(income - monthly_obligations) if income > 0 else None

    # Net worth
    net_worth = total_assets + total_savings + total_investments - total_debt

    # Emergency fund runway (months)
    monthly_essential = total_monthly_expenses if total_monthly_expenses > 0 else monthly_obligations
    emergency_runway_months = round(total_savings / monthly_essential) if monthly_essential > 0 and total_savings > 0 else None

    return {
        "income": income,
        "total_debt": total_debt,
        "car_insurance": car_insurance,
        "monthly_obligations": round(monthly_obligations),
        "estimated_min_debt_payment": estimated_min_debt_payment,
        "real_min_payments_available": real_min_payments > 0,
        "monthly_surplus": monthly_surplus,
        "dti_pct": dti_pct,
        "total_savings": total_savings,
        "total_monthly_expenses": total_monthly_expenses,
        "total_investments": total_investments,
        "total_assets": total_assets,
        "net_worth": net_worth,
        "emergency_runway_months": emergency_runway_months,
    }


def build_system_prompt(profile: dict, debt_accounts: list, savings_accounts: list, expenses: list, investment_accounts: list, assets: list) -> str:
    """Build a structured, calculation-ready system prompt for Claude."""

    snap = build_financial_snapshot(profile, debt_accounts, savings_accounts, expenses, investment_accounts, assets) if profile else {}

    # ── Role & principles ────────────────────────────────────────────────────
    lines = [
        "You are FinPath's AI Financial Mentor — a warm, non-judgmental advisor",
        "specializing in financial wellness for underserved and immigrant communities.",
        "",
        "CORE RULES:",
        "• Use plain language. Define any financial term you use.",
        "• Be empathetic. Never shame or lecture.",
        "• Be specific — use the user's actual numbers in your answer.",
        "• When you can calculate something (runway, payoff time, ratio), do it and show the math simply.",
        "• If a number you need isn't in the snapshot, ask one focused follow-up question.",
        "• End every response with exactly one concrete next step the user can take today.",
        "",
    ]

    # ── Financial snapshot ───────────────────────────────────────────────────
    if profile:
        lines.append("USER FINANCIAL SNAPSHOT:")

        income = snap["income"]
        total_debt = snap["total_debt"]

        if income > 0:
            lines.append(f"• Monthly take-home income: ${income:,.0f}")
        else:
            lines.append("• Monthly income: not provided")

        lines.append(f"• Total debt: ${total_debt:,.0f}" if total_debt else "• Total debt: $0")

        if profile.get("credit_card_count"):
            lines.append(f"• Credit cards: {profile['credit_card_count']}")

        # Debt breakdown from debt_accounts
        if debt_accounts:
            lines.append(f"• Debt accounts ({len(debt_accounts)}):")
            for d in debt_accounts:
                name = d.get("name", "Unknown")
                dtype = (d.get("type") or "").replace("_", " ")
                balance = float(d.get("balance") or 0)
                rate = d.get("interest_rate")
                minpay = d.get("min_payment")
                detail = f"  - {name} ({dtype}): ${balance:,.0f}"
                if rate:
                    detail += f" at {rate}% APR"
                if minpay:
                    detail += f", min payment ${float(minpay):,.0f}/mo"
                lines.append(detail)

        # Savings breakdown
        if snap.get("total_savings", 0) > 0:
            lines.append(f"• Total savings: ${snap['total_savings']:,.0f}")
            for s in savings_accounts:
                sname = s.get("name", "Unknown")
                stype = (s.get("type") or "").replace("_", " ")
                sbal = float(s.get("balance") or 0)
                if sbal > 0:
                    lines.append(f"  - {sname} ({stype}): ${sbal:,.0f}")
        else:
            lines.append("• Liquid savings / emergency fund: $0 or not on file")

        # Monthly expenses
        if snap.get("total_monthly_expenses", 0) > 0:
            lines.append(f"• Total monthly expenses: ${snap['total_monthly_expenses']:,.0f}")
            for e in expenses:
                ename = e.get("name", "Unknown")
                eamt = float(e.get("amount") or 0)
                if eamt > 0:
                    lines.append(f"  - {ename}: ${eamt:,.0f}/mo")

        # Investments
        if snap.get("total_investments", 0) > 0:
            lines.append(f"• Total investments: ${snap['total_investments']:,.0f}")
            for inv in investment_accounts:
                iname = inv.get("name", "Unknown")
                itype = (inv.get("type") or "").replace("_", " ")
                ibal = float(inv.get("balance") or 0)
                if ibal > 0:
                    lines.append(f"  - {iname} ({itype}): ${ibal:,.0f}")

        # Assets
        if snap.get("total_assets", 0) > 0:
            lines.append(f"• Total assets: ${snap['total_assets']:,.0f}")
            for a in assets:
                aname = a.get("name", "Unknown")
                aval = float(a.get("value") or 0)
                if aval > 0:
                    lines.append(f"  - {aname}: ${aval:,.0f}")

        # Net worth
        lines.append(f"• Estimated net worth: ${snap['net_worth']:,.0f}")

        if snap["car_insurance"] > 0:
            provider = profile.get("car_insurance_provider", "unknown provider")
            lines.append(f"• Car insurance: ${snap['car_insurance']:,.0f}/mo ({provider})")

        if snap["monthly_obligations"] > 0:
            label = "actual" if snap["real_min_payments_available"] else "estimated"
            lines.append(f"• Known fixed monthly obligations: ~${snap['monthly_obligations']:,.0f}/mo")
            lines.append(f"  (expenses + {label} min debt payments + car insurance)")

        if snap["monthly_surplus"] is not None:
            surplus = snap["monthly_surplus"]
            label = "surplus" if surplus >= 0 else "shortfall"
            lines.append(f"• Estimated monthly {label}: ${abs(surplus):,.0f}")

        if snap["dti_pct"] is not None:
            lines.append(f"• Debt-to-income ratio: {snap['dti_pct']}%")

        if snap["emergency_runway_months"] is not None:
            lines.append(f"• Emergency fund runway: ~{snap['emergency_runway_months']} months")

        lines.append("")
        lines.append("LIFE CONTEXT:")

        employment_type = profile.get("employment_type", "").upper()
        is_employed = profile.get("is_employed")
        if employment_type:
            lines.append(f"• Employment type: {employment_type}")
        if is_employed is not None:
            lines.append(f"• Currently employed: {'yes' if is_employed else 'no'}")
        if profile.get("is_student"):
            lines.append("• Currently a student")

        household = profile.get("household_size")
        if household:
            provider = "main income provider" if profile.get("is_main_income_provider") else "not the main provider"
            lines.append(f"• Household size: {household} ({provider})")

        if profile.get("has_kids"):
            lines.append("• Has dependents (children)")

        if profile.get("is_homeowner"):
            lines.append("• Homeowner")

        if profile.get("state"):
            lines.append(f"• Location: {profile['state']} {profile.get('zip_code', '')}")

        if profile.get("income_sources"):
            lines.append(f"• Income sources: {', '.join(profile['income_sources'])}")

        is_citizen = profile.get("is_us_citizen")
        if is_citizen is False:
            status = profile.get("visa_status") or "non-citizen"
            lines.append(f"• Immigration status: {status}")
            lines.append("  → Factor visa status into advice about government benefits, unemployment, and credit access.")

        lines.append("")
        lines.append("CALCULATION GUIDANCE:")
        if snap.get("emergency_runway_months") is not None:
            lines.append(f"• Emergency fund covers ~{snap['emergency_runway_months']} months of expenses.")
        elif snap.get("total_savings", 0) == 0:
            lines.append("• User has no savings on file. Ask if they have any liquid savings for emergency calculations.")
        if income > 0 and snap["monthly_obligations"] > 0:
            lines.append(f"• If income stopped, user's known fixed costs are ~${snap['monthly_obligations']:,.0f}/mo.")
        if total_debt > 0 and income > 0 and snap["monthly_surplus"] and snap["monthly_surplus"] > 0:
            months_to_payoff = round(total_debt / snap["monthly_surplus"])
            lines.append(f"• At current surplus, debt payoff would take ~{months_to_payoff} months (rough estimate).")
        if employment_type == "1099" or employment_type == "BOTH":
            lines.append("• User is a contractor — they are NOT eligible for traditional unemployment insurance.")
        elif employment_type == "W2":
            lines.append("• User is a W-2 employee — they ARE likely eligible for unemployment insurance if laid off.")

    lines += [
        "",
        "Keep responses under 250 words. Use short paragraphs. Avoid bullet-point lists unless listing steps.",
    ]

    return "\n".join(lines)


# ── Route ─────────────────────────────────────────────────────────────────────

@app.post("/chat")
async def chat(request: ChatRequest):
    if not request.history or not request.history[-1].get("content", "").strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not configured.")

    client = anthropic.Anthropic(api_key=api_key)
    system_prompt = build_system_prompt(
        request.user_profile,
        request.debt_accounts,
        request.savings_accounts,
        request.expenses,
        request.investment_accounts,
        request.assets,
    )

    # Claude requires alternating user/assistant turns — sanitize just in case
    messages = [{"role": m["role"], "content": m["content"]} for m in request.history]

    def generate():
        with client.messages.stream(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            system=system_prompt,
            messages=messages,
        ) as stream:
            for text in stream.text_stream:
                yield text

    return StreamingResponse(generate(), media_type="text/plain")


@app.post("/summary")
async def summary(request: SummaryRequest):
    """Stream a brief, plain-language interpretation of the user's financial health score."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not configured.")

    client = anthropic.Anthropic(api_key=api_key)
    profile_context = build_system_prompt(
        request.user_profile,
        request.debt_accounts,
        request.savings_accounts,
        request.expenses,
        request.investment_accounts,
        request.assets,
    )

    score = request.health_score
    band = (
        "strong (70-100)" if score >= 70
        else "middling (40-69)" if score >= 40
        else "needs work (0-39)"
    )

    system_prompt = (
        profile_context
        + "\n\nYou are now summarizing the user's Financial Health Score report. "
        "The score is out of 100 and reflects debt-to-income, savings, and obligations. "
        "Explain what THIS user's score means for THEM specifically, grounded in the snapshot above. "
        "Be warm, concrete, and non-judgmental."
    )

    user_message = (
        f"My Financial Health Score is {score}/100 ({band}). "
        "In 2-3 short sentences, tell me what it means and exactly what to do next. "
        "Be brief, concise, and straight to the point. Do NOT ask me any questions. "
        "Give direct advice on the next step I should take. "
        "Sound like a real person, natural and warm, not a coach or chatbot. "
        "Plain text only. No markdown, no bold (**), no bullets, no headings, no labels like 'Next step:'. "
        "Do not use dashes ('-' or '—') as separators; use commas or full stops. "
        "Address me as 'you'."
    )

    def generate():
        with client.messages.stream(
            model="claude-haiku-4-5-20251001",
            max_tokens=200,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        ) as stream:
            for text in stream.text_stream:
                yield text

    return StreamingResponse(generate(), media_type="text/plain")


@app.get("/health")
async def health():
    return {"status": "ok"}
