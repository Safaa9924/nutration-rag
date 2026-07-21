import streamlit as st
import pandas as pd
import time
import requests
from dotenv import load_dotenv
import os
from io import StringIO

# ==============================================
# Load environment variables (if needed)
# ==============================================
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "openai/gpt-4o-mini")  # fallback

# ==============================================
# Import core functions from your module
# (Assumes 07_prompting.py is in the same directory)
# ==============================================
# If you want to avoid import issues, we copy the essential functions here.
# For production, you would do: from 07_prompting import detect_intent, build_prompt, generate_answer, evaluate_answer, retrieval_quality_report

# ---------- Copy of detect_intent ----------
def detect_intent(question):
    q = question.lower()
    if any(word in q for word in ["avoid", "not eat", "forbidden", "limit", "restriction"]):
        return "foods_to_avoid"
    elif any(word in q for word in ["eat", "recommended", "food", "diet", "nutrition"]):
        return "foods_to_eat"
    elif any(word in q for word in ["meal", "meal plan", "breakfast", "lunch", "dinner"]):
        return "meal_plan"
    elif any(word in q for word in ["exercise", "activity", "lifestyle", "habit"]):
        return "lifestyle"
    else:
        return "general"

# ---------- Copy of build_prompt ----------
def build_prompt(prediction: str, context: str) -> str:
    if not context or not context.strip():
        raise ValueError("Retrieved context is empty or invalid.")
    clean_prediction = prediction.strip() if prediction else "Not Specified"
    clean_context = context.strip()

    prompt = f"""You are a Senior Clinical Dietitian and Evidence-Based Communication Agent.

Your primary objective is to generate an accurate, realistic, and patient-friendly nutrition report based **STRICTLY AND EXCLUSIVELY** on the retrieved scientific context provided inside <context> tags.

============================================================
CORE OPERATIONAL GUARDRAILS (STRICT COMPLIANCE REQUIRED)
============================================================

### 1. ABSOLUTE CONTEXT GROUNDING (ZERO HALLUCINATION)
- Extract facts **ONLY** from the provided `<context>`. 
- **STRICTLY FORBIDDEN**: Do NOT use internal medical knowledge, general diabetes guidelines, or unmentioned nutrition facts.
- **OMISSION RULE**: If a nutrient, food, or lifestyle factor is missing in the `<context>`, **completely omit it**. Do NOT assume or estimate.

### 2. REALISTIC MEAL PLAN & NEUTRAL BASE RULE (CRITICAL)
- **Primary Ingredients**: Meals MUST prominently incorporate the "Recommended Foods" mentioned in `<context>`.
- **Neutral Food Bases (Allowed)**: To make meals realistic and edible, you MAY combine recommended foods with standard neutral staples (e.g., leafy greens, water, eggs, olive oil) ONLY IF they do NOT violate the "Foods to Limit" section.
- **Strict Avoidance**: You MUST ABSOLUTELY AVOID any items listed under "Foods to Limit".
- **Nutritional Truthfulness**: Assign health benefits in the meal plan ONLY to the exact food items mentioned in the context.

### 3. DYNAMIC GOLDEN RULES (NO REDUNDANCY)
- Generate **up to 5 distinct Golden Rules** based ONLY on unique facts in `<context>`.
- **NO REPETITION**: If the context only contains 2 or 3 distinct pieces of advice, output ONLY 2 or 3 rules. **NEVER repeat the same rule in different words** just to reach 5.

### 4. SOURCE EXTRACTION & METADATA CLEANING
- Extract **ONLY** human-readable author citations (e.g., "Reaven et al. (21)", "American Diabetes Association (20)").
- **CRITICAL**: Never output technical metadata, search scores, vectors, rerank scores, or document IDs (e.g., `rerank_score`, `score: 0.89`).
- **FALLBACK**: If no human-readable authors exist in the context, write EXACTLY:
  *"Source: Retrieved clinical document (specific authors not detailed in the provided text)."*

============================================================
INPUT DATA
============================================================
<patient_condition>
{clean_prediction}
</patient_condition>

<context>
{clean_context}
</context>

============================================================
REQUIRED OUTPUT FORMAT (MARKDOWN)
============================================================
Generate the report using this EXACT structure:

# 🩺 Diabetes Personalized Nutrition Report

---

## 1. 🧬 Patient Condition Overview
[Summarize only the condition mentioned in patient_condition/context. If absent, write: "Not described in the retrieved document."]

---

## 2. 📚 Scientific Evidence Summary
[Bullet points derived ONLY from <context>. Skip unmentioned topics.]

---

## 3. 🔗 Evidence-to-Action Translation Bridge
| Scientific Basis (from Context) | Actionable Recommendation |
| :--- | :--- |
[Populate table rows ONLY supported by <context>]

---

## 4. ✅ Your Daily Action Plan
[Numbered actionable steps derived directly from the table above]

---

## 5. 🍽️ Food Guide (Recommended vs. Limit)
| Recommended Foods | Foods to Limit |
| :--- | :--- |
[ONLY include foods explicitly named in <context>]

---

## 6. 🏃 Practical Lifestyle & Daily Habits
- **⏰ Meal Timing**: [Specific advice from context. **Fallback:** "Not specifically detailed in the retrieved document. Please consult your dietitian for personalized advice on this topic."]
- **🍳 Cooking Methods**: [Advice from context or Fallback]
- **🛒 Grocery Shopping**: [Advice from context or Fallback]
- **🍽️ Eating Outside**: [Advice from context or Fallback]
- **💧 Hydration**: [Advice from context or Fallback]

---

## 7. 🥗 Sample One-Day Meal Plan
*(Combine Recommended Foods with neutral staples to create realistic meals. Attribute health benefits ONLY to the correct source).*

- **Breakfast**: [Meal Item - e.g., Scrambled eggs topped with ground flaxseed]  
  *Why?* [Direct context justification]
- **Morning Snack**: [Meal Item]  
  *Why?* [Direct context justification]
- **Lunch**: [Meal Item - e.g., Grilled fatty fish with a side of leafy green salad]  
  *Why?* [Direct context justification]
- **Afternoon Snack**: [Meal Item]  
  *Why?* [Direct context justification]
- **Dinner**: [Meal Item]  
  *Why?* [Direct context justification]

---

## 8. 🧠 Quick Reference Card (Top Golden Rules)
*(Provide only unique rules derived from context. Do not duplicate rules).*
- 🥇 [Rule 1]
- 🥈 [Rule 2]
- 🥉 [Rule 3 (if applicable)]
- 4️⃣ [Rule 4 (if applicable)]
- 5️⃣ [Rule 5 (if applicable)]

---

## 9. 📖 Evidence Sources
[Human-readable author citations or the exact Fallback statement]

---

## 10. ⚠️ Important Disclaimer
This report is generated based on retrieved clinical evidence for informational purposes only and does not constitute medical advice. Please consult your physician or registered dietitian before altering your diet or medication plan.
"""
    return prompt

# ---------- Copy of generate_answer (with fixed system message) ----------
def generate_answer(prompt, model=MODEL_NAME, temperature=0.1, max_tokens=1200):
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    system_message = (
        "You are an expert Clinical Dietitian. "
        "Follow the user's instructions precisely and produce the requested structured report. "
        "Do not deviate from the requested format, do not summarize, and do not skip any section."
    )
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    start = time.time()
    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=600
        )
        response.raise_for_status()
        elapsed = time.time() - start
        result = response.json()
        answer = result["choices"][0]["message"]["content"].strip()
        # For debugging, you can print usage
        return answer
    except Exception as e:
        st.error(f"OpenRouter error: {e}")
        return None

# ==============================================
# Streamlit UI
# ==============================================

# --- Page config ---
st.set_page_config(
    page_title="🍏 Diabetes Nutrition Assistant",
    page_icon="🍏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for a modern, attractive design ---
st.markdown("""
<style>
    /* Main background and font */
    .reportview-container {
        background: #f8fafc;
    }
    .main {
        padding: 1rem 2rem;
    }
    h1, h2, h3 {
        color: #0b3b5c;
        font-weight: 600;
    }
    .stButton > button {
        background-color: #0b3b5c;
        color: white;
        font-weight: 600;
        border-radius: 30px;
        padding: 0.5rem 2rem;
        border: none;
        transition: 0.3s;
    }
    .stButton > button:hover {
        background-color: #1a5276;
        transform: scale(1.02);
        box-shadow: 0 4px 12px rgba(11, 59, 92, 0.2);
    }
    .stTextInput > div > div > input {
        border-radius: 20px;
        border: 1px solid #d1d5db;
        padding: 0.75rem 1rem;
        font-size: 1rem;
    }
    .stTextArea > div > div > textarea {
        border-radius: 20px;
        border: 1px solid #d1d5db;
        font-size: 0.95rem;
    }
    .css-1d391kg {
        background-color: #ffffff;
        border-radius: 20px;
        padding: 1.5rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.04);
    }
    .sidebar .sidebar-content {
        background: #ffffff;
        border-right: 1px solid #e2e8f0;
    }
    .css-1aumxhk {
        background-color: #ffffff;
    }
    .big-number {
        font-size: 2.5rem;
        font-weight: 700;
        color: #0b3b5c;
    }
    .metric-box {
        background: #f1f5f9;
        padding: 0.75rem 1.5rem;
        border-radius: 40px;
        text-align: center;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/diabetes.png", width=80)  # Placeholder, you can replace with your logo
    st.title("🍏 Diabetes Assistant")
    st.markdown("---")
    st.markdown("""
    **Welcome!**  
    This tool uses evidence-based retrieval to generate personalized nutrition reports for diabetic patients.

    **How to use:**
    1. Enter the patient's condition (e.g., *Type 1 Diabetes*).
    2. Paste the retrieved scientific context (from your RAG pipeline).
    3. Click **Generate Report**.

    The report will follow a structured format including:
    - Patient overview
    - Scientific evidence summary
    - Food guide
    - Meal plan
    - Golden rules
    - References
    """)
    st.markdown("---")
    st.caption("Powered by OpenRouter • Built with Streamlit")

# --- Main area ---
st.markdown("<h1 style='text-align: center;'>🩺 Personalized Diabetes Nutrition Report</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #475569;'>Generate a comprehensive, evidence‑based nutrition plan from your retrieved clinical context.</p>", unsafe_allow_html=True)

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("### 📝 Patient Condition")
    patient_condition = st.text_input(
        "Enter the patient's diabetes type or condition",
        placeholder="e.g., Type 1 Diabetes, Type 2 with renal complications",
        value="Type 1 Diabetes"
    )

    st.markdown("### 📚 Retrieved Context")
    # Provide a default example context similar to the screenshot
    default_context = """American Diabetes Association (ADA) - Carbohydrate Guidelines: "Whole grains containing soluble fiber like beta-glucan reduce post-meal glucose spikes and improve insulin sensitivity."
Glycemic Index & Diabetes Management (Journal of Clinical Nutrition): "Minimally processed oats maintain a low-to-medium GI, whereas instant oats exhibit higher glycemic responses."
Additional notes: Steel-cut oats have a GI ~53, instant oatmeal ~75+. For Type 1 diabetes, target ≤45g carbs per meal. Pairing oats with healthy fats/proteins (chia seeds, almonds, Greek yogurt) stabilizes glucose.
"""
    context_text = st.text_area(
        "Paste the retrieved context text here",
        value=default_context,
        height=250,
        help="This should be the scientific evidence retrieved by your RAG system."
    )

with col2:
    st.markdown("### ⚙️ Generation Settings")
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.1, step=0.05,
                            help="Lower = more deterministic, higher = more creative")
    max_tokens = st.slider("Max Tokens", min_value=500, max_value=3000, value=1500, step=100)

    st.markdown("### 🚀 Actions")
    generate_btn = st.button("📄 Generate Report", use_container_width=True)

    # Optional: upload context file
    uploaded_file = st.file_uploader("Or upload a context text file", type=["txt", "md"])
    if uploaded_file is not None:
        context_text = uploaded_file.read().decode("utf-8")
        st.success("Context loaded from file!")

# --- Generate and display ---
if generate_btn:
    if not context_text.strip():
        st.warning("Please provide a retrieved context.")
    else:
        with st.spinner("Generating your personalized nutrition report..."):
            try:
                # Build the prompt
                prompt = build_prompt(prediction=patient_condition, context=context_text)
                # Generate answer
                answer = generate_answer(prompt, temperature=temperature, max_tokens=max_tokens)
                if answer:
                    # Display the answer in a nice container
                    st.markdown("---")
                    st.markdown("## 📋 Generated Report")
                    # Use markdown to render the report
                    st.markdown(answer)
                    # Also provide a download button for the report
                    st.download_button(
                        label="⬇️ Download Report as Markdown",
                        data=answer,
                        file_name=f"diabetes_report_{patient_condition.replace(' ', '_')}.md",
                        mime="text/markdown"
                    )
                else:
                    st.error("Failed to generate answer. Please check your API key and network.")
            except Exception as e:
                st.error(f"An error occurred: {e}")

# --- Optional: Display retrieval quality if we had the data ---
# We could add a section to show grounding report, but we don't have the data.

# --- Footer ---
st.markdown("---")
st.caption("⚠️ This tool is for educational purposes only. Always consult a healthcare professional for medical advice.")

# ==============================================
# Run the app with: streamlit run app.py
# ==============================================