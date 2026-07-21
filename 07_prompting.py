# ==================================================
# STAGE 15: PROMPT BUILDER (OPTIMIZED V5 - REALISTIC MEALS)
# ==================================================

def build_prompt(prediction: str, context: str) -> str:
    if not context or not context.strip():
        raise ValueError("Retrieved context is empty or invalid.")

    # Clean inputs
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



    import requests
import time

# ==================================================
# OLLAMA CONFIGURATION
# ==================================================

from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Read API key from .env
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Read model name from .env
MODEL_NAME = os.getenv("MODEL_NAME")

# OpenRouter endpoint
API_URL = "https://openrouter.ai/api/v1/chat/completions"


# ==================================================
# LLM Generation (OpenRouter)
# ==================================================

def generate_answer(
    prompt,
    model=MODEL_NAME,
    temperature=0.1,
    max_tokens=1200
):

    headers = {

        "Authorization": f"Bearer {OPENROUTER_API_KEY}",

        "Content-Type": "application/json"

    }

    payload = {

        "model": model,

        "messages": [

            {
                "role": "user",
                "content": prompt
            }

        ],

        "temperature": temperature,

        "max_tokens": max_tokens

    }

    start = time.time()

    try:

        response = requests.post(

            API_URL,

            headers=headers,

            json=payload,

            timeout=600

        )

        response.raise_for_status()

        elapsed = time.time() - start

        result = response.json()

        answer = result["choices"][0]["message"]["content"].strip()

        print("=" * 60)
        print("OPENROUTER GENERATION")
        print("=" * 60)
        print(f"Model          : {model}")
        print(f"Inference Time : {elapsed:.2f} sec")

        usage = result.get("usage")

        if usage:

            print(f"Prompt Tokens     : {usage.get('prompt_tokens')}")
            print(f"Completion Tokens : {usage.get('completion_tokens')}")
            print(f"Total Tokens      : {usage.get('total_tokens')}")

        return answer

    except Exception as e:

        print("=" * 60)
        print("OPENROUTER ERROR")
        print("=" * 60)
        print(e)

        return None


# ==================================================
# GENERATE FINAL RAG ANSWER
# ==================================================

best_score = reranked.iloc[0]["rerank_score"]

THRESHOLD = 3.0


if best_score >= THRESHOLD:

    prompt = build_prompt(

        prediction=prediction,

        context=context["context_text"]

    )

    answer = generate_answer(

        prompt=prompt,

        temperature=0.1,

        max_tokens=1200,

        seed=42

    )

else:

    answer = (
        "Low-confidence retrieval. "
        "No reliable answer could be generated from the retrieved document."
    )


# ==================================================
# FINAL OUTPUT
# ==================================================

print("=" * 60)
print("FINAL RAG RESULT")
print("=" * 60)

print(f"Patient Condition : {prediction}")
print(f"Retrieved Sources : {context['num_sources']}")
print(f"Context Words     : {context['used_words']}")
print(f"Top Rerank Score  : {best_score:.3f}")

print("=" * 60)
print("GENERATED ANSWER")
print("=" * 60)

print(answer)




from IPython.display import Markdown, display

print("=" * 60)
print("GENERATED ANSWER")
print("=" * 60)

if "answer" not in globals():

    print("❌ Variable 'answer' does not exist.")

elif answer is None or answer.strip() == "":

    print("❌ No answer generated.")

else:

    display(Markdown(answer))


    import pandas as pd

# ==================================================
# STAGE 18: RAG GROUNDING REPORT
# ==================================================

def confidence_label(score):

    if score >= 2.5:
        return "High"

    elif score >= 1.0:
        return "Medium"

    else:
        return "Low"


# --------------------------------------------------
# Build Grounding Report
# --------------------------------------------------

grounded_sources = context["selected_df"][
    ["chunk_id", "rerank_score"]
].copy()

grounded_sources["confidence"] = grounded_sources[
    "rerank_score"
].apply(confidence_label)


# --------------------------------------------------
# Summary Statistics
# --------------------------------------------------

num_sources = len(grounded_sources)

avg_score = grounded_sources["rerank_score"].mean()

max_score = grounded_sources["rerank_score"].max()

min_score = grounded_sources["rerank_score"].min()


# --------------------------------------------------
# Display Report
# --------------------------------------------------

print("=" * 60)
print("RAG GROUNDING REPORT")
print("=" * 60)

print(f"Retrieved Sources : {num_sources}")
print(f"Average Score     : {avg_score:.3f}")
print(f"Highest Score     : {max_score:.3f}")
print(f"Lowest Score      : {min_score:.3f}")

print()

print("Selected Evidence Chunks")
print("-" * 60)

display(grounded_sources)

print()

print("Grounding Status")
print("-" * 60)
print(
    "The final nutrition report was generated only from the "
    "retrieved and reranked evidence shown above."
)





# ==================================================
# STAGE: RETRIEVAL QUALITY REPORT
# ==================================================

def retrieval_quality_report(final_context):

    print("=" * 60)
    print("RAG RETRIEVAL QUALITY REPORT")
    print("=" * 60)

    num_chunks = len(final_context)
    unique_chunks = final_context["chunk_id"].nunique()

    best_score = final_context["rerank_score"].max()
    avg_score = final_context["rerank_score"].mean()
    worst_score = final_context["rerank_score"].min()

    # Confidence Level
    if best_score >= 2.5:
        confidence = "High"
    elif best_score >= 1.0:
        confidence = "Medium"
    else:
        confidence = "Low"

    print(f"Retrieved Chunks : {num_chunks}")
    print(f"Unique Chunks    : {unique_chunks}")
    print(f"Best Score       : {best_score:.3f}")
    print(f"Average Score    : {avg_score:.3f}")
    print(f"Worst Score      : {worst_score:.3f}")
    print(f"Confidence Level : {confidence}")

    return {
        "retrieved_chunks": num_chunks,
        "unique_chunks": unique_chunks,
        "best_score": best_score,
        "average_score": avg_score,
        "worst_score": worst_score,
        "confidence": confidence
    }


# ==================================================
# RUN
# ==================================================

retrieval_report = retrieval_quality_report(context["selected_df"])



# ==================================================
# Answer Grounding Evaluation
# ==================================================

from sklearn.metrics.pairwise import cosine_similarity


def evaluate_answer(answer, context_text):

    if not answer or not context_text:
        raise ValueError("Answer or retrieved context is empty.")

    embeddings = embedding_model.encode(
        [answer, context_text],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    similarity = cosine_similarity(
        [embeddings[0]],
        [embeddings[1]]
    )[0][0]

    print("=" * 60)
    print("ANSWER GROUNDING EVALUATION")
    print("=" * 60)
    print(f"Semantic Similarity : {similarity:.3f}")

    if similarity >= 0.80:
        quality = "Excellent"
    elif similarity >= 0.60:
        quality = "Good"
    elif similarity >= 0.40:
        quality = "Moderate"
    else:
        quality = "Poor"

    print(f"Grounding Quality   : {quality}")

    return similarity


# ==================================================
# RUN
# ==================================================

grounding_score = evaluate_answer(
    answer,
    context["context_text"]
)