import streamlit as st
import pandas as pd
import time
import os
from dotenv import load_dotenv

# ============================================================
# 1. استيراد وحدات المشروع الجديدة
# ============================================================
# تأكد من وجود هذه الملفات في نفس المسار:
# query_builder.py, intent_detector.py, context_builder.py,
# 06_retrieve_context.py, 07_prompting.py

from query_builder import build_query
from intent_detector import detect_intent
from context_builder import build_context_package
from retrieval_report import retrieval_quality_report  # لو موجود، وإلا استخدم الدالة المدمجة

# استيراد دوال الاسترجاع والتوليد من ملفاتك
# (افترض أن 06_retrieve_context.py يحتوي على retrieve_top_k_hybrid و rerank_candidates)
from _06_retrieve_context import (
    retrieve_top_k_hybrid,
    rerank_candidates,
    tfidf_vectorizer,
    tfidf_matrix,
    bm25,
    embedding_model,
    embedding_matrix,
    chunks_df
)

from _07_prompting import build_prompt, generate_answer

load_dotenv()

# ============================================================
# 2. إعدادات الصفحة والـ CSS
# ============================================================
st.set_page_config(
    page_title="🍏 RAG Diabetes Nutrition Assistant",
    page_icon="🍏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS مخصص للشكل الاحترافي
st.markdown("""
<style>
    .reportview-container { background: #f8fafc; }
    .main { padding: 1rem 2rem; }
    h1, h2, h3 { color: #0b3b5c; font-weight: 600; }
    .stButton > button {
        background-color: #0b3b5c; color: white; font-weight: 600;
        border-radius: 30px; padding: 0.5rem 2rem; border: none;
        transition: 0.3s;
    }
    .stButton > button:hover {
        background-color: #1a5276; transform: scale(1.02);
        box-shadow: 0 4px 12px rgba(11, 59, 92, 0.2);
    }
    .stTextInput > div > div > input {
        border-radius: 20px; border: 1px solid #d1d5db;
        padding: 0.75rem 1rem; font-size: 1rem;
    }
    .stTextArea > div > div > textarea {
        border-radius: 20px; border: 1px solid #d1d5db;
        font-size: 0.95rem;
    }
    .css-1d391kg {
        background-color: #ffffff; border-radius: 20px;
        padding: 1.5rem; box-shadow: 0 4px 20px rgba(0,0,0,0.04);
    }
    .sidebar .sidebar-content { background: #ffffff; border-right: 1px solid #e2e8f0; }
    .big-number { font-size: 2.5rem; font-weight: 700; color: #0b3b5c; }
    .metric-box {
        background: #f1f5f9; padding: 0.75rem 1.5rem; border-radius: 40px;
        text-align: center; margin: 0.5rem 0;
    }
    .stExpander { border: 1px solid #e2e8f0; border-radius: 12px; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# 3. الشريط الجانبي (الإعدادات والمدخلات)
# ============================================================
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/diabetes.png", width=80)
    st.title("🍏 Diabetes Assistant")
    st.markdown("---")
    
    # اختيار نوع السكري
    diabetes_type = st.selectbox(
        "🩺 نوع السكري",
        options=["Type 2 Diabetes", "Type 1 Diabetes", "Pre-Diabetes", "Gestational Diabetes", "Custom"],
        index=0
    )
    
    # السؤال المخصص
    user_question = st.text_area(
        "💬 اسأل عن النظام الغذائي",
        value="What are the current dietary recommendations for Type 2 diabetes, including carbohydrate intake, dietary fat, fibre and weight management?",
        height=120
    )
    
    st.markdown("---")
    st.caption("⚙️ الإعدادات (تم ضبطها مسبقاً للحصول على أفضل سياق)")
    st.caption(f"📚 عدد القطع المسترجعة: 60")
    st.caption(f"📌 عدد القطع المعاد ترتيبها: 15")
    st.caption(f"📄 حجم السياق: 3000 كلمة")
    
    generate_btn = st.button("🚀 إنشاء التقرير", use_container_width=True)

# ============================================================
# 4. المنطق الرئيسي (عند الضغط على الزر)
# ============================================================
if generate_btn:
    if not user_question.strip():
        st.warning("⚠️ من فضلك أدخل سؤالاً.")
    else:
        with st.spinner("⏳ جاري تحليل السؤال واسترجاع المعلومات..."):
            # 4.1 بناء الاستعلام الموسع
            query = build_query(diabetes_type) if diabetes_type != "Custom" else user_question
            st.session_state['query'] = query
            
            # 4.2 كشف النية (اختياري للتوسع لاحقاً)
            intent = detect_intent(user_question)
            st.session_state['intent'] = intent
            
            # 4.3 الاسترجاع الهجين (k=60)
            results = retrieve_top_k_hybrid(
                query=query,
                tfidf_vectorizer=tfidf_vectorizer,
                tfidf_matrix=tfidf_matrix,
                bm25=bm25,
                embedding_model=embedding_model,
                embedding_matrix=embedding_matrix,
                chunks_df=chunks_df,
                tfidf_weight=0.34,
                bm25_weight=0.33,
                semantic_weight=0.33,
                k=60  # ← كما طلبنا
            )
            
            # 4.4 إعادة الترتيب (top_n=15)
            reranked = rerank_candidates(
                query=query,
                candidates_df=results,
                top_n=15  # ← كما طلبنا
            )
            
            # 4.5 بناء السياق (بالمعلمات المحسنة)
            context = build_context_package(
                query=query,
                reranked_df=reranked,
                max_context_chunks=12,
                word_budget=3000,
                min_score_ratio=0.25
            )
            
            st.session_state['context'] = context
            st.session_state['reranked'] = reranked
            
            # 4.6 التحقق من الثقة
            best_score = reranked.iloc[0]["rerank_score"] if not reranked.empty else 0
            THRESHOLD = 3.0
            
            if best_score < THRESHOLD:
                st.error("❌ مستوى الثقة منخفض جداً (أقل من 3.0). تأكد من وجود سياق كافٍ.")
                st.stop()
            
            # 4.7 بناء الـ Prompt
            prompt = build_prompt(
                prediction=diabetes_type,
                context=context["context_text"]
            )
            
            # 4.8 توليد الإجابة
            with st.spinner("🧠 جاري توليد التقرير المخصص..."):
                answer = generate_answer(
                    prompt=prompt,
                    temperature=0.1,
                    max_tokens=1500,
                    seed=42
                )
            
            st.session_state['answer'] = answer
            st.session_state['prompt'] = prompt

# ============================================================
# 5. عرض النتائج (إذا كانت موجودة في Session State)
# ============================================================
if 'answer' in st.session_state and st.session_state['answer']:
    answer = st.session_state['answer']
    context = st.session_state.get('context', {})
    reranked = st.session_state.get('reranked', pd.DataFrame())
    
    # 5.1 عرض التقرير
    st.markdown("---")
    st.markdown("## 📋 التقرير النهائي")
    st.markdown(answer)
    
    # 5.2 أزرار تحميل
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            label="⬇️ تحميل التقرير (Markdown)",
            data=answer,
            file_name=f"diabetes_report_{diabetes_type.replace(' ', '_')}.md",
            mime="text/markdown"
        )
    with col2:
        if 'prompt' in st.session_state:
            st.download_button(
                label="⬇️ تحميل الـ Prompt (للتطوير)",
                data=st.session_state['prompt'],
                file_name="prompt.txt",
                mime="text/plain"
            )
    
    # 5.3 لوحة المصادر والتقييم (في الأسفل)
    with st.expander("🔍 عرض المصادر المسترجعة (Reranked)", expanded=False):
        if not reranked.empty:
            display_cols = ["chunk_id", "rerank_score", "chunk_text"]
            st.dataframe(
                reranked[display_cols].head(10),
                use_container_width=True,
                height=400
            )
    
    with st.expander("📊 تقرير جودة الاسترجاع (Retrieval Report)", expanded=False):
        if not reranked.empty:
            # حساب الإحصائيات
            num_chunks = len(reranked)
            avg_score = reranked["rerank_score"].mean()
            max_score = reranked["rerank_score"].max()
            min_score = reranked["rerank_score"].min()
            
            # دالة الثقة
            def conf_label(s):
                return "High" if s >= 2.5 else "Medium" if s >= 1.0 else "Low"
            
            col_a, col_b, col_c, col_d = st.columns(4)
            col_a.metric("📦 عدد القطع", num_chunks)
            col_b.metric("⭐ أعلى درجة", f"{max_score:.2f}")
            col_c.metric("📉 متوسط الدرجة", f"{avg_score:.2f}")
            col_d.metric("🔒 مستوى الثقة", conf_label(max_score))
            
            st.caption("ملاحظة: تم استخدام المعاملات المُحسّنة (k=60, top_n=15, budget=3000, ratio=0.25)")
    
    # 5.4 عرض السياق الخام (للتدقيق)
    with st.expander("📄 عرض السياق الخام (Context Text)", expanded=False):
        st.text_area("النص الكامل للسياق", context.get("context_text", ""), height=300)
    
    # 5.5 تقييم التوثيق (Grounding) اختياري
    with st.expander("📈 تقييم التوثيق (Grounding Score)", expanded=False):
        try:
            from sklearn.metrics.pairwise import cosine_similarity
            emb = embedding_model.encode([answer, context.get("context_text", "")], convert_to_numpy=True, normalize_embeddings=True)
            sim = cosine_similarity([emb[0]], [emb[1]])[0][0]
            
            quality = "Excellent" if sim >= 0.8 else "Good" if sim >= 0.6 else "Moderate" if sim >= 0.4 else "Poor"
            st.metric("🔗 التشابه الدلالي", f"{sim:.3f}")
            st.caption(f"جودة التوثيق: **{quality}**")
        except Exception as e:
            st.warning(f"لم نتمكن من حساب درجة التوثيق: {e}")

else:
    # رسالة ترحيب عند أول فتح للتطبيق
    st.markdown("""
    <div style="text-align: center; margin-top: 50px;">
        <h1>🩺 مرحباً بك في مساعد التغذية للسكري</h1>
        <p style="color: #475569; font-size: 1.2rem;">
            اختر نوع السكري من القائمة الجانبية، وأدخل سؤالك، ثم اضغط على <strong>"إنشاء التقرير"</strong>.
        </p>
        <p style="color: #94a3b8; font-size: 0.9rem;">
            يعتمد هذا النظام على تقنية RAG مع استرجاع هجين (TF-IDF + BM25 + Embeddings) وإعادة ترتيب باستخدام Cross-Encoder.
        </p>
    </div>
    """, unsafe_allow_html=True)
