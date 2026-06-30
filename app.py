import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# ─── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Flipkart Product Intelligence",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #f8f9fa; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #1a1a2e;
    }
    [data-testid="stSidebar"] * { color: #e0e0e0 !important; }
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stTextInput label { color: #a0aec0 !important; }

    /* Cards */
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
        border-left: 4px solid #2ecc71;
    }
    .metric-label { font-size: 0.8rem; color: #6c757d; text-transform: uppercase; letter-spacing: 0.5px; }
    .metric-value { font-size: 2rem; font-weight: 700; color: #1a1a2e; }

    /* Product card */
    .product-card {
        background: white;
        border-radius: 10px;
        padding: 1rem 1.25rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        border: 1px solid #e9ecef;
        margin-bottom: 0.75rem;
        transition: box-shadow 0.2s;
    }
    .product-card:hover { box-shadow: 0 4px 12px rgba(0,0,0,0.1); }

    /* Section headers */
    .section-header {
        font-size: 1.4rem;
        font-weight: 700;
        color: #1a1a2e;
        border-bottom: 2px solid #2ecc71;
        padding-bottom: 0.5rem;
        margin-bottom: 1rem;
    }

    /* Tags */
    .tag {
        display: inline-block;
        background: #e8f5e9;
        color: #2e7d32;
        border-radius: 20px;
        padding: 2px 10px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 4px;
    }
    .tag-orange { background: #fff3e0; color: #e65100; }

    /* Prediction result */
    .pred-box {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border-radius: 12px;
        padding: 1.5rem 2rem;
        color: white;
        text-align: center;
        margin-top: 1rem;
    }
    .pred-rating { font-size: 3rem; font-weight: 800; color: #2ecc71; }
    .pred-stars { font-size: 1.5rem; color: #f39c12; }

    /* Hide Streamlit branding */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─── Load & prepare data ────────────────────────────────────────────────────
@st.cache_data
def load_data():
    """Load and preprocess the Flipkart dataset."""
    try:
        df = pd.read_csv("cleaned_data.csv")
    except FileNotFoundError:
        st.error("⚠️ `cleaned_data.csv` not found. Please place it in the same directory as `app.py`.")
        st.stop()

    # Ensure numeric columns
    for col in ["selling_price", "mrp", "product_rating", "seller_rating"]:
        if df[col].dtype == object:
            df[col] = (
                df[col]
                .str.replace("₹", "", regex=False)
                .str.replace(",", "", regex=False)
                .astype(float)
            )

    # Fill missing numeric values
    df["mrp"] = df["mrp"].fillna(df["mrp"].mean())
    df["selling_price"] = df["selling_price"].fillna(df["selling_price"].mean())
    df["product_rating"] = df["product_rating"].fillna(df["product_rating"].mean())

    # Recompute discount columns
    df["discount"] = df["mrp"] - df["selling_price"]
    df["discount_percent"] = ((df["mrp"] - df["selling_price"]) / df["mrp"]) * 100

    # Product name for TF-IDF
    df["product_name"] = (
        df["category_1"].fillna("") + " " +
        df["category_2"].fillna("") + " " +
        df["category_3"].fillna("")
    )

    return df


@st.cache_resource
def build_similarity(df):
    """Build TF-IDF matrix and cosine similarity."""
    tfidf = TfidfVectorizer(stop_words="english")
    matrix = tfidf.fit_transform(df["product_name"])
    sim = cosine_similarity(matrix)
    return sim


@st.cache_resource
def build_model(df):
    """Train linear regression model for rating prediction."""
    X = df[["mrp", "discount_percent"]].fillna(0)
    y = df["product_rating"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    mse = mean_squared_error(y_test, pred)
    return model, mse


def recommend(query, df, similarity, n=5):
    """Return top-n similar products for a keyword query."""
    query = query.lower().strip()
    matches = df[df["product_name"].str.lower().str.contains(query, na=False)]
    if matches.empty:
        return None, "No products found matching that query."
    idx = matches.index[0]
    scores = list(enumerate(similarity[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1 : n + 1]
    product_indices = [i[0] for i in scores]
    return df.iloc[product_indices], None


def render_stars(rating):
    full = int(round(rating))
    return "⭐" * full + "☆" * (5 - full)


# ─── Load everything ────────────────────────────────────────────────────────
df = load_data()
similarity = build_similarity(df)
model, mse = build_model(df)


# ─── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛒 Flipkart ML")
    st.markdown("**Product Intelligence Dashboard**")
    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["🔍 Recommend", "📈 Predict Rating", "📊 Dataset Insights"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("**Dataset stats**")
    st.markdown(f"- Products: **{len(df):,}**")
    st.markdown(f"- Categories: **{df['category_1'].nunique()}**")
    st.markdown(f"- Avg. rating: **{df['product_rating'].mean():.2f}**")
    st.markdown(f"- Avg. discount: **{df['discount_percent'].mean():.1f}%**")
    st.markdown("---")
    st.markdown(
        "<small style='color:#555'>Model: TF-IDF + Cosine Similarity<br>"
        "Rating model: Linear Regression<br>"
        f"MSE: {mse:.4f}</small>",
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════
#  PAGE 1 — RECOMMENDATION
# ═══════════════════════════════════════════════════════════════════════════
if page == "🔍 Recommend":
    st.markdown('<p class="section-header">Product Recommendation Engine</p>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Total Products</div>'
            f'<div class="metric-value">{len(df):,}</div></div>',
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Avg. Discount</div>'
            f'<div class="metric-value">{df["discount_percent"].mean():.1f}%</div></div>',
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Avg. Rating</div>'
            f'<div class="metric-value">{df["product_rating"].mean():.2f}</div></div>',
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Unique Categories</div>'
            f'<div class="metric-value">{df["category_1"].nunique()}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Search
    query = st.text_input(
        "🔎 Search for a product (e.g. Cricket, Electronics, Toys, Mobiles…)",
        placeholder="Type a product name or category…",
    )

    num_recs = st.slider("Number of recommendations", 3, 10, 5)

    if query:
        recs, error = recommend(query, df, similarity, n=num_recs)

        if error:
            st.warning(error)
        else:
            st.success(f"✅ Found **{len(recs)}** recommendations for **'{query}'**")

            for _, row in recs.iterrows():
                disc_pct = row["discount_percent"]
                sp = row.get("selling_price", 0)
                mrp = row.get("mrp", 0)
                rating = row.get("product_rating", 0)
                title = row.get("title", "—")
                seller = row.get("seller_name", "Unknown seller")
                img = row.get("image_links", "")

                col_img, col_info = st.columns([1, 5])
                with col_img:
                    if isinstance(img, str) and img.startswith("http"):
                        st.image(img, width=90)
                    else:
                        st.markdown("🛒")
                with col_info:
                    st.markdown(
                        f"""<div class="product-card">
                        <strong>{str(title)[:100]}…</strong><br>
                        <span class="tag">{row["category_2"]}</span>
                        <span class="tag">{row["category_3"]}</span>
                        <span class="tag tag-orange">{disc_pct:.1f}% off</span><br>
                        <span style="font-size:1.1rem;font-weight:700;color:#1a1a2e;">₹{sp:,.0f}</span>
                        <span style="font-size:0.85rem;color:#aaa;text-decoration:line-through;margin-left:6px;">₹{mrp:,.0f}</span>
                        &nbsp;&nbsp; {render_stars(rating)} <span style="font-size:0.85rem;color:#6c757d;">{rating:.1f}</span><br>
                        <span style="font-size:0.8rem;color:#888;">Sold by {seller}</span>
                        </div>""",
                        unsafe_allow_html=True,
                    )
    else:
        # Show sample popular products
        st.markdown("#### 🔥 Sample products from dataset")
        sample = df.sample(6, random_state=7)
        cols = st.columns(3)
        for i, (_, row) in enumerate(sample.iterrows()):
            with cols[i % 3]:
                img = row.get("image_links", "")
                sp = row.get("selling_price", 0)
                disc_pct = row.get("discount_percent", 0)
                rating = row.get("product_rating", 0)
                title = row.get("title", "—")
                with st.container():
                    if isinstance(img, str) and img.startswith("http"):
                        st.image(img, use_container_width=True)
                    st.markdown(f"**{str(title)[:60]}…**")
                    st.markdown(
                        f'<span class="tag">{row["category_2"]}</span>'
                        f'<span class="tag tag-orange">{disc_pct:.0f}% off</span>',
                        unsafe_allow_html=True,
                    )
                    st.markdown(f"₹{sp:,.0f} · {render_stars(rating)} {rating:.1f}")


# ═══════════════════════════════════════════════════════════════════════════
#  PAGE 2 — RATING PREDICTOR
# ═══════════════════════════════════════════════════════════════════════════
elif page == "📈 Predict Rating":
    st.markdown('<p class="section-header">Product Rating Predictor</p>', unsafe_allow_html=True)
    st.markdown(
        "Enter a product's **MRP** and **discount %** to predict its expected rating "
        "using a Linear Regression model trained on the Flipkart dataset."
    )

    col_form, col_info = st.columns([1, 1])

    with col_form:
        st.markdown("#### Input features")
        mrp_input = st.number_input("MRP (₹)", min_value=0.0, value=1500.0, step=100.0)
        disc_input = st.number_input("Discount %", min_value=0.0, max_value=100.0, value=50.0, step=1.0)
        selling_price_calc = mrp_input * (1 - disc_input / 100)
        st.markdown(
            f"→ Selling price: **₹{selling_price_calc:,.2f}**",
            unsafe_allow_html=True,
        )

        predict_btn = st.button("⚡ Predict Rating", use_container_width=True)

        if predict_btn:
            X_input = np.array([[mrp_input, disc_input]])
            pred_rating = float(model.predict(X_input)[0])
            pred_rating = max(1.0, min(5.0, pred_rating))

            st.markdown(
                f"""<div class="pred-box">
                <div style="font-size:0.9rem;color:#a0aec0;margin-bottom:4px;">Predicted Rating</div>
                <div class="pred-rating">{pred_rating:.2f}</div>
                <div class="pred-stars">{render_stars(pred_rating)}</div>
                <div style="font-size:0.8rem;color:#718096;margin-top:8px;">
                  Based on MRP ₹{mrp_input:,.0f} · {disc_input:.1f}% discount
                </div>
                </div>""",
                unsafe_allow_html=True,
            )

    with col_info:
        st.markdown("#### Model details")
        info_data = {
            "Algorithm": "Linear Regression",
            "Features": "MRP, Discount %",
            "Target": "Product Rating (1–5)",
            "Train/test split": "80 / 20",
            f"MSE (test)": f"{mse:.4f}",
            "Training samples": f"{int(len(df) * 0.8):,}",
        }
        for k, v in info_data.items():
            st.markdown(f"- **{k}:** {v}")

        st.markdown("#### Distribution of actual ratings")
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.hist(df["product_rating"].dropna(), bins=20, color="#2ecc71", edgecolor="white", alpha=0.9)
        ax.set_xlabel("Rating")
        ax.set_ylabel("Count")
        ax.set_facecolor("#f8f9fa")
        fig.patch.set_facecolor("#f8f9fa")
        st.pyplot(fig)
        plt.close()


# ═══════════════════════════════════════════════════════════════════════════
#  PAGE 3 — DATASET INSIGHTS
# ═══════════════════════════════════════════════════════════════════════════
elif page == "📊 Dataset Insights":
    st.markdown('<p class="section-header">Dataset Insights</p>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📦 Categories", "⭐ Ratings", "💸 Pricing & Discounts"])

    # ── Tab 1: Categories ──────────────────────────────────────────────────
    with tab1:
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("##### Top category_1 (main categories)")
            top1 = df["category_1"].value_counts().head(8)
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.barplot(x=top1.values, y=top1.index, ax=ax, palette="Greens_r")
            ax.set_xlabel("Count")
            ax.set_ylabel("")
            ax.set_facecolor("#f8f9fa")
            fig.patch.set_facecolor("#f8f9fa")
            st.pyplot(fig)
            plt.close()

        with col_b:
            st.markdown("##### Top category_2 (sub-categories)")
            top2 = df["category_2"].value_counts().head(8)
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.barplot(x=top2.values, y=top2.index, ax=ax, palette="Blues_r")
            ax.set_xlabel("Count")
            ax.set_ylabel("")
            ax.set_facecolor("#f8f9fa")
            fig.patch.set_facecolor("#f8f9fa")
            st.pyplot(fig)
            plt.close()

        st.markdown("##### Top category_3 (leaf categories)")
        top3 = df["category_3"].value_counts().head(12)
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.barplot(x=top3.index, y=top3.values, ax=ax, palette="Purples_r")
        ax.set_xlabel("")
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=40)
        ax.set_facecolor("#f8f9fa")
        fig.patch.set_facecolor("#f8f9fa")
        st.pyplot(fig)
        plt.close()

    # ── Tab 2: Ratings ─────────────────────────────────────────────────────
    with tab2:
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("##### Product rating distribution")
            fig, ax = plt.subplots(figsize=(5, 3.5))
            ax.hist(df["product_rating"].dropna(), bins=25, color="#2ecc71", edgecolor="white")
            ax.set_xlabel("Rating")
            ax.set_ylabel("Count")
            ax.set_facecolor("#f8f9fa")
            fig.patch.set_facecolor("#f8f9fa")
            st.pyplot(fig)
            plt.close()

        with col_b:
            st.markdown("##### Seller rating distribution")
            fig, ax = plt.subplots(figsize=(5, 3.5))
            ax.hist(df["seller_rating"].dropna(), bins=25, color="#3498db", edgecolor="white")
            ax.set_xlabel("Seller Rating")
            ax.set_ylabel("Count")
            ax.set_facecolor("#f8f9fa")
            fig.patch.set_facecolor("#f8f9fa")
            st.pyplot(fig)
            plt.close()

        st.markdown("##### Average product rating by top category_2")
        top_cats = df["category_2"].value_counts().head(10).index
        avg_rating = (
            df[df["category_2"].isin(top_cats)]
            .groupby("category_2")["product_rating"]
            .mean()
            .sort_values(ascending=False)
        )
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.barplot(x=avg_rating.index, y=avg_rating.values, ax=ax, palette="YlOrRd_r")
        ax.set_ylim(0, 5)
        ax.set_xlabel("")
        ax.set_ylabel("Avg. Rating")
        ax.tick_params(axis="x", rotation=40)
        ax.set_facecolor("#f8f9fa")
        fig.patch.set_facecolor("#f8f9fa")
        st.pyplot(fig)
        plt.close()

    # ── Tab 3: Pricing & Discounts ─────────────────────────────────────────
    with tab3:
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("##### Discount % distribution")
            fig, ax = plt.subplots(figsize=(5, 3.5))
            ax.hist(
                df["discount_percent"].dropna().clip(0, 100),
                bins=30,
                color="#e74c3c",
                edgecolor="white",
                alpha=0.85,
            )
            ax.set_xlabel("Discount %")
            ax.set_ylabel("Count")
            ax.set_facecolor("#f8f9fa")
            fig.patch.set_facecolor("#f8f9fa")
            st.pyplot(fig)
            plt.close()

        with col_b:
            st.markdown("##### Discount % vs. Product rating")
            fig, ax = plt.subplots(figsize=(5, 3.5))
            sample_plot = df.sample(min(1000, len(df)), random_state=1)
            ax.scatter(
                sample_plot["discount_percent"],
                sample_plot["product_rating"],
                alpha=0.25,
                color="#9b59b6",
                s=18,
            )
            ax.set_xlabel("Discount %")
            ax.set_ylabel("Product Rating")
            ax.set_facecolor("#f8f9fa")
            fig.patch.set_facecolor("#f8f9fa")
            st.pyplot(fig)
            plt.close()

        st.markdown("##### Top 10 sellers by product count")
        top_sellers = df["seller_name"].value_counts().head(10)
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.barplot(x=top_sellers.values, y=top_sellers.index, ax=ax, palette="crest")
        ax.set_xlabel("Products listed")
        ax.set_ylabel("")
        ax.set_facecolor("#f8f9fa")
        fig.patch.set_facecolor("#f8f9fa")
        st.pyplot(fig)
        plt.close()