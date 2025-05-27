import streamlit as st
import sqlite3
from datetime import datetime
import os
import matplotlib.pyplot as plt
import pandas as pd
from fpdf import FPDF
import tempfile
import requests
from bs4 import BeautifulSoup
import re
from telethon.sync import TelegramClient
from telethon.tl.functions.messages import SearchRequest
from telethon.tl.types import InputPeerEmpty
import asyncio
import nest_asyncio
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import MultinomialNB

# Import spacy with error handling
try:
    import spacy

    nlp = spacy.load("en_core_web_sm")
except OSError:
    st.error("⚠️ spaCy English model not found. Please run: python -m spacy download en_core_web_sm")
    st.stop()
except ImportError:
    st.error("⚠️ spaCy not installed. Please run: pip install spacy")
    st.stop()

from PIL import Image

# Selenium import with error handling
try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service
    from webdriver_manager.chrome import ChromeDriverManager
except ImportError:
    st.error("⚠️ Selenium not installed. Please run: pip install selenium webdriver-manager")
    st.stop()

import time
import json
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import base64

nest_asyncio.apply()

# Configuration
DB_NAME = 'cases_streamlit.db'
TELEGRAM_API_ID = 'your_api_id'  # Replace with your actual API ID
TELEGRAM_API_HASH = 'your_api_hash'  # Replace with your actual API hash

st.set_page_config(page_title="Anti-Trafficking Intelligence Toolkit", layout="wide")
st.title("Anti-Trafficking Intelligence Toolkit 🇮🇱")

# Sidebar menu
menu = ["Forum Monitor", "Telegram Scan", "Evidence Capture", "Export Cases", "Case Clustering"]
choice = st.sidebar.selectbox("Menu", menu)


# Database setup
@st.cache_resource
def init_database():
    conn = sqlite3.connect(DB_NAME, check_same_thread=False)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS cases (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        source TEXT,
        url TEXT,
        content TEXT,
        date TEXT,
        risk_score REAL,
        named_entities TEXT
    )''')
    conn.commit()
    return conn


conn = init_database()


# Initialize ML model
@st.cache_resource
def init_model():
    model = make_pipeline(TfidfVectorizer(), LogisticRegression())
    # Expanded training data for better classification
    train_texts = [
        "escort service available",
        "forced prostitution case",
        "minor involved trafficking",
        "suspicious advertisement",
        "massage parlor services",
        "young girl available",
        "legitimate business ad",
        "normal dating service",
        "legal adult entertainment",
        "regular classified ad"
    ]
    train_labels = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0]  # 1 = high risk, 0 = low risk
    model.fit(train_texts, train_labels)
    return model


model = init_model()

# Forum Monitor Section
if choice == "Forum Monitor":
    st.subheader("Forum Monitor - סריקת פורומים")

    with st.expander("ℹ️ Instructions"):
        st.write("This tool monitors forums for suspicious content and calculates risk scores.")

    url = st.text_input("Forum URL:", value="http://www.sexondbar.com/forum36.asp")

    if st.button("🔍 Scan Forum"):
        try:
            with st.spinner("Scanning forum..."):
                headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
                r = requests.get(url, headers=headers, timeout=10)
                soup = BeautifulSoup(r.content, 'html.parser')
                posts = soup.find_all("td", class_="Forum_Subject")

                if not posts:
                    st.warning("No posts found. The forum structure might have changed.")

                for i, p in enumerate(posts[:10]):  # Limit to first 10 posts
                    text = p.text.strip()
                    if not text:
                        continue

                    link = "http://www.sexondbar.com/" + p.a['href'] if p.a else url

                    with st.container():
                        col1, col2 = st.columns([3, 1])

                        with col1:
                            st.markdown(f"**Post {i + 1}:** {text}")
                            st.markdown(f"🔗 [Link]({link})")

                            # Named entity recognition
                            doc = nlp(text)
                            ents = ", ".join([f"{ent.text} ({ent.label_})" for ent in doc.ents])

                            # Risk assessment
                            risk = model.predict_proba([text])[0][1] if len(model.classes_) > 1 else 0.5

                            # Display results with color coding
                            risk_color = "🔴" if risk > 0.7 else "🟡" if risk > 0.4 else "🟢"
                            st.write(f"{risk_color} **Risk Score:** {risk:.2f}")
                            if ents:
                                st.write(f"🏷️ **Entities:** {ents}")

                        with col2:
                            if st.button(f"💾 Save", key=f"save_{i}"):
                                c = conn.cursor()
                                c.execute('''INSERT INTO cases (source, url, content, date, risk_score, named_entities) 
                                           VALUES (?, ?, ?, ?, ?, ?)''',
                                          ("Forum", link, text, str(datetime.now()), risk, ents))
                                conn.commit()
                                st.success("✅ Saved!")

                    st.divider()

        except requests.RequestException as e:
            st.error(f"❌ Error accessing forum: {str(e)}")
        except Exception as e:
            st.error(f"❌ Unexpected error: {str(e)}")

# Telegram Scan Section
elif choice == "Telegram Scan":
    st.subheader("Telegram Keyword Scanner")

    with st.expander("⚠️ Setup Required"):
        st.write("""
        **Before using this feature:**
        1. Get Telegram API credentials from https://my.telegram.org
        2. Replace TELEGRAM_API_ID and TELEGRAM_API_HASH in the code
        3. Install telethon: `pip install telethon`
        """)

    keyword = st.text_input("🔍 Enter keyword (Hebrew supported):")
    channel = st.text_input("📱 Telegram channel username (without @):")

    if st.button("🚀 Scan Telegram"):
        if TELEGRAM_API_ID == 'your_api_id':
            st.error("❌ Please configure your Telegram API credentials first!")
        elif not keyword or not channel:
            st.warning("⚠️ Please enter both keyword and channel name")
        else:
            try:
                async def tele_search():
                    with st.spinner("Connecting to Telegram..."):
                        client = TelegramClient('session_name', TELEGRAM_API_ID, TELEGRAM_API_HASH)
                        await client.start()

                        messages = await client(SearchRequest(
                            peer=channel,
                            q=keyword,
                            filter=InputPeerEmpty(),
                            min_date=None,
                            max_date=None,
                            offset_id=0,
                            add_offset=0,
                            limit=10,
                            max_id=0,
                            min_id=0,
                            hash=0
                        ))

                        if messages.messages:
                            st.success(f"✅ Found {len(messages.messages)} messages")
                            for i, msg in enumerate(messages.messages):
                                st.write(f"**Message {i + 1}:** {msg.message}")
                                st.divider()
                        else:
                            st.info("No messages found with that keyword")

                        await client.disconnect()


                asyncio.run(tele_search())

            except Exception as e:
                st.error(f"❌ Telegram scan error: {str(e)}")

# Evidence Capture Section
elif choice == "Evidence Capture":
    st.subheader("📸 Capture Screenshot of URL")

    with st.expander("ℹ️ Requirements"):
        st.write("Chrome browser must be installed for screenshot capture.")

    target_url = st.text_input("🌐 Enter URL to capture:")

    if st.button("📷 Capture Screenshot"):
        if not target_url:
            st.warning("⚠️ Please enter a URL")
        else:
            try:
                with st.spinner("Capturing screenshot..."):
                    chrome_options = Options()
                    chrome_options.add_argument("--headless")
                    chrome_options.add_argument("--no-sandbox")
                    chrome_options.add_argument("--disable-dev-shm-usage")
                    chrome_options.add_argument("--window-size=1920,1080")

                    # Use webdriver-manager to handle ChromeDriver automatically
                    service = Service(ChromeDriverManager().install())
                    driver = webdriver.Chrome(service=service, options=chrome_options)

                    driver.get(target_url)
                    time.sleep(3)  # Wait for page to load

                    filename = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
                    driver.save_screenshot(filename)
                    driver.quit()

                    st.success("✅ Screenshot captured!")
                    st.image(filename, caption=f"Evidence: {target_url}")

                    # Option to save to database
                    if st.button("💾 Save to Case Database"):
                        c = conn.cursor()
                        c.execute('''INSERT INTO cases (source, url, content, date, risk_score, named_entities) 
                                   VALUES (?, ?, ?, ?, ?, ?)''',
                                  ("Screenshot", target_url, f"Screenshot of {target_url}",
                                   str(datetime.now()), 0.0, ""))
                        conn.commit()
                        st.success("✅ Saved to database!")

            except Exception as e:
                st.error(f"❌ Screenshot capture failed: {str(e)}")
                st.info("💡 Make sure Chrome is installed and try again")

# Export Cases Section
elif choice == "Export Cases":
    st.subheader("📊 Export Case Log")

    try:
        df = pd.read_sql_query("SELECT * FROM cases ORDER BY date DESC", conn)

        if df.empty:
            st.info("📝 No cases found in database")
        else:
            st.write(f"**Total Cases:** {len(df)}")

            # Display summary statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📈 High Risk Cases", len(df[df['risk_score'] > 0.7]))
            with col2:
                st.metric("🟡 Medium Risk Cases", len(df[(df['risk_score'] > 0.4) & (df['risk_score'] <= 0.7)]))
            with col3:
                st.metric("✅ Low Risk Cases", len(df[df['risk_score'] <= 0.4]))

            # Display data
            st.dataframe(df, use_container_width=True)

            # Export options
            col1, col2 = st.columns(2)

            with col1:
                if st.button("📄 Export to CSV"):
                    csv_file = "case_export.csv"
                    df.to_csv(csv_file, index=False)
                    st.success(f"✅ Exported to {csv_file}")

                    # Download button
                    with open(csv_file, "rb") as f:
                        st.download_button(
                            "⬇️ Download CSV",
                            f.read(),
                            file_name="cases.csv",
                            mime="text/csv"
                        )

            with col2:
                if st.button("📑 Export to PDF"):
                    try:
                        pdf = FPDF()
                        pdf.add_page()
                        pdf.set_font("Arial", size=10)

                        # Add title
                        pdf.set_font("Arial", 'B', 16)
                        pdf.cell(0, 10, "Anti-Trafficking Case Report", ln=True, align='C')
                        pdf.ln(10)

                        pdf.set_font("Arial", size=10)
                        for index, row in df.iterrows():
                            text = (
                                f"Date: {row['date']}\n"
                                f"Source: {row['source']}\n"
                                f"URL: {row['url']}\n"
                                f"Risk Score: {row['risk_score']:.2f}\n"
                                f"Content: {row['content'][:200]}...\n"
                                f"{'=' * 50}\n"
                            )
                            try:
                                pdf.multi_cell(0, 5, txt=text.encode('latin1', 'replace').decode('latin1'))
                                pdf.ln(5)
                            except:
                                pdf.multi_cell(0, 5, txt="[Content contains non-ASCII characters]")
                                pdf.ln(5)

                        pdf_path = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False).name
                        pdf.output(pdf_path)

                        with open(pdf_path, "rb") as f:
                            st.download_button(
                                "⬇️ Download PDF",
                                f.read(),
                                file_name="cases.pdf",
                                mime="application/pdf"
                            )
                        st.success("✅ PDF generated!")

                    except Exception as e:
                        st.error(f"❌ PDF generation failed: {str(e)}")

    except Exception as e:
        st.error(f"❌ Database error: {str(e)}")

# Case Clustering Section
elif choice == "Case Clustering":
    st.subheader("🔍 Case Clustering & Similarity Analysis")

    try:
        df = pd.read_sql_query("SELECT id, content, risk_score, source FROM cases", conn)

        if df.empty:
            st.info("📝 No cases found for clustering analysis")
        elif len(df) < 2:
            st.warning("⚠️ Need at least 2 cases for clustering analysis")
        else:
            with st.spinner("Analyzing cases..."):
                # TF-IDF vectorization
                tfidf = TfidfVectorizer(max_features=100, stop_words='english')
                tfidf_matrix = tfidf.fit_transform(df['content'].fillna(''))

                # Determine optimal number of clusters
                n_clusters = min(5, len(df), max(2, len(df) // 3))

                # K-means clustering
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                df['cluster'] = kmeans.fit_predict(tfidf_matrix)

                # Display results
                st.write(f"**Cases grouped into {n_clusters} clusters:**")

                # Cluster summary
                cluster_summary = df.groupby('cluster').agg({
                    'risk_score': ['mean', 'count'],
                    'source': lambda x: ', '.join(x.unique())
                }).round(2)

                cluster_summary.columns = ['Avg_Risk_Score', 'Case_Count', 'Sources']
                st.dataframe(cluster_summary)

                # Detailed view by cluster
                for cluster_id in sorted(df['cluster'].unique()):
                    with st.expander(f"🔍 Cluster {cluster_id} Details ({len(df[df['cluster'] == cluster_id])} cases)"):
                        cluster_data = df[df['cluster'] == cluster_id][['id', 'content', 'risk_score', 'source']]
                        st.dataframe(cluster_data, use_container_width=True)

                # Similarity matrix for small datasets
                if len(df) <= 20:
                    st.subheader("📊 Case Similarity Matrix")
                    similarity_matrix = cosine_similarity(tfidf_matrix)

                    fig, ax = plt.subplots(figsize=(10, 8))
                    im = ax.imshow(similarity_matrix, cmap='YlOrRd')
                    ax.set_title("Case Similarity Heatmap")
                    ax.set_xlabel("Case ID")
                    ax.set_ylabel("Case ID")
                    plt.colorbar(im)
                    st.pyplot(fig)

    except Exception as e:
        st.error(f"❌ Clustering analysis failed: {str(e)}")

# Cleanup
try:
    conn.close()
except:
    pass