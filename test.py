import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sentence_transformers import SentenceTransformer

# Example DataFrame
df = pd.DataFrame({
    'latitude': [0, 1, 2],
    'longitude': [70, 71, 72],
    'salinity': [35, 35.1, 34.9],
    'temperature': [22, 22.5, 21.8],
    'time': ['2023-03-01']*3,
    'description': [
        "Surface water profile near 70°E",
        "10m depth reading in the Indian Ocean",
        "20m depth temperature and salinity data"
    ]
})

# 1️⃣ Initialize embedding model (only once)
model = SentenceTransformer("all-MiniLM-L6-v2")

# Precompute embeddings for descriptions
df["embedding"] = df["description"].apply(lambda x: model.encode([x])[0])

# 2️⃣ Pandas-based query (replacing Chroma)
def pandas_query(query: str, n_results: int = 5):
    """
    Mimics Chroma query but uses pandas DataFrame.
    Returns top n_results rows most similar to query.
    """
    query_emb = model.encode([query])[0]

    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    df["similarity"] = df["embedding"].apply(lambda x: cosine_similarity(query_emb, x))

    return df.sort_values(by="similarity", ascending=False).head(n_results)




# --- Initialize Session State ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Custom CSS for styling tabs like your image
st.markdown(
    """
    <style>
    /* Center the tabs */
    div[data-baseweb="tab-list"] {
        display: flex;
        justify-content: center;
        background: rgba(255, 255, 255, 0.08); /* transparent white */
        backdrop-filter: blur(10px);           /* blur effect */
        -webkit-backdrop-filter: blur(10px);   /* for Safari */
        padding: 8px;
        border-radius: 25px;
        border: 1px solid black;
        gap: 10px;
        width:70%;
        margin-left:15%;
    }

    /* Default tab */
    button[data-baseweb="tab"] {
        color: white !important;
        font-weight: 600 !important;
        border-radius: 20px !important;
        padding: 6px 16px !important;
        background: transparent !important;
    }

    /* Active tab */
    button[data-baseweb="tab"][aria-selected="true"] {
        background-color: black !important;
        color: white !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Tabs
table_home, table_dashboard, table_chat, table_contact, table_team = st.tabs(
    ["Home", "Dashboard", "Chatbot", "Contact Us", "Our Team"]
)
















# =========================
# HOME TAB WITH BACKGROUND
# =========================
with table_home:

 st.markdown("""
 <style>
 /* Background Image */
 .stApp {
    background: url('download (86).jpeg') no-repeat center center fixed;
    background-size: cover;
    color: white;
    font-family: 'Arial', sans-serif;
}

  /* Hero Section */
  .hero {
    text-align: center;
    margin-top: 12%;
}
 .hero h1 {
    font-size: 90px;
    font-weight: 900;
    margin: 0;
}
.hero h2 {
    font-size: 40px;
    letter-spacing: 8px;
    font-weight: 200;
    margin-top: -20px;
}
.explore-btn {
    display: inline-block;
    margin-top: 30px;
    padding: 12px 28px;
    border: 1px solid white;
    border-radius: 25px;
    font-size: 18px;
    transition: 0.3s;
    text-decoration: none;   /* removes underline */
    color: white;            /* makes text white */
}
.explore-btn:hover {
    background: white;
    color: black;
}

/* Social Media Sidebar */
.social-sidebar {
    position: fixed;
    top: 50%;
    left: 20px;
    transform: translateY(-50%);
    writing-mode: vertical-rl;
    text-orientation: mixed;
    display: flex;
    gap: 20px;
}
.social-sidebar a {
    text-decoration: none;
    color: white;
    font-size: 14px;
    font-weight: 500;
    transition: 0.3s;
}
.social-sidebar a:hover {
    color: #00f2fe;
}

/* Footer */
.footer {
    position: fixed;
    bottom: 20px;
    left: 20px;
    font-size: 12px;
    color: rgba(255, 255, 255, 0.7);
}
 </style>
 """, unsafe_allow_html=True)

# =========================
# HERO SECTION
# =========================
 st.markdown("""
 <div class="hero">
    <h1>WATER</h1>
    <h2>SPACE</h2>
    <a class="explore-btn" href="#">EXPLORE</a>
 </div>
 """, unsafe_allow_html=True)

# =========================
# SOCIAL MEDIA LINKS
# =========================
 st.markdown("""
 <div class="social-sidebar">
    <a href="#">FACEBOOK</a>
    <a href="#">TWITTER</a>
    <a href="#">INSTAGRAM</a>
 </div>
 """, unsafe_allow_html=True)



# =========================
# FOOTER
# =========================
 st.markdown("""
 <div class="footer">
    @Argonauts
 </div>
 """, unsafe_allow_html=True)
























# --- Contact Tab ---
with table_contact:
    st.markdown("<style> body { background-color: #0d001a; color: white; } </style>", unsafe_allow_html=True)
    
    st.markdown("""
    <div style='text-align: center; padding: 40px 0;'>
        <h1 style='color: white;'>Contact Us</h1>
        <p style='color: #ccc;'>Start the conversation to establish good relationship and business.</p>
    </div>
    """, unsafe_allow_html=True)

    left_col, right_col = st.columns([1, 1.2])

    # === LEFT COLUMN (Contact Info) ===
    with left_col:
        st.markdown("### Get in Touch")
        st.markdown("## Seamless Communication, Global Impact.")
        st.markdown(
            "Connect with us to explore real-time ARGO ocean data, collaborate, or share feedback on our AI-powered dashboard."
        )

        st.markdown("#### Location")
        st.markdown("Manipal University Jaipur, Rajasthan")

        st.markdown("#### Let's Talk")
        st.markdown("Phone: +918527919739")

        st.markdown("#### Email Support")
        st.markdown("official.theargonauts@gmail.com  \nraimayankofficial@gmail.coma")

        st.markdown("#### Working Hours")
        st.markdown("Monday - Friday  \n07am - 09pm")

        # Social media icons
        st.markdown("#### Follow our social media")
        icons = {
            "instagram": "https://instagram-brands-solid-full.tiiny.site/instagram-brands-solid-full.png",
            "linkedin": "https://i.postimg.cc/vZ5q3Hbn/square-linkedin-brands-solid-full.png",
            "youtube": "https://i.postimg.cc/hjY1KHKg/youtube-brands-solid-full.png"
        }

        cols = st.columns(len(icons))
        for i, (name, url) in enumerate(icons.items()):
            with cols[i]:
                st.markdown(
                    f"<a href='#'><img src='{url}' width='32' style='margin: 5px'></a>",
                    unsafe_allow_html=True
                )

    # === RIGHT COLUMN (Form) ===
    with right_col:
        st.markdown("## Send us a message")
        st.markdown("Please feel free to send us any questions, feedback or suggestions you might have.")

        with st.form(key="contact_form"):
            name = st.text_input("Name", key="name")
            company = st.text_input("Company", key="company")
            phone = st.text_input("Phone", key="phone")
            email = st.text_input("Email", key="email")
            subject1 = st.text_input("Subject", key="subject1")
            message = st.text_area("Message", height=150, key="message")

            submit_button = st.form_submit_button(label="📨 Send Message")

            if submit_button:
                st.success("Thanks for reaching out! We'll get back to you soon.")
with table_chat:
    st.markdown("<h1 style='text-align:left;'>FloatChat</h1>", unsafe_allow_html=True)

    if "placeholder_text" not in st.session_state:
        st.session_state.placeholder_text = "Type your message..."

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Make the chat area wider with columns
    chat_col, _ = st.columns([4, 1])  # 4:1 width ratio

    with chat_col:
        # Display chat history messages
        for chat in st.session_state.chat_history:
            if chat["role"] == "user":
                with st.chat_message("user"):
                    st.markdown(chat["message"])
            else:
                with st.chat_message("assistant"):
                    st.markdown(chat["message"])

    # --- Chat input ---
    if user_input := st.chat_input(st.session_state.placeholder_text):
        st.session_state.chat_history.append({"role": "user", "message": user_input})
        st.session_state.chat_history.append({"role": "bot", "message": "🤖 I'm just a demo bot!"})
        st.session_state.placeholder_text = "Ask a Follow-up..."

    # --- Prebuilt Questions Section ---
    st.markdown(" Quick Questions")
    preset_questions = [
        "Show both chlorophyll and oxygen levels vs depth.",
        "Plot temperature vs depth for floats around 70°E longitude.",
        "Give me metadata about float 4901234.",
    ]

    # Show buttons in a row
    button_per_row = 2
    for i in range(0, len(preset_questions), button_per_row):
        q_cols = st.columns(button_per_row)
        for j, col in enumerate(q_cols):
            if (i + j) < len(preset_questions):
                q = preset_questions[i + j]
                if col.button(q):
                    st.session_state.chat_history.append({"role": "user", "message": q})
                    st.session_state.chat_history.append({"role": "bot", "message": f"🤖 Answer to: {q}"})
                    st.session_state.placeholder_text = "Ask a Follow-up..."
                    st.rerun()

with table_team:
    st.markdown("<h1 style='text-align:center; color:white;'>👨‍💻 Our Team</h1>", unsafe_allow_html=True)

    # --- Custom CSS for Team Cards ---
    st.markdown(
        """
        <style>
        .team-card {
            background-color: rgba(20, 40, 50, 0.9); /* same dark grid as input boxes */
            border: 1px solid rgba(124, 249, 214, 0.4);
            border-radius: 12px;
            padding: 16px;
            margin: 12px 0;
            color: black;
            text-align: center;
            box-shadow: 0 0 12px rgba(124, 249, 214, 0.15);
            transition: 0.3s;
        }
        .team-card:hover {
            box-shadow: 0 0 20px #7cf9d6;
            transform: scale(1.02);
        }
        .team-card h3 {
            color: #7cf9d6;
            margin-bottom: 6px;
        }
        .team-card h4 {
            color: #bdebe1;
            font-size: 16px;
            margin-bottom: 10px;
        }
        .team-card p {
            color: #a8cfc9;
            font-size: 14px;
        }
        .team-icons img {
            margin: 5px;
            transition: 0.3s;
        }
        .team-icons img:hover {
            transform: scale(1.15);
            filter: drop-shadow(0 0 6px #7cf9d6);
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Sample team data
    team = [
        {"name": "Mayank Rai", "role": "Backend Developer", "desc": "Passionate about building scalable web apps.",
         "linkedin": "#", "github": "#", "instagram": "#"},
        {"name": "Aarohi Dand", "role": "AI Engineer", "desc": "Designs user-friendly and modern interfaces.",
         "linkedin": "#", "github": "#", "instagram": "#"},
        {"name": "Kriti Saraogi", "role": "Frontend Developer", "desc": "Loves turning data into actionable insights.",
         "linkedin": "#", "github": "#", "instagram": "#"},
        {"name": "Sumit Sharma", "role": "Frontend Developer", "desc": "Works on machine learning and NLP models.",
         "linkedin": "#", "github": "#", "instagram": "#"},
        {"name": "Akshat Kumar", "role": "Frontend Developer", "desc": "Specializes in APIs and database management.",
         "linkedin": "#", "github": "#", "instagram": "#"},
        {"name": "Chinmayee Khanna", "role": "Researcher", "desc": "Keeps the team on track and organized.",
         "linkedin": "#", "github": "#", "instagram": "#"},
    ]

    # Icon links
    icons = {
    "linkedin": "https://i.postimg.cc/vZ5q3Hbn/square-linkedin-brands-solid-full.png",
    "github": "https://i.postimg.cc/X7R1fyS3/github-brands-solid-full.png",
    "instagram": "https://instagram-brands-solid-full.tiiny.site/instagram-brands-solid-full.png"
}


    # Layout: 3 cards per row
    for i in range(0, len(team), 3):
        cols = st.columns(3)
        for j, member in enumerate(team[i:i+3]):
            with cols[j]:
                st.markdown(
                    f"""
                    <div class="team-card">
                        <h3>{member['name']}</h3>
                        <h4>{member['role']}</h4>
                        <div class="team-icons">
                            <a href='{member['linkedin']}' target="_blank">
                                <img src='{icons['linkedin']}' width='28'>
                            </a>
                            <a href='{member['github']}' target="_blank">
                                <img src='{icons['github']}' width='28'>
                            </a>
                            <a href='{member['instagram']}' target="_blank">
                                <img src='{icons['instagram']}' width='28'>
                            </a>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )












st.markdown(
    """
    <style>
    .stApp {
        /* Deep ocean gradient */
        background: radial-gradient(
            circle at center bottom,
            rgba(124, 249, 214, 0.6) 0%,
            rgba(111, 238, 183, 0.3) 20%,
            rgba(127, 204, 182, 0.15) 40%,
            rgba(0, 40, 60, 0.9) 80%,
            rgba(0, 20, 30, 1) 100%
        ),
        linear-gradient(to top, rgba(0, 40, 60, 0.9), rgba(0, 0, 0, 0.95));
        
        background-size: cover;
        background-attachment: fixed;
        color: #ffffff;
    }

    /* Optional glowing pulse animation */
    @keyframes glowPulse {
        0% { background-position: 50% 100%; }
        50% { background-position: 50% 90%; }
        100% { background-position: 50% 100%; }
    }

    .stApp {
        animation: glowPulse 10s ease-in-out infinite;
    }
    </style>
    """,
    unsafe_allow_html=True
)








with table_dashboard:
    # ================== THEME & CSS ==================
    st.markdown("""
        <style>
            /* page bg */
            

            /* layout helpers */
            .card {
                background: #ffffff;
                border-radius: 22px;
                padding: 20px;
                box-shadow: 0 10px 30px rgba(20, 20, 60, 0.08);
            }
            .card-tight { padding:16px; }

            .grad-purple {
                background: linear-gradient(135deg, #6c63ff 0%, #9a7cff 45%, #f47cc3 100%);
                color: #fff;
            }
            .grad-violet {
                background: linear-gradient(135deg, #7c6cfb 0%, #5c49e0 100%);
                color: #fff;
            }
            .grad-pink {
                background: linear-gradient(135deg, #ff7da0 0%, #ff9fb6 100%);
                color: #fff;
            }

            .kpi { font-size: 14px; opacity: .9; margin-bottom: 2px; }
            .kpi-value { font-size: 26px; font-weight: 800; margin-bottom: 4px; }

            .pill {
                display:inline-block; padding:8px 12px; border-radius: 999px;
                background: #f0f3fb; color:#39405f; font-size: 12px; margin: 4px 6px 0 0;
            }
            .pill-ghost {
                display:inline-block; padding:8px 12px; border-radius: 999px;
                border: 1px solid rgba(255,255,255,.35); color:#fff; font-size: 12px; margin: 4px 6px 0 0;
            }

            /* left rail */
            .rail {
                width: 64px; background:#000000; border-radius: 22px;
                padding: 10px 8px; display:flex; flex-direction:column; gap:8px;
                align-items:center; box-shadow: 0 10px 30px rgba(20, 20, 60, 0.08);
            }
            .rail .ico {
                width:44px; height:44px; border-radius:16px; display:flex; align-items:center; justify-content:center;
                background:#f3f5ff; color:#5c49e0; font-size:20px; cursor:default;
            }
            .rail .ico.active { background:#5c49e0; color:#fff; }

            /* right sidebar list items */
            .friend {
                display:flex; align-items:center; justify-content:space-between;
                padding:10px 12px; border-radius:14px; background:#65426F; margin-bottom:8px;
                box-shadow: 0 6px 16px rgba(20,20,60,0.06);
            }
            .small { font-size:12px; opacity:.75; }
            .heading { font-weight:800; letter-spacing:.2px; }
            .muted { color:#8b92b0; }

            .tiny-map { border-radius: 16px; overflow:hidden; }
            .section { font-size:18px; font-weight:800; color:#404665; margin: 6px 0 10px; }

            .progress-wrap{
                background:#f3f5ff; height:10px; border-radius: 999px; overflow:hidden; width:100%;
            }
            .progress-bar{
                height:10px; border-radius:999px;
                background: linear-gradient(90deg, #6c63ff, #ff7da0);
            }
        </style>
    """, unsafe_allow_html=True)

    # ================== TOP BAR ==================
    top_l, top_r = st.columns([7, 3], gap="medium")
    with top_l:
        st.markdown("<div class='section'>Dashboard</div>", unsafe_allow_html=True)

    with top_r:
        t1, t2 = st.columns([4,1])
        with t1:
            q = st.text_input("Search", placeholder="Search ARGO floats, regions, profiles…")
        with t2:
            st.write("") ; st.write("")
            st.write("🙂")  # avatar placeholder

    st.write("")

        # ---------- Middle (Main) ----------

    # Filters (drive plots)
    fl1, fl2, fl3, fl4 = st.columns([2,2,2,2])
    with fl1:
        date_range = st.date_input("Date range", [])  # user can pick later
    with fl2:
        # Changed from selectbox → multiselect
        param = st.multiselect(
            "Parameter", 
            ["Temperature", "Salinity", "Oxygen", "Chl-a"],
            default=["Temperature"]
        )
    with fl3:
        depth = st.slider("Pressure (m)", 0, 2000, (0, 1000), step=50)
    with fl4:
        region = st.selectbox("Region", ["Indian Ocean", "Pacific Ocean", "Atlantic Ocean"])

    # New latitude-longitude filters (extra row)
    lat_range = st.slider("Latitude range", -90, 90, (-20, 20))
    lon_range = st.slider("Longitude range", -180, 180, (60, 100))

    # ------- Top row: Overview + two stacked cards -------
    top_big, top_s1, top_s2 = st.columns([2.3, 1.2, 1.2])


    # ================== MAIN 3-COLUMN GRID ==================
    rail_col, main_col, right_col = st.columns([1, 7, 3], gap="large")
   

    # ---------- Middle (Main) ----------
    with main_col:
       

        # ------- Top row: Overview + two stacked cards -------
        top_big, top_s1, top_s2 = st.columns([2.3, 1.2, 1.2])

        # Overview gradient card with line chart
        with top_big:
            st.markdown("<div class='card grad-purple'>", unsafe_allow_html=True)
            st.markdown("### Overview")
            st.markdown("<span class='pill-ghost'>Monthly</span>", unsafe_allow_html=True)

            # Fake data demo – replace with real ARGO aggregates
            import pandas as pd, numpy as np
            months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
            y = np.round(np.linspace(7.5, 9.5, 12) + np.random.randn(12)*0.15, 3)  # e.g., mean salinity/temperature index
            df = pd.DataFrame({"month": months, "value": y})
            df.set_index("month", inplace=True)

            st.line_chart(df, height=220, use_container_width=True)

            cK1, cK2, cK3 = st.columns(3)
            with cK1:
                st.markdown("<div class='kpi'>Total Profiles</div>", unsafe_allow_html=True)
                st.markdown("<div class='kpi-value'>9,178</div>", unsafe_allow_html=True)
                st.markdown("<div class='small'>April</div>", unsafe_allow_html=True)
            with cK2:
                st.markdown(f"<div class='kpi'>Parameter</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='kpi-value'>{param[0]}</div>", unsafe_allow_html=True)
                st.markdown("<div class='small'>Current filter</div>", unsafe_allow_html=True)
            with cK3:
                st.markdown("<div class='kpi'>Depth Window</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='kpi-value'>{depth[0]}–{depth[1]} m</div>", unsafe_allow_html=True)
                st.markdown("<div class='small'>Selected</div>", unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)

        # Daily Query (violet)
        with top_s1:
            st.markdown("<div class='card grad-violet card-tight'>", unsafe_allow_html=True)
            st.markdown("#### Daily Query")
            st.markdown("<div class='small'>Today’s executed data requests</div>", unsafe_allow_html=True)
            st.metric("Queries", value="42", delta="+7 vs. yesterday")
            st.markdown("<span class='pill-ghost'>SQL</span><span class='pill-ghost'>RAG</span>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # My Workspace (pink)
        with top_s2:
            st.markdown("<div class='card grad-pink card-tight'>", unsafe_allow_html=True)
            st.markdown("#### My Workspace")
            st.markdown("<div class='small'>Saved views & exports</div>", unsafe_allow_html=True)
            st.metric("Exports", value="12", delta="+3")
            st.markdown("<span class='pill-ghost'>Parquet</span><span class='pill-ghost'>NetCDF</span>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        st.write("")

        # ------- Activity Tiles (3 cards) -------
        a1, a2, a3 = st.columns(3)
        with a1:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("🚴 Salinity Near Equator (Mar 2023)")
            st.markdown("</div>", unsafe_allow_html=True)

        with a2:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("🏃 Compare BGC (Arabian Sea, 6 mo)")
            st.markdown("</div>", unsafe_allow_html=True)

        with a3:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("💪 Nearest ARGO Floats (Point)") 
            st.markdown("</div>", unsafe_allow_html=True)

        st.write("")


    # ---------- Right Panel ----------
    with right_col:

        st.markdown("<div class='section'>Live map</div>", unsafe_allow_html=True)
        # Mini map (Indian Ocean center-ish)
        import pandas as pd
        map_df = pd.DataFrame({"lat":[-5,10,-15,5], "lon":[70,80,60,90]})
        st.map(map_df, height=210)



st.markdown(
    """
    <style>
    /* Input boxes, text areas, and select boxes */
    .stTextInput > div > div > input,
    .stTextArea > div > textarea,
    .stSelectbox > div > div,
    .stNumberInput > div > div > input {
        background-color: rgba(20, 40, 50, 0.9); /* dark grid color */
        color: #e0fdfb;  /* neon aqua text */
        border: 1px solid rgba(124, 249, 214, 0.4);
        border-radius: 10px;
        padding: 8px;
    }

    /* Change focus glow */
    .stTextInput > div > div > input:focus,
    .stTextArea > div > textarea:focus,
    .stSelectbox > div > div:focus,
    .stNumberInput > div > div > input:focus {
        border: 1px solid #7cf9d6;
        box-shadow: 0 0 12px #7cf9d6;
        outline: none;
    }

    /* Buttons */
    .stButton > button {
        background-color: rgba(30, 60, 70, 0.95);
        color: #7cf9d6;
        border: 1px solid #7cf9d6;
        border-radius: 12px;
        padding: 8px 16px;
        transition: 0.3s;
    }
    .stButton > button:hover {
        background-color: #7cf9d6;
        color: #0a1a1f;
        box-shadow: 0 0 15px #7cf9d6;
    }
    </style>
    """,
    unsafe_allow_html=True
)