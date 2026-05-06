import streamlit as st
import os
import pandas as pd

st.set_page_config(
    page_title="AI Declaration Analysis — CEUR & Non-CEUR",
    layout="wide",
    page_icon="📄",
    initial_sidebar_state="expanded",
)

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR    = r"Gen_AI/images"
NON_CEUR    = os.path.join(BASE_DIR, "Non_CEUR_publications")
CEUR        = os.path.join(BASE_DIR, "CEUR_publications")

NON_CEUR_FIGURES = {
    "declaration_status":         os.path.join(NON_CEUR, "fig_declaration_status.png"),
    "top_ai_tools":               os.path.join(NON_CEUR, "fig_top_ai_tools.png"),
    "papers_by_tool_count":       os.path.join(NON_CEUR, "fig_papers_by_tool_count.png"),
    "contribution_roles":         os.path.join(NON_CEUR, "fig_contribution_roles.png"),
    "role_categories_pie":        os.path.join(NON_CEUR, "fig_role_categories_pie.png"),
    "tool_role_heatmap":          os.path.join(NON_CEUR, "fig_tool_role_heatmap.png"),
    "country_ai_count":           os.path.join(NON_CEUR, "fig1a_country_ai_count.PNG"),
    "institution_ai_usage":       os.path.join(NON_CEUR, "fig2_institution_ai_usage.png"),
    "temporal_rate":              os.path.join(NON_CEUR, "fig3b_temporal_rate.png"),
    "tools_by_country":           os.path.join(NON_CEUR, "fig5_top10_tools_by_country_stacked.png"),
    "institution_tool_heatmap":   os.path.join(NON_CEUR, "fig6a_institution_tool_heatmap.png"),
    "institution_role_heatmap":   os.path.join(NON_CEUR, "fig6b_institution_role_heatmap.png"),
}

NON_CEUR_TABLES = {
    "ai_usage_summary":           os.path.join(NON_CEUR, "ai_usage_summary.csv"),
    "contribution_roles":         os.path.join(NON_CEUR, "contribution_roles_table.csv"),
    "countries_detailed":         os.path.join(NON_CEUR, "countries_detailed.csv"),
    "institution_ai_statistics":  os.path.join(NON_CEUR, "institution_ai_statistics.csv"),
}

CEUR_FIGURES = {
    "monthly_ai_trend":           os.path.join(CEUR, "fig_monthly_ai_trend_CEUR_event_and_paper_published_date-1.png"),
    "tools_evolution":            os.path.join(CEUR, "fig_tools_evolution_stacked_bars-1.png"),
    "country_ai_rate":            os.path.join(CEUR, "fig1b_country_ai_rate-1.png"),
    "institution_ai_usage":       os.path.join(CEUR, "fig2_institution_ai_usage-1.png"),
    "temporal_ai_usage_rate":     os.path.join(CEUR, "fig3b_temporal_AI_usage_rate-1.png"),
}

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main > div { padding-left: 1rem; padding-right: 1rem; max-width: none; }
    .block-container { padding-top: 1rem; padding-bottom: 0rem;
                       padding-left: 1.5rem; padding-right: 1.5rem; max-width: none; }

    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2.5rem 2rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 6px 20px rgba(0,0,0,0.2);
    }
    .main-header h1  { margin: 0; font-size: 2.2rem; letter-spacing: -0.5px; }
    .main-header .subtitle { margin: 0.6rem 0 0; opacity: 0.85; font-size: 1.05rem; }
    .main-header .note { margin: 0.4rem 0 0; opacity: 0.7; font-size: 0.9rem; }

    .ceur-header {
        background: linear-gradient(135deg, #1b2838 0%, #2a475e 50%, #1b4965 100%);
        padding: 2.5rem 2rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 6px 20px rgba(0,0,0,0.2);
    }
    .ceur-header h1  { margin: 0; font-size: 2.2rem; letter-spacing: -0.5px; }
    .ceur-header .subtitle { margin: 0.6rem 0 0; opacity: 0.85; font-size: 1.05rem; }
    .ceur-header .note { margin: 0.4rem 0 0; opacity: 0.7; font-size: 0.9rem; }

    .section-header {
        background: #f8f9fa;
        padding: 1rem 1.2rem;
        border-radius: 8px;
        margin: 1.5rem 0 1rem;
        border-left: 5px solid #0f3460;
    }
    .section-header h2 { margin: 0; color: #0f3460; font-size: 1.4rem; }
    .section-header p  { margin: 0.3rem 0 0; color: #666; font-size: 0.9rem; }

    .section-header-ceur {
        background: #f0f8ff;
        padding: 1rem 1.2rem;
        border-radius: 8px;
        margin: 1.5rem 0 1rem;
        border-left: 5px solid #1b4965;
    }
    .section-header-ceur h2 { margin: 0; color: #1b4965; font-size: 1.4rem; }
    .section-header-ceur p  { margin: 0.3rem 0 0; color: #666; font-size: 0.9rem; }

    .metric-card {
        background: white;
        border: 1px solid #e8e8e8;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        height: 100%;
    }
    .metric-card .val  { font-size: 2rem; font-weight: 700; color: #0f3460; }
    .metric-card .lbl  { font-size: 0.82rem; color: #666; margin-top: 0.2rem; }

    .stemlist-box {
        background: #f0f4ff;
        border: 1px solid #c5d3f0;
        border-radius: 10px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1.5rem;
    }
    .stemlist-box h3 { margin: 0 0 0.8rem; color: #0f3460; }
    .stemlist-box ul { margin: 0; padding-left: 1.4rem; }
    .stemlist-box li { margin-bottom: 0.4rem; color: #333; font-size: 0.95rem; }

    .stemlist-box-ceur {
        background: #eaf4fb;
        border: 1px solid #a8cfe0;
        border-radius: 10px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1.5rem;
    }
    .stemlist-box-ceur h3 { margin: 0 0 0.8rem; color: #1b4965; }
    .stemlist-box-ceur ul { margin: 0; padding-left: 1.4rem; }
    .stemlist-box-ceur li { margin-bottom: 0.4rem; color: #333; font-size: 0.95rem; }

    [data-testid="metric-container"] {
        background: #f8f9fa; border: 1px solid #e0e0e0;
        padding: 1rem; border-radius: 8px;
    }
    #MainMenu { visibility: hidden; }
    footer     { visibility: hidden; }
    iframe { width: 100% !important; border: none; }
</style>
""", unsafe_allow_html=True)

# ─── Helpers ──────────────────────────────────────────────────────────────────

def show_figure(path: str):
    if not os.path.exists(path):
        st.error(f"Figure not found: `{path}`")
        return
    st.image(path, use_container_width=True)


def load_csv(path: str) -> pd.DataFrame | None:
    if not os.path.exists(path):
        st.error(f"Table not found: `{path}`")
        return None
    return pd.read_csv(path)


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📂 Dataset")
    dataset = st.radio(
        "Choose dataset",
        ["Non-CEUR Publications", "CEUR Publications"],
        label_visibility="collapsed",
    )
    st.markdown("---")

    if dataset == "Non-CEUR Publications":
        st.markdown("### 🗂️ Sections")
        page = st.radio(
            "Select a page",
            [
                "1 · Overview & Key Metrics",
                "2 · Declaration Status",
                "3 · Top AI Tools",
                "4 · Papers by Tool Count",
                "5 · Contribution Roles",
                "6 · Role Categories",
                "7 · Tool–Role Heatmap",
                "8 · Country Analysis",
                "9 · Institution Analysis",
                "10 · Temporal Trends",
                "11 · Tools by Country",
                "12 · Institution Heatmaps",
                "13 · Data Tables",
            ],
            label_visibility="collapsed",
        )
    else:
        st.markdown("### 🗂️ Sections")
        page = st.radio(
            "Select a page",
            [
                "C1 · Monthly AI Trend",
                "C2 · Tools Evolution",
                "C3 · Country AI Rate",
                "C4 · Institution AI Usage",
                "C5 · Temporal AI Usage Rate",
            ],
            label_visibility="collapsed",
        )

    st.markdown("---")
    st.markdown(
        "<small style='color:#888;'>TPDL AI Declaration Study<br/>Powered by Streamlit</small>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# NON-CEUR PUBLICATIONS
# ══════════════════════════════════════════════════════════════════════════════
if dataset == "Non-CEUR Publications":

    st.markdown("""
    <div class="main-header">
        <h1>📄 Non-CEUR Publication AI Usage Analysis</h1>
        <div class="subtitle">Exploratory Data Analysis of AI Tool Declarations in Non-CEUR Proceedings</div>
        <div class="note">
            Analysis covers 15,409 papers · AI declarations, tool usage, contribution roles, geography &amp; temporal trends
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="stemlist-box">
        <h3>📋 Analysis Sections — Quick Reference</h3>
        <ul>
            <li><strong>1 · Overview &amp; Key Metrics</strong> — dataset-level summary statistics</li>
            <li><strong>2 · Declaration Status</strong> — share of papers declaring AI use vs. no-declaration</li>
            <li><strong>3 · Top AI Tools</strong> — most frequently mentioned AI tools across papers</li>
            <li><strong>4 · Papers by Tool Count</strong> — distribution of papers by number of tools used</li>
            <li><strong>5 · Contribution Roles</strong> — how authors used AI (writing, coding, translation …)</li>
            <li><strong>6 · Role Categories</strong> — pie-chart breakdown of high-level role families</li>
            <li><strong>7 · Tool–Role Heatmap</strong> — co-occurrence matrix of tools and contribution roles</li>
            <li><strong>8 · Country Analysis</strong> — AI usage counts per country</li>
            <li><strong>9 · Institution Analysis</strong> — AI usage rates by institution</li>
            <li><strong>10 · Temporal Trends</strong> — AI adoption rate over time</li>
            <li><strong>11 · Tools by Country</strong> — top-10 tools stacked by country</li>
            <li><strong>12 · Institution Heatmaps</strong> — co-occurrence heatmaps of institutions vs. tools and vs. roles</li>
            <li><strong>13 · Data Tables</strong> — raw CSV tables for summary metrics, roles, countries &amp; institutions</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # ── Page 1 — Overview & Key Metrics ──────────────────────────────────────
    if page == "1 · Overview & Key Metrics":
        st.markdown("""
        <div class="section-header">
            <h2>📊 Overview &amp; Key Metrics</h2>
            <p>Dataset-level summary of AI tool usage declarations in Non-CEUR proceedings.</p>
        </div>""", unsafe_allow_html=True)

        df_sum = load_csv(NON_CEUR_TABLES["ai_usage_summary"])

        if df_sum is not None:
            metrics = dict(zip(df_sum["Metric"], df_sum["Value"]))

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown(f"""<div class="metric-card">
                    <div class="val">{metrics.get('Total Papers','—')}</div>
                    <div class="lbl">Total Papers</div></div>""", unsafe_allow_html=True)
            with c2:
                st.markdown(f"""<div class="metric-card">
                    <div class="val">{metrics.get('Papers Declaring AI Use','—')}</div>
                    <div class="lbl">Declare AI Use</div></div>""", unsafe_allow_html=True)
            with c3:
                st.markdown(f"""<div class="metric-card">
                    <div class="val">{metrics.get('Papers Declaring No AI Use','—')}</div>
                    <div class="lbl">Declare No AI Use</div></div>""", unsafe_allow_html=True)
            with c4:
                st.markdown(f"""<div class="metric-card">
                    <div class="val">{metrics.get('Papers with No Declaration Found','—')}</div>
                    <div class="lbl">No Declaration Found</div></div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            c5, c6, c7, c8 = st.columns(4)
            with c5:
                st.markdown(f"""<div class="metric-card">
                    <div class="val">{metrics.get('AI Adoption Rate (%)','—')}%</div>
                    <div class="lbl">AI Adoption Rate</div></div>""", unsafe_allow_html=True)
            with c6:
                st.markdown(f"""<div class="metric-card">
                    <div class="val">{metrics.get('Unique AI Tools','—')}</div>
                    <div class="lbl">Unique AI Tools</div></div>""", unsafe_allow_html=True)
            with c7:
                st.markdown(f"""<div class="metric-card">
                    <div class="val">{metrics.get('Papers Using Multiple Tools','—')}</div>
                    <div class="lbl">Papers w/ Multiple Tools</div></div>""", unsafe_allow_html=True)
            with c8:
                st.markdown(f"""<div class="metric-card">
                    <div class="val">{metrics.get('Multi-tool Rate (%)','—')}%</div>
                    <div class="lbl">Multi-tool Rate</div></div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            c9, c10, c11, c12 = st.columns(4)
            with c9:
                st.markdown(f"""<div class="metric-card">
                    <div class="val">{metrics.get('Average Distinct Tools per Paper (AI users)','—')}</div>
                    <div class="lbl">Avg Tools / Paper</div></div>""", unsafe_allow_html=True)
            with c10:
                st.markdown(f"""<div class="metric-card">
                    <div class="val">{metrics.get('Maximum Tools per Paper','—')}</div>
                    <div class="lbl">Max Tools in One Paper</div></div>""", unsafe_allow_html=True)
            with c11:
                st.markdown(f"""<div class="metric-card">
                    <div class="val" style="font-size:1.3rem;">{metrics.get('Most Used Tool','—')}</div>
                    <div class="lbl">Most Used Tool</div></div>""", unsafe_allow_html=True)
            with c12:
                st.markdown(f"""<div class="metric-card">
                    <div class="val" style="font-size:1rem;">{metrics.get('Most Common Contribution Role','—')}</div>
                    <div class="lbl">Top Contribution Role</div></div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("#### Full Summary Table")
            st.dataframe(df_sum, use_container_width=True, hide_index=True)

    # ── Page 2 — Declaration Status ───────────────────────────────────────────
    elif page == "2 · Declaration Status":
        st.markdown("""
        <div class="section-header">
            <h2>📋 AI Declaration Status</h2>
            <p>Distribution of papers across three declaration categories:
            <em>Declares AI Use</em>, <em>Declares No AI Use</em>, and <em>No Declaration Found</em>.</p>
        </div>""", unsafe_allow_html=True)

        show_figure(NON_CEUR_FIGURES["declaration_status"])
        st.info(
            "**Interpretation:** Only ~1.3 % of Non-CEUR papers explicitly declare AI tool usage. "
            "A large majority (>98 %) contain no declaration at all, highlighting the nascent state "
            "of AI transparency in these venues."
        )

    # ── Page 3 — Top AI Tools ─────────────────────────────────────────────────
    elif page == "3 · Top AI Tools":
        st.markdown("""
        <div class="section-header">
            <h2>🤖 Top AI Tools</h2>
            <p>Most frequently mentioned AI tools across all papers that declared AI usage.</p>
        </div>""", unsafe_allow_html=True)

        show_figure(NON_CEUR_FIGURES["top_ai_tools"])
        st.info(
            "**Key finding:** ChatGPT dominates as the most cited AI tool, "
            "followed by Grammarly and other LLM-based assistants. "
            "75 unique tools were identified across 203 papers."
        )

    # ── Page 4 — Papers by Tool Count ────────────────────────────────────────
    elif page == "4 · Papers by Tool Count":
        st.markdown("""
        <div class="section-header">
            <h2>📦 Papers by Number of AI Tools Used</h2>
            <p>How many papers used one tool, two tools, three tools, etc.</p>
        </div>""", unsafe_allow_html=True)

        show_figure(NON_CEUR_FIGURES["papers_by_tool_count"])
        st.info(
            "**Key finding:** Most AI-declaring papers used only a single tool. "
            "About 40 % used multiple tools (max 6 tools in a single paper). "
            "The average is 1.54 distinct tools per paper."
        )

    # ── Page 5 — Contribution Roles ───────────────────────────────────────────
    elif page == "5 · Contribution Roles":
        st.markdown("""
        <div class="section-header">
            <h2>✍️ AI Contribution Roles</h2>
            <p>What tasks did authors use AI for? Ranked by frequency of mention.</p>
        </div>""", unsafe_allow_html=True)

        show_figure(NON_CEUR_FIGURES["contribution_roles"])

        st.markdown("#### Detailed Contribution Roles Table")
        df_roles = load_csv(NON_CEUR_TABLES["contribution_roles"])
        if df_roles is not None:
            st.dataframe(
                df_roles,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "% of Mentions":  st.column_config.ProgressColumn("% of Mentions",  min_value=0, max_value=100),
                    "Cumulative (%)": st.column_config.ProgressColumn("Cumulative (%)", min_value=0, max_value=100),
                },
            )
        st.info(
            "**Key finding:** Language-enhancement tasks dominate — "
            "'Improve writing style' (35.8 %) and 'Grammar and spelling check' (27.9 %) "
            "together account for nearly two-thirds of all AI role mentions."
        )

    # ── Page 6 — Role Categories ──────────────────────────────────────────────
    elif page == "6 · Role Categories":
        st.markdown("""
        <div class="section-header">
            <h2>🥧 Role Category Distribution</h2>
            <p>High-level grouping of contribution roles into functional categories
            (Language Enhancement, Technical, Content Generation, etc.).</p>
        </div>""", unsafe_allow_html=True)

        show_figure(NON_CEUR_FIGURES["role_categories_pie"])
        st.info(
            "**Key finding:** 'Language Enhancement' is the dominant category, "
            "comprising writing style improvement, grammar checking, and paraphrasing tasks. "
            "Technical and content-generation categories follow at a distance."
        )

    # ── Page 7 — Tool–Role Heatmap ────────────────────────────────────────────
    elif page == "7 · Tool–Role Heatmap":
        st.markdown("""
        <div class="section-header">
            <h2>🗺️ Tool–Role Co-occurrence Heatmap</h2>
            <p>Which AI tools were used for which contribution roles?
            Darker cells indicate higher co-occurrence frequency.</p>
        </div>""", unsafe_allow_html=True)

        show_figure(NON_CEUR_FIGURES["tool_role_heatmap"])
        st.info(
            "**Key finding:** ChatGPT appears across nearly all role types, "
            "while specialised tools (e.g. Grammarly, DeepL) are concentrated "
            "in narrow role columns such as grammar-checking and translation."
        )

    # ── Page 8 — Country Analysis ─────────────────────────────────────────────
    elif page == "8 · Country Analysis":
        st.markdown("""
        <div class="section-header">
            <h2>🌍 Country-Level AI Usage</h2>
            <p>Number of papers declaring AI tool usage per country of affiliation.</p>
        </div>""", unsafe_allow_html=True)

        show_figure(NON_CEUR_FIGURES["country_ai_count"])

        st.markdown("#### Countries — Detailed Table")
        df_countries = load_csv(NON_CEUR_TABLES["countries_detailed"])
        if df_countries is not None:
            q = st.text_input("Filter by country", placeholder="e.g. Germany, China …")
            if q:
                mask = df_countries.apply(lambda col: col.astype(str).str.contains(q, case=False)).any(axis=1)
                df_countries = df_countries[mask]
            st.dataframe(df_countries, use_container_width=True, hide_index=True)
            st.download_button(
                "⬇️ Download CSV",
                data=df_countries.to_csv(index=False).encode("utf-8"),
                file_name="countries_detailed.csv",
                mime="text/csv",
            )

    # ── Page 9 — Institution Analysis ─────────────────────────────────────────
    elif page == "9 · Institution Analysis":
        st.markdown("""
        <div class="section-header">
            <h2>🏛️ Institution-Level AI Usage</h2>
            <p>AI adoption rates and paper counts broken down by affiliated institution.</p>
        </div>""", unsafe_allow_html=True)

        show_figure(NON_CEUR_FIGURES["institution_ai_usage"])

        st.markdown("#### Institution AI Statistics Table")
        df_inst = load_csv(NON_CEUR_TABLES["institution_ai_statistics"])
        if df_inst is not None:
            q = st.text_input("Filter by institution", placeholder="e.g. MIT, TU Berlin …")
            if q:
                mask = df_inst.apply(lambda col: col.astype(str).str.contains(q, case=False)).any(axis=1)
                df_inst = df_inst[mask]
            st.dataframe(df_inst, use_container_width=True, hide_index=True)
            st.download_button(
                "⬇️ Download CSV",
                data=df_inst.to_csv(index=False).encode("utf-8"),
                file_name="institution_ai_statistics.csv",
                mime="text/csv",
            )

    # ── Page 10 — Temporal Trends ─────────────────────────────────────────────
    elif page == "10 · Temporal Trends":
        st.markdown("""
        <div class="section-header">
            <h2>📈 Temporal AI Adoption Trend</h2>
            <p>How AI declaration rates in Non-CEUR papers evolved over time.</p>
        </div>""", unsafe_allow_html=True)

        show_figure(NON_CEUR_FIGURES["temporal_rate"])
        st.info(
            "**Key finding:** AI declarations show a clear upward trend, "
            "reflecting the broader acceleration of LLM adoption in research workflows "
            "following the public release of ChatGPT in late 2022."
        )

    # ── Page 11 — Tools by Country ────────────────────────────────────────────
    elif page == "11 · Tools by Country":
        st.markdown("""
        <div class="section-header">
            <h2>🌐 Top-10 AI Tools by Country (Stacked)</h2>
            <p>Country-level breakdown of which AI tools are most commonly used,
            shown as a stacked bar chart for the top countries.</p>
        </div>""", unsafe_allow_html=True)

        show_figure(NON_CEUR_FIGURES["tools_by_country"])
        st.info(
            "**Key finding:** ChatGPT is the dominant tool across all top countries. "
            "Grammarly usage is especially prominent in English-speaking regions, "
            "while DeepL appears more frequently in European countries."
        )

    # ── Page 12 — Institution Heatmaps ───────────────────────────────────────
    elif page == "12 · Institution Heatmaps":
        st.markdown("""
        <div class="section-header">
            <h2>🗺️ Institution Co-occurrence Heatmaps</h2>
            <p>Two complementary heatmaps showing which institutions used which AI tools
            and which contribution roles, based on papers that declared AI usage.</p>
        </div>""", unsafe_allow_html=True)

        st.markdown("#### Institution × Tool Co-occurrence")
        show_figure(NON_CEUR_FIGURES["institution_tool_heatmap"])
        st.info(
            "**Key finding:** Darker cells indicate an institution's papers frequently used a given tool. "
            "ChatGPT dominates across most institutions, while niche tools cluster in specific research groups."
        )

        st.markdown("---")
        st.markdown("#### Institution × Contribution Role Co-occurrence")
        show_figure(NON_CEUR_FIGURES["institution_role_heatmap"])
        st.info(
            "**Key finding:** Language-enhancement roles (writing style, grammar) are broadly distributed "
            "across institutions, whereas technical roles (code generation, data analysis) are more "
            "concentrated in a subset of research-oriented institutions."
        )

    # ── Page 13 — Data Tables ─────────────────────────────────────────────────
    elif page == "13 · Data Tables":
        st.markdown("""
        <div class="section-header">
            <h2>📑 Data Tables</h2>
            <p>Raw tabular data underpinning the analysis — filterable and sortable.</p>
        </div>""", unsafe_allow_html=True)

        tab_sum, tab_roles, tab_countries, tab_inst = st.tabs([
            "AI Usage Summary", "Contribution Roles", "Countries", "Institutions"
        ])

        with tab_sum:
            st.markdown("##### AI Usage Summary Metrics")
            df_sum = load_csv(NON_CEUR_TABLES["ai_usage_summary"])
            if df_sum is not None:
                st.dataframe(df_sum, use_container_width=True, hide_index=True)
                st.download_button(
                    "⬇️ Download CSV",
                    data=df_sum.to_csv(index=False).encode("utf-8"),
                    file_name="ai_usage_summary.csv",
                    mime="text/csv",
                )

        with tab_roles:
            st.markdown("##### Contribution Roles — Full Table")
            df_roles = load_csv(NON_CEUR_TABLES["contribution_roles"])
            if df_roles is not None:
                q = st.text_input("Filter by role or category", placeholder="e.g. writing, translation…")
                if q:
                    mask = df_roles.apply(lambda col: col.astype(str).str.contains(q, case=False)).any(axis=1)
                    df_roles = df_roles[mask]
                st.dataframe(
                    df_roles,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "% of Mentions":  st.column_config.ProgressColumn("% of Mentions",  min_value=0, max_value=100),
                        "Cumulative (%)": st.column_config.ProgressColumn("Cumulative (%)", min_value=0, max_value=100),
                        "Top 5 Tools":    st.column_config.TextColumn("Top 5 Tools", width="large"),
                    },
                )
                st.download_button(
                    "⬇️ Download CSV",
                    data=df_roles.to_csv(index=False).encode("utf-8"),
                    file_name="contribution_roles_table.csv",
                    mime="text/csv",
                )

        with tab_countries:
            st.markdown("##### Countries — Detailed Table")
            df_countries = load_csv(NON_CEUR_TABLES["countries_detailed"])
            if df_countries is not None:
                st.dataframe(df_countries, use_container_width=True, hide_index=True)
                st.download_button(
                    "⬇️ Download CSV",
                    data=df_countries.to_csv(index=False).encode("utf-8"),
                    file_name="countries_detailed.csv",
                    mime="text/csv",
                )

        with tab_inst:
            st.markdown("##### Institution AI Statistics")
            df_inst = load_csv(NON_CEUR_TABLES["institution_ai_statistics"])
            if df_inst is not None:
                st.dataframe(df_inst, use_container_width=True, hide_index=True)
                st.download_button(
                    "⬇️ Download CSV",
                    data=df_inst.to_csv(index=False).encode("utf-8"),
                    file_name="institution_ai_statistics.csv",
                    mime="text/csv",
                )


# ══════════════════════════════════════════════════════════════════════════════
# CEUR PUBLICATIONS
# ══════════════════════════════════════════════════════════════════════════════
else:

    st.markdown("""
    <div class="ceur-header">
        <h1>📚 CEUR Publication AI Usage Analysis</h1>
        <div class="subtitle">Exploratory Data Analysis of AI Tool Declarations in CEUR-WS Proceedings</div>
        <div class="note">
            Monthly trends, tool evolution, country rates, institution usage &amp; temporal patterns
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="stemlist-box-ceur">
        <h3>📋 Analysis Sections — Quick Reference</h3>
        <ul>
            <li><strong>C1 · Monthly AI Trend</strong> — AI declarations over months, by event and paper published date</li>
            <li><strong>C2 · Tools Evolution</strong> — how individual AI tools' usage share evolved over time (stacked bars)</li>
            <li><strong>C3 · Country AI Rate</strong> — AI adoption rates per country of affiliation</li>
            <li><strong>C4 · Institution AI Usage</strong> — AI usage breakdown by institution</li>
            <li><strong>C5 · Temporal AI Usage Rate</strong> — overall AI adoption rate trend over time</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # ── C1 — Monthly AI Trend ─────────────────────────────────────────────────
    if page == "C1 · Monthly AI Trend":
        st.markdown("""
        <div class="section-header-ceur">
            <h2>📅 Monthly AI Trend (Event &amp; Paper Published Date)</h2>
            <p>Number of CEUR papers declaring AI usage plotted by both event date and
            paper publication date, showing how declarations grew month-by-month.</p>
        </div>""", unsafe_allow_html=True)

        show_figure(CEUR_FIGURES["monthly_ai_trend"])
        st.info(
            "**Key finding:** AI declarations in CEUR proceedings accelerated sharply from early 2023 onwards, "
            "mirroring the broader adoption of ChatGPT and similar tools in academic writing workflows."
        )

    # ── C2 — Tools Evolution ──────────────────────────────────────────────────
    elif page == "C2 · Tools Evolution":
        st.markdown("""
        <div class="section-header-ceur">
            <h2>📊 AI Tools Evolution (Stacked Bars)</h2>
            <p>How the share of individual AI tools in CEUR papers shifted over successive time periods.</p>
        </div>""", unsafe_allow_html=True)

        show_figure(CEUR_FIGURES["tools_evolution"])
        st.info(
            "**Key finding:** ChatGPT's share rose steeply in 2023 and remained dominant, "
            "while newer tools gradually captured a growing minority share in 2024."
        )

    # ── C3 — Country AI Rate ──────────────────────────────────────────────────
    elif page == "C3 · Country AI Rate":
        st.markdown("""
        <div class="section-header-ceur">
            <h2>🌍 Country AI Declaration Rate</h2>
            <p>Percentage of CEUR papers from each country that included an AI usage declaration.</p>
        </div>""", unsafe_allow_html=True)

        show_figure(CEUR_FIGURES["country_ai_rate"])
        st.info(
            "**Key finding:** AI declaration rates vary considerably by country, "
            "reflecting differences in institutional policies, venue requirements, and awareness of reporting norms."
        )

    # ── C4 — Institution AI Usage ─────────────────────────────────────────────
    elif page == "C4 · Institution AI Usage":
        st.markdown("""
        <div class="section-header-ceur">
            <h2>🏛️ Institution-Level AI Usage (CEUR)</h2>
            <p>AI declaration counts and rates broken down by affiliated institution in CEUR proceedings.</p>
        </div>""", unsafe_allow_html=True)

        show_figure(CEUR_FIGURES["institution_ai_usage"])
        st.info(
            "**Key finding:** A small number of high-output institutions account for a disproportionate "
            "share of AI-declaring papers, suggesting early adoption is clustered in specific research groups."
        )

    # ── C5 — Temporal AI Usage Rate ───────────────────────────────────────────
    elif page == "C5 · Temporal AI Usage Rate":
        st.markdown("""
        <div class="section-header-ceur">
            <h2>📈 Temporal AI Usage Rate (CEUR)</h2>
            <p>Overall AI declaration rate in CEUR papers as a percentage, plotted over time.</p>
        </div>""", unsafe_allow_html=True)

        show_figure(CEUR_FIGURES["temporal_ai_usage_rate"])
        st.info(
            "**Key finding:** The AI declaration rate in CEUR proceedings climbed from near zero "
            "in 2022 to several percent by 2024, indicating a rapid normalisation of AI disclosure practices."
        )


# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center;padding:1rem 0;color:#888;'>
    <strong>AI Declaration Analysis Dashboard — CEUR &amp; Non-CEUR Publications</strong><br>
    <small>TPDL AI Declaration Study · Powered by Streamlit</small>
</div>
""", unsafe_allow_html=True)
