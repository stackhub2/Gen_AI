
import matplotlib.pyplot as plt
print(plt.style.available)

# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.cluster import hierarchy
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import networkx as nx
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-bright')
# plt.style.use('seaborn')
sns.set_palette('husl')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

print("Libraries loaded successfully!")
print(f"Analysis timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


fpath=(r"path/to/your/file.csv")


df_tools_filtered=pd.read_csv(fpath)
df_tools_filtered
print(df_tools_filtered)


df_tools_filtered.columns


df_tools_filtered=df_tools_filtered[["index_id",'paper_id', 'AI-tool', 'published', 'Tool-usage','event_date']]
df_tools_filtered


df_tools_filtered.columns=['index_id','paper_id', 'tool', 'published', 'contribution_roles','event_date']
df_tools_filtered


import pandas as pd
import os

# ── Reset index & rename columns to correct convention ───────────────────────
df = df_tools_filtered.copy()

# Rename to clear, correct convention
df = df.rename(columns={
    'index_id':    'paper_id',      # actual paper identifier
    'paper_id':    'volume_id',     # CEUR-WS volume (Vol-XXXX)
    'AI-tool':     'tool',          # AI tool used
    'Tool-usage':  'contribution_role',  # how the tool was used
})
df


df.columns


output_folder = r"path/to/output/folder"


# Published year


import pandas as pd
import os

# ── Derive ai_used from tool column ──────────────────────────────────────────
df['ai_used'] = df['tool'] != 'No AI tool'

# ── Parse BOTH dates ──────────────────────────────────────────────────────────
# event_date format: "November-2025"
df['event_date_parsed']    = pd.to_datetime(df['event_date'], format='%B-%Y', errors='coerce')
df['event_year_month']     = df['event_date_parsed'].dt.to_period('M')

# published format: "2024-05-16"
df['published_parsed']     = pd.to_datetime(df['published'], format='%Y-%m-%d', errors='coerce')
df['published_year_month'] = df['published_parsed'].dt.to_period('M')

# ── Sanity checks ─────────────────────────────────────────────────────────────
print(f"Rows with missing event date:     {df['event_date_parsed'].isna().sum()}")
print(f"Rows with missing published date: {df['published_parsed'].isna().sum()}")
print(f"\nai_used distribution:\n{df['ai_used'].value_counts()}")

# ══════════════════════════════════════════════════════════════════════════════
# HELPER: build monthly stats table from any time column
# ══════════════════════════════════════════════════════════════════════════════
def build_monthly_table(df_source, time_col):
    # Deduplicate to one row per actual paper
    df_papers = (
        df_source.groupby(['paper_id', 'volume_id', time_col], as_index=False)['ai_used']
        .max()
    )

    print(f"\nUnique VOLUMES  (volume_id): {df_papers['volume_id'].nunique()}")
    print(f"Unique PAPERS    (paper_id): {df_papers['paper_id'].nunique()}")

    # Monthly totals
    monthly_total = (
        df_papers.groupby(time_col)['paper_id']
        .nunique().reset_index()
    )
    monthly_total.columns = ['year_month', 'total_papers']

    # Monthly AI users
    monthly_ai = (
        df_papers[df_papers['ai_used'] == True]
        .groupby(time_col)['paper_id']
        .nunique().reset_index()
    )
    monthly_ai.columns = ['year_month', 'papers_using_ai']

    # Merge & compute
    tbl = monthly_total.merge(monthly_ai, on='year_month', how='left')
    tbl['papers_using_ai'] = tbl['papers_using_ai'].fillna(0).astype(int)
    tbl['papers_no_ai']    = tbl['total_papers'] - tbl['papers_using_ai']
    tbl['ai_rate']         = (tbl['papers_using_ai'] / tbl['total_papers'] * 100).round(2)
    tbl = tbl.sort_values('year_month')

    tbl_display = tbl.rename(columns={
        'year_month':       'Month',
        'total_papers':     'Total Papers',
        'papers_using_ai':  'Papers Using AI',
        'papers_no_ai':     'No AI',
        'ai_rate':          'AI Rate (%)',
    })

    return df_papers, tbl_display


# ══════════════════════════════════════════════════════════════════════════════
# HELPER: print summary from deduplicated df_papers
# ══════════════════════════════════════════════════════════════════════════════
def print_summary(df_papers, tbl_display, label):
    total_volumes = df_papers['volume_id'].nunique()
    total_papers  = df_papers['paper_id'].nunique()
    total_ai      = df_papers[df_papers['ai_used'] == True]['paper_id'].nunique()
    total_no_ai   = df_papers[df_papers['ai_used'] == False]['paper_id'].nunique()
    max_idx       = tbl_display['AI Rate (%)'].idxmax()
    min_idx       = tbl_display['AI Rate (%)'].idxmin()

    print(f"\n{'=' * 80}")
    print(f"SUMMARY STATISTICS ({label}):")
    print(f"{'=' * 80}")
    print(f"Total volumes  (volume_id):  {total_volumes}")
    print(f"Total papers    (paper_id):  {total_papers}")
    print(f"  ├─ Using AI:               {total_ai}")
    print(f"  └─ No AI:                  {total_no_ai}")
    print(f"Overall AI adoption rate:    {total_ai / total_papers * 100:.2f}%")
    print(f"Total months covered:        {len(tbl_display)}")
    print(f"Average monthly AI rate:     {tbl_display['AI Rate (%)'].mean():.2f}%")
    print(f"Highest AI adoption rate:    {tbl_display.loc[max_idx, 'AI Rate (%)']:.2f}% in {tbl_display.loc[max_idx, 'Month']}")
    print(f"Lowest AI adoption rate:     {tbl_display.loc[min_idx, 'AI Rate (%)']:.2f}% in {tbl_display.loc[min_idx, 'Month']}")


# ══════════════════════════════════════════════════════════════════════════════
# HELPER: export CSV + LaTeX
# ══════════════════════════════════════════════════════════════════════════════
def df_to_latex(df, caption, label, col_format):
    lines = [
        r"\begin{table}[ht]", r"\centering",
        f"\\caption{{{caption}}}", f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{col_format}}}", r"\hline",
        " & ".join(str(c) for c in df.columns) + r" \\", r"\hline",
    ]
    for _, row in df.iterrows():
        lines.append(" & ".join(
            f"{v:.2f}" if isinstance(v, float) else str(v) for v in row
        ) + r" \\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)

def export_results(tbl_display, csv_name, tex_name, caption, label):
    os.makedirs(output_folder, exist_ok=True)

    # CSV
    tbl_display.to_csv(os.path.join(output_folder, csv_name), index=False)
    print(f"\n✓ CSV exported to '{output_folder}/{csv_name}'")

    # LaTeX
    latex = df_to_latex(tbl_display, caption=caption, label=label, col_format='lrrrr')
    print(f"\n{'=' * 80}\nLATEX TABLE CODE:\n{'=' * 80}")
    print(latex)
    with open(os.path.join(output_folder, tex_name), 'w') as f:
        f.write(latex)
    print(f"\n✓ LaTeX exported to '{output_folder}/{tex_name}'")


# ══════════════════════════════════════════════════════════════════════════════
# 1) BY EVENT DATE
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "█" * 80)
print("ANALYSIS 1: BY EVENT DATE")
print("█" * 80)

df_papers_event, tbl_event = build_monthly_table(df, 'event_year_month')

print("\nMONTHLY AI USAGE STATISTICS (by event date):")
print("=" * 80)
print(tbl_event.to_string(index=False))

export_results(
    tbl_event,
    csv_name='monthly_ai_usage_by_event_date.csv',
    tex_name='monthly_ai_usage_by_event_date.tex',
    caption='Monthly AI usage statistics by event date',
    label='tab:ai_usage_event'
)
print_summary(df_papers_event, tbl_event, label='BY EVENT DATE')


# ══════════════════════════════════════════════════════════════════════════════
# 2) BY PUBLISHED DATE
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "█" * 80)
print("ANALYSIS 2: BY PUBLISHED DATE")
print("█" * 80)

df_papers_pub, tbl_pub = build_monthly_table(df, 'published_year_month')

print("\nMONTHLY AI USAGE STATISTICS (by published date):")
print("=" * 80)
print(tbl_pub.to_string(index=False))

export_results(
    tbl_pub,
    csv_name='monthly_ai_usage_by_published_date.csv',
    tex_name='monthly_ai_usage_by_published_date.tex',
    caption='Monthly AI usage statistics by published date',
    label='tab:ai_usage_published'
)
print_summary(df_papers_pub, tbl_pub, label='BY PUBLISHED DATE')


df


df.columns


print(df)


papers_per_year = (
    df.drop_duplicates(subset='paper_id')['event_date_parsed']
    .dt.year
    .value_counts()
    .sort_index()
    .rename_axis('year')
    .reset_index(name='paper_count')
)

print(papers_per_year)


# Deduplicate by paper_id
unique_papers = df.drop_duplicates(subset='paper_id')

# Total papers per year
papers_per_year = (
    unique_papers['event_date_parsed']
    .dt.year
    .value_counts()
    .sort_index()
)

# AI used vs not, per year
ai_by_year = (
    unique_papers.groupby(unique_papers['event_date_parsed'].dt.year)['ai_used']
    .value_counts()
    .unstack(fill_value=0)
    .rename(columns={True: 'ai_used', False: 'not_used'})
)

ai_by_year['total'] = ai_by_year['ai_used'] + ai_by_year['not_used']
ai_by_year['ai_rate_%'] = (ai_by_year['ai_used'] / ai_by_year['total'] * 100).round(1)

print(ai_by_year)

for year in ai_by_year.index:
    row = ai_by_year.loc[year]
    print(
        f"For {year}: N={row['total']}, "
        f"declared GenAI use: {row['ai_used']}, "
        f"declared not used: {row['not_used']}, "
        f"GenAI usage rate: {row['ai_rate_%']}%"
    )


# ──────────────────────────────────────────────────────────────────────────
# ## 2. Overall Statistics
# ──────────────────────────────────────────────────────────────────────────


# Calculate key statistics — use df (renamed copy) not df_tools_filtered
total_papers         = df['paper_id'].nunique()
papers_with_ai       = df[df['ai_used']]['paper_id'].nunique()
papers_without_ai    = total_papers - papers_with_ai
adoption_rate        = (papers_with_ai / total_papers) * 100
total_tools          = df[df['tool'] != 'No AI tool']['tool'].nunique()
total_uses           = len(df[df['ai_used']])

print("KEY STATISTICS")
print("=" * 60)
print(f"Total papers analyzed:       {total_papers:,}")
print(f"Papers using AI tools:       {papers_with_ai:,} ({adoption_rate:.1f}%)")
print(f"Papers without AI tools:     {papers_without_ai:,} ({100-adoption_rate:.1f}%)")
print(f"Unique AI tools identified:  {total_tools}")
print(f"Total AI tool uses recorded: {total_uses:,}")
print()

# Tools per paper statistics
tools_per_paper = df[df['ai_used']].groupby('paper_id')['tool'].nunique()
print("Tools per Paper (AI users only):")
print(f"  Average:                   {tools_per_paper.mean():.2f}")
print(f"  Median:                    {tools_per_paper.median():.0f}")
print(f"  Maximum:                   {tools_per_paper.max()}")
print(f"  Papers using multiple tools: {(tools_per_paper > 1).sum()} ({(tools_per_paper > 1).sum()/len(tools_per_paper)*100:.1f}%)")


# Show papers with more than 6 AI tools
papers_more_than_6 = tools_per_paper[tools_per_paper > 5]

print("Papers with more than 6 AI tools:")
print("=" * 50)
print(f"Total papers with >4 tools: {len(papers_more_than_6)}")
print("\nDetailed list:")
for paper_id, tool_count in papers_more_than_6.items():
    print(f"{paper_id}: {tool_count} tools")


tool_counts1 = df[df['tool'] != 'No AI tool']['tool'].unique()
len(tool_counts1)


# ──────────────────────────────────────────────────────────────────────────
# ## 3. Tool Popularity Analysis
# ──────────────────────────────────────────────────────────────────────────


# Analyze tool popularity
tool_counts = df[df['tool'] != 'No AI tool']['tool'].value_counts()

print("TOP 15 MOST USED AI TOOLS:")
print("-"*60)
for i, (tool, count) in enumerate(tool_counts.head(15).items(), 1):
    percentage = (count / total_uses) * 100
    papers = df[df['tool'] == tool]['paper_id'].nunique()
    print(f"{i:2}. {tool:<25} {count:4} uses ({percentage:5.1f}%) in {papers} papers")


# total_tools = df_tools_filtered[df_tools_filtered['tool'] != 'No AI tool']['tool'].nunique()


df_tools_filtered_ai = df[
    (df['tool'] != 'No AI tool') & 
    (df['tool'] != 'AI (unspecified)')
]
total_tools = df_tools_filtered_ai['tool'].nunique()
total_tools

tool_counts=df_tools_filtered_ai['tool'].value_counts()


import matplotlib.pyplot as plt
import numpy as np
import os

# Set professional style for research papers
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 16,
    'axes.labelsize': 18,
    'axes.titlesize': 18,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'axes.linewidth': 1.5,
    'axes.edgecolor': '#333333',
    'axes.labelcolor': '#333333',
    'text.color': '#333333',
    'xtick.color': '#333333',
    'ytick.color': '#333333',
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.grid': False,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

# Professional color palette (suitable for print and colorblind-friendly)
colors_blue = '#2E5A87'      # Deep blue
colors_gradient = plt.cm.Blues(np.linspace(0.4, 0.85, 15))  # Blue gradient

# ============================================================
# Figure 1: Top 15 Most Used AI Tools
# ============================================================
fig1, ax1 = plt.subplots(figsize=(9, 6), dpi=300)

top_15_tools = tool_counts.head(15)
y_pos = np.arange(len(top_15_tools))

# Create horizontal bars with professional blue gradient
bars = ax1.barh(y_pos, top_15_tools.values, 
                color=colors_gradient,
                edgecolor='#333333', 
                linewidth=0.7,
                height=0.75)

ax1.set_yticks(y_pos)
ax1.set_yticklabels(top_15_tools.index, fontsize=20)
ax1.set_xlabel('Number of Paper', fontsize=20)
# ax1.set_title('Top 15 Used AI Tools/Models', fontweight='bold', fontsize=18, pad=10)
ax1.invert_yaxis()

# Add value labels - INCREASED FONT SIZE
max_val = top_15_tools.values.max()
for bar, value in zip(bars, top_15_tools.values):
    ax1.text(value + max_val*0.02, bar.get_y() + bar.get_height()/2, 
             str(value), va='center', fontsize=16, fontweight='bold', color='#333333')

# Set x-axis limit to accommodate labels
ax1.set_xlim(0, max_val * 1.18)

# X-axis tick font size
ax1.tick_params(axis='x', labelsize=18)

# Add subtle spine styling
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['left'].set_linewidth(1.0)
ax1.spines['bottom'].set_linewidth(1.0)

# Add subtle horizontal grid lines only
ax1.xaxis.grid(True, linestyle='--', alpha=0.4, color='gray')
ax1.set_axisbelow(True)

plt.tight_layout()

# Create output folder if needed
os.makedirs(output_folder, exist_ok=True)

# Save the plot as PDF
plt.savefig(os.path.join(output_folder, 'fig_top_ai_tools.pdf'), bbox_inches='tight', facecolor='white')
plt.show()


# ============================================================
# Figure 2: Distribution of Papers by Number of AI Tools Used
# ============================================================
fig2, ax2 = plt.subplots(figsize=(8, 5))

# Create a copy and group values > 6 into "6+"
tools_per_paper_grouped = tools_per_paper.copy()
tools_per_paper_grouped = tools_per_paper_grouped.apply(lambda x: 6 if x >= 6 else x)

# Get distribution
tool_dist = tools_per_paper_grouped.value_counts().sort_index()

# Create labels (replace 6 with "6+")
labels = [str(int(x)) if x < 6 else "6+" for x in tool_dist.index]

# Professional gray-blue color scheme
bar_colors = plt.cm.GnBu(np.linspace(0.3, 0.8, len(tool_dist)))
bars2 = ax2.bar(labels, tool_dist.values, 
                color=bar_colors,
                edgecolor='#333333', 
                linewidth=0.8,
                width=0.6)
ax2.set_xlabel('Number of Tools', fontsize=20)
ax2.set_ylabel('Number of Papers', fontsize=20)
# ax2.set_title('Distribution of Papers by AI Tools/Models Count', fontweight='bold', pad=15)

# Add value labels
max_val = tool_dist.values.max()
for i, (x, y) in enumerate(zip(range(len(tool_dist)), tool_dist.values)):
    ax2.text(x, y + max_val*0.02, str(y), ha='center', va='bottom', 
             fontsize=16, fontweight='bold', color='#333333')

# Set y-axis limit to accommodate labels
ax2.set_ylim(0, max_val * 1.12)

# Remove top and right spines
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['left'].set_linewidth(0.8)
ax2.spines['bottom'].set_linewidth(0.8)

# Add subtle horizontal grid lines only
ax2.yaxis.grid(True, linestyle='--', alpha=0.4, color='gray')
ax2.set_axisbelow(True)

plt.tight_layout()

# Create output folder if needed
os.makedirs(output_folder, exist_ok=True)

# Save the plot as PDF
plt.savefig(os.path.join(output_folder, 'fig_papers_by_tool_count.pdf'), bbox_inches='tight', facecolor='white')
plt.show()


# ──────────────────────────────────────────────────────────────────────────
# ## 4. Contribution Roles Analysis
# ──────────────────────────────────────────────────────────────────────────


# Extract and analyze contribution roles
all_roles = []
for roles in df[df['contribution_roles'] != 'No contribution mentioned']['contribution_roles']:
    if pd.notna(roles):
        all_roles.extend([r.strip() for r in roles.split(',')])

role_counts = pd.Series(all_roles).value_counts()

print("TOP 15 CONTRIBUTION ROLES:")
print("-"*60)
for i, (role, count) in enumerate(role_counts.head(15).items(), 1):
    percentage = (count / len(all_roles)) * 100
    print(f"{i:2}. {role:<45} {count:4} ({percentage:5.1f}%)")


import matplotlib.pyplot as plt
import numpy as np
import os

# Set professional style for research papers
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 16,
    'axes.labelsize': 18,
    'axes.titlesize': 18,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'axes.linewidth': 1.5,
    'axes.edgecolor': '#333333',
    'axes.labelcolor': '#333333',
    'text.color': '#333333',
    'xtick.color': '#333333',
    'ytick.color': '#333333',
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.grid': False,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

# Extract and analyze contribution roles
all_roles = []
for roles in df[df['contribution_roles'] != 'No contribution mentioned']['contribution_roles']:
    if pd.notna(roles):
        all_roles.extend([r.strip() for r in roles.split(',')])

role_counts = pd.Series(all_roles).value_counts()

print("TOP 15 CONTRIBUTION ROLES:")
print("-"*60)
for i, (role, count) in enumerate(role_counts.head(15).items(), 1):
    percentage = (count / len(all_roles)) * 100
    print(f"{i:2}. {role:<45} {count:4} ({percentage:5.1f}%)")

# ============================================================
# Figure: Top 10 AI Contribution Roles
# ============================================================
fig, ax = plt.subplots(figsize=(8, 5))

# Professional gray-blue color scheme (same as fig_papers_by_tool_count.pdf)
top_roles = role_counts.head(15)
bar_colors = plt.cm.GnBu(np.linspace(0.3, 0.8, len(top_roles)))

y_pos = np.arange(len(top_roles))

# Create horizontal bars
bars = ax.barh(y_pos, top_roles.values, 
               color=bar_colors,
               edgecolor='#333333', 
               linewidth=0.8,
               height=0.7)

ax.set_yticks(y_pos)
ax.set_yticklabels([role[:35] + '...' if len(role) > 35 else role for role in top_roles.index])
ax.set_xlabel('Frequency')
# ax.set_title('Top 10 AI Contribution Roles', fontweight='bold', pad=10)
ax.invert_yaxis()

# Add value labels
max_val = top_roles.values.max()
for bar, value in zip(bars, top_roles.values):
    ax.text(value + max_val*0.02, bar.get_y() + bar.get_height()/2, 
            str(value), va='center', fontsize=16, fontweight='bold', color='#333333')

# Set x-axis limit to accommodate labels
ax.set_xlim(0, max_val * 1.15)

# Remove top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(0.8)
ax.spines['bottom'].set_linewidth(0.8)

# Add subtle horizontal grid lines only
ax.xaxis.grid(True, linestyle='--', alpha=0.4, color='gray')
ax.set_axisbelow(True)

plt.tight_layout()

# Create output folder if needed
os.makedirs(output_folder, exist_ok=True)

# Save the plot as PDF
plt.savefig(os.path.join(output_folder, 'fig_contribution_roles.pdf'), bbox_inches='tight', facecolor='white')
plt.show()


#........................All Tool..........................
import pandas as pd

# Extract and analyze contribution roles with enhanced metrics
all_roles = []
role_paper_mapping = {}  # Track which papers use which roles

for idx, row in df[df['contribution_roles'] != 'No contribution mentioned'].iterrows():
    roles = row['contribution_roles']
    paper_id = row['paper_id']
    tool = row['tool']
    
    if pd.notna(roles):
        for role in [r.strip() for r in roles.split(',')]:
            all_roles.append({'role': role, 'paper_id': paper_id, 'tool': tool})

roles_df = pd.DataFrame(all_roles)

# Calculate metrics
role_counts = roles_df['role'].value_counts()
total_count = len(all_roles)

# Create enhanced table
table_data = []
cumulative = 0

# Define categories
category_mapping = {
    'Grammar and spelling check': 'Language Enhancement',
    'Paraphrase and reword': 'Language Enhancement',
    'Improve writing style': 'Language Enhancement',
    'Text Translation': 'Translation',
    'Drafting content': 'Content Generation',
    'Plagiarism detection': 'Quality Assurance',
    'Generate images': 'Visual/Media',
    'Formatting assistance': 'Technical',
    'Abstract drafting': 'Content Generation',
    'Generate literature review': 'Content Generation',
    'Content enhancement': 'Language Enhancement',
    'Citation management': 'Technical',
    'Peer review simulation': 'Quality Assurance',
    'Fact checking': 'Quality Assurance',
    'Code assistance': 'Technical'
}

for i, (role, count) in enumerate(role_counts.head(15).items(), 1):
    percentage = (count / total_count) * 100
    cumulative += percentage
    
    # Get unique papers for this role
    unique_papers = roles_df[roles_df['role'] == role]['paper_id'].nunique()
    
    # Get ALL tools for this role with their counts
    tool_counts = roles_df[roles_df['role'] == role]['tool'].value_counts()
    all_tools = ', '.join([f"{tool} ({cnt})" for tool, cnt in tool_counts.items()])
    
    # Average per paper
    avg_per_paper = count / unique_papers if unique_papers > 0 else 0
    
    # Category
    category = category_mapping.get(role, 'Other')
    
    table_data.append({
        'Rank': i,
        'Contribution Role': role,
        'Count': count,
        'Percentage (%)': round(percentage, 1),
        'Cumulative (%)': round(cumulative, 1),
        'Category': category,
        'Unique Papers': unique_papers,
        'Avg per Paper': round(avg_per_paper, 2),
        'All Tools (count)': all_tools
    })

# Create DataFrame
enhanced_table = pd.DataFrame(table_data)

# Display table
print("\nENHANCED CONTRIBUTION ROLES TABLE:")
print("="*120)
print(enhanced_table.to_string(index=False))

# Save to CSV
enhanced_table.to_csv(os.path.join(output_folder, 'contribution_roles_table.csv'), index=False)

# Print summary insights
print("\n" + "="*60)
print("KEY INSIGHTS:")
print("="*60)
print(f"• Top 3 roles account for {enhanced_table['Cumulative (%)'].iloc[2]:.1f}% of all AI contributions")
print(f"• Language Enhancement is the dominant category")
print(f"• Total unique contribution instances: {total_count}")

# Optional: Create a separate detailed tool breakdown by role
print("\n" + "="*60)
print("DETAILED TOOL BREAKDOWN BY CONTRIBUTION ROLE:")
print("="*60)
for role in role_counts.head(15).index:
    print(f"\n{role}:")
    tool_breakdown = roles_df[roles_df['role'] == role]['tool'].value_counts()
    for tool, count in tool_breakdown.items():
        percentage = (count / role_counts[role]) * 100
        print(f"  • {tool}: {count} ({percentage:.1f}%)")


import pandas as pd

all_roles = []
for idx, row in df[df['contribution_roles'] != 'No contribution mentioned'].iterrows():
    roles = row['contribution_roles']
    paper_id = row['paper_id']
    tool = row['tool']
    if pd.notna(roles):
        for role in [r.strip() for r in roles.split(',')]:
            all_roles.append({'role': role, 'paper_id': paper_id, 'tool': tool})

roles_df = pd.DataFrame(all_roles)
role_counts = roles_df['role'].value_counts()
total_count = len(all_roles)

category_mapping = {
    'Grammar and spelling check': 'Language Enhancement',
    'Paraphrase and reword': 'Language Enhancement',
    'Improve writing style': 'Language Enhancement',
    'Text Translation': 'Translation',
    'Drafting content': 'Content Generation',
    'Plagiarism detection': 'Quality Assurance',
    'Generate images': 'Visual/Media',
    'Formatting assistance': 'Technical',
    'Abstract drafting': 'Content Generation',
    'Generate literature review': 'Content Generation',
    'Content enhancement': 'Language Enhancement',
    'Citation management': 'Technical',
    'Peer review simulation': 'Quality Assurance',
    'Fact checking': 'Quality Assurance',
    'Code assistance': 'Technical'
}

table_data = []
cumulative = 0

# ── removed .head(15) ──
for i, (role, count) in enumerate(role_counts.items(), 1):
    percentage = (count / total_count) * 100
    cumulative += percentage
    unique_papers = roles_df[roles_df['role'] == role]['paper_id'].nunique()
    tool_counts = roles_df[roles_df['role'] == role]['tool'].value_counts().head(6)
    top5_tools = ', '.join([f"{tool} ({cnt})" for tool, cnt in tool_counts.items()])
    avg_per_paper = count / unique_papers if unique_papers > 0 else 0
    category = category_mapping.get(role, 'Other')

    table_data.append({
        'Rank': i,
        'Contribution Role': role,
        'Count': count,
        'Percentage (%)': round(percentage, 1),
        'Cumulative (%)': round(cumulative, 1),
        'Category': category,
        'Unique Papers': unique_papers,
        'Avg per Paper': round(avg_per_paper, 2),
        'All Tools (count)': top5_tools
    })

enhanced_table = pd.DataFrame(table_data)

print("\nENHANCED CONTRIBUTION ROLES TABLE:")
print("="*120)
print(enhanced_table.to_string(index=False))          # ← fixed

enhanced_table.to_csv(                                 # ← fixed
    os.path.join(output_folder, 'contribution_roles_table.csv'), index=False
)

print("\n" + "="*60)
print("KEY INSIGHTS:")
print("="*60)
print(f"• Total unique roles found: {len(role_counts)}")
print(f"• Top 3 roles account for {enhanced_table['Cumulative (%)'].iloc[2]:.1f}% of all AI contributions")
print(f"• Language Enhancement is the dominant category")
print(f"• Total unique contribution instances: {total_count}")

print("\n" + "="*60)
print("DETAILED TOOL BREAKDOWN BY CONTRIBUTION ROLE:")
print("="*60)
# ── removed .head(15) ──
for role in role_counts.index:
    print(f"\n{role}:")
    tool_breakdown = roles_df[roles_df['role'] == role]['tool'].value_counts().head(5)
    role_total = role_counts[role]
    for tool, cnt in tool_breakdown.items():
        pct = (cnt / role_total) * 100
        print(f"  • {tool}: {cnt} ({pct:.1f}%)")


df


import pandas as pd

total_unique_papers = df['paper_id'].nunique()

# ── Extract contribution roles ─────────────────────────────────────────────────
all_roles = []
for idx, row in df[df['contribution_roles'] != 'No contribution mentioned'].iterrows():
    roles    = row['contribution_roles']
    paper_id = row['paper_id']
    tool     = row['tool']
    if pd.notna(roles):
        for role in [r.strip() for r in roles.split(',')]:
            all_roles.append({'role': role, 'paper_id': paper_id, 'tool': tool})

roles_df    = pd.DataFrame(all_roles)
role_counts = roles_df['role'].value_counts()
total_count = len(all_roles)  # total role-instance mentions

# ── Category mapping ───────────────────────────────────────────────────────────
category_mapping = {
    'Grammar and spelling check':  'Language Enhancement',
    'Paraphrase and reword':       'Language Enhancement',
    'Improve writing style':       'Language Enhancement',
    'Content enhancement':         'Language Enhancement',
    'Text Translation':            'Translation',
    'Drafting content':            'Content Generation',
    'Abstract drafting':           'Content Generation',
    'Generate literature review':  'Content Generation',
    'Plagiarism detection':        'Quality Assurance',
    'Peer review simulation':      'Quality Assurance',
    'Fact checking':               'Quality Assurance',
    'Generate images':             'Visual/Media',
    'Formatting assistance':       'Technical',
    'Citation management':         'Technical',
    'Code assistance':             'Technical',
}

# ── Build enhanced table ───────────────────────────────────────────────────────
table_data = []
cumulative = 0

for i, (role, count) in enumerate(role_counts.head(15).items(), 1):

    # % of total role-instances (frequency share)
    pct_of_instances = (count / total_count) * 100
    cumulative += pct_of_instances

    # unique papers that declared this role
    unique_papers = roles_df[roles_df['role'] == role]['paper_id'].nunique()

    # % of ALL papers that used this role
    pct_of_papers = (unique_papers / total_unique_papers) * 100

    # avg distinct tools used per paper for this role
    avg_tools_per_paper = (
        roles_df[roles_df['role'] == role]
        .groupby('paper_id')['tool']
        .nunique()
        .mean()
    )

    # top 5 tools for this role
    tool_counts  = roles_df[roles_df['role'] == role]['tool'].value_counts().head(5)
    top5_tools   = ', '.join([f"{t} ({c})" for t, c in tool_counts.items()])

    category = category_mapping.get(role, 'Other')

    table_data.append({
        'Rank':                   i,
        'Contribution Role':      role,
        'Mentions':               count,
        '% of Mentions':          round(pct_of_instances, 1),
        'Cumulative (%)':         round(cumulative, 1),
        'Unique Papers':          unique_papers,
        '% of Papers':            round(pct_of_papers, 1),
        'Avg Tools/Paper':        round(avg_tools_per_paper, 2),
        'Category':               category,
        'Top 5 Tools (count)':    top5_tools,
    })

enhanced_table = pd.DataFrame(table_data)

# ── Display ────────────────────────────────────────────────────────────────────
print("\nENHANCED CONTRIBUTION ROLES TABLE:")
print("=" * 120)
print(enhanced_table.to_string(index=False))

# ── Export ─────────────────────────────────────────────────────────────────────
os.makedirs(output_folder, exist_ok=True)
enhanced_table.to_csv(os.path.join(output_folder, 'contribution_roles_table.csv'), index=False)
print(f"\n✓ CSV exported to '{output_folder}/contribution_roles_table.csv'")

# ── Summary insights ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("KEY INSIGHTS:")
print("=" * 60)
print(f"• Total role-mention instances:        {total_count:,}")
print(f"• Total unique papers in analysis:     {total_unique_papers:,}")
print(f"• Top 3 roles account for:             {enhanced_table['Cumulative (%)'].iloc[2]:.1f}% of all mentions")
print(f"• Most common role covers:             {enhanced_table['% of Papers'].iloc[0]:.1f}% of all papers")

# ── Detailed tool breakdown per role ──────────────────────────────────────────
print("\n" + "=" * 60)
print("DETAILED TOOL BREAKDOWN BY CONTRIBUTION ROLE (TOP 5):")
print("=" * 60)
for role in role_counts.head(15).index:
    print(f"\n{role}:")
    tool_breakdown = roles_df[roles_df['role'] == role]['tool'].value_counts().head(5)
    role_total     = role_counts[role]
    for tool, cnt in tool_breakdown.items():
        pct = (cnt / role_total) * 100
        print(f"  • {tool}: {cnt} ({pct:.1f}%)")


# ──────────────────────────────────────────────────────────────────────────
# ## 5. Temporal Analysis
# ──────────────────────────────────────────────────────────────────────────


# ── Add year and year_month columns to df (from event_date) ──────────────────
df['year']       = df['event_date_parsed'].dt.year
df['year_month'] = df['event_year_month'].astype(str)

# ── Deduplicate to one row per paper (for correct counting) ───────────────────
df_papers = (
    df.groupby(['paper_id', 'volume_id', 'year', 'year_month'], as_index=False)['ai_used']
    .max()
)

# ════════════════════════════════════════════════════════════════════════════════
# TEMPORAL STATISTICS BY YEAR
# ════════════════════════════════════════════════════════════════════════════════
print("TEMPORAL STATISTICS BY YEAR:")
print("-" * 60)

year_stats = (
    df_papers.groupby('year')
    .agg(
        total_papers  = ('paper_id',  'nunique'),
        papers_using_ai = ('ai_used', 'sum'),      # sum of True = count of AI papers
    )
    .reset_index()
)
year_stats['ai_adoption_rate'] = (year_stats['papers_using_ai'] / year_stats['total_papers'] * 100).round(1)
year_stats['papers_no_ai']     = year_stats['total_papers'] - year_stats['papers_using_ai']

for _, row in year_stats.iterrows():
    print(f"Year {int(row['year'])}: "
          f"{int(row['total_papers'])} papers total, "
          f"{int(row['papers_using_ai'])} using AI "
          f"({row['ai_adoption_rate']:.1f}%), "
          f"{int(row['papers_no_ai'])} without AI")

# ════════════════════════════════════════════════════════════════════════════════
# TEMPORAL STATISTICS BY MONTH
# ════════════════════════════════════════════════════════════════════════════════
print("\nTEMPORAL STATISTICS BY MONTH:")
print("-" * 60)

month_stats = (
    df_papers.groupby('year_month')
    .agg(
        total_papers    = ('paper_id', 'nunique'),
        papers_using_ai = ('ai_used',  'sum'),
    )
    .reset_index()
    .sort_values('year_month')
)
month_stats['ai_adoption_rate'] = (month_stats['papers_using_ai'] / month_stats['total_papers'] * 100).round(1)
month_stats['papers_no_ai']     = month_stats['total_papers'] - month_stats['papers_using_ai']

for _, row in month_stats.iterrows():
    print(f"{row['year_month']}: "
          f"{int(row['total_papers'])} papers total, "
          f"{int(row['papers_using_ai'])} using AI "
          f"({row['ai_adoption_rate']:.1f}%), "
          f"{int(row['papers_no_ai'])} without AI")

# ════════════════════════════════════════════════════════════════════════════════
# YEAR-OVER-YEAR CHANGE
# ════════════════════════════════════════════════════════════════════════════════
if len(year_stats) > 1:
    year_stats_idx = year_stats.set_index('year')
    yoy_change = year_stats_idx['ai_adoption_rate'].pct_change() * 100
    print("\nYear-over-Year Growth in AI Adoption Rate:")
    print("-" * 60)
    for year, change in yoy_change.dropna().items():
        print(f"  {int(year)}: {change:+.1f}%")

# ════════════════════════════════════════════════════════════════════════════════
# MONTH-OVER-MONTH CHANGE (last 6 months)
# ════════════════════════════════════════════════════════════════════════════════
if len(month_stats) > 1:
    month_stats_idx = month_stats.set_index('year_month')
    mom_change = month_stats_idx['ai_adoption_rate'].pct_change() * 100
    print("\nMonth-over-Month Growth in AI Adoption Rate (last 6 months):")
    print("-" * 60)
    for ym, change in mom_change.dropna().tail(6).items():
        print(f"  {ym}: {change:+.1f}%")

# ── Full summary table ────────────────────────────────────────────────────────
print("\nYEAR SUMMARY TABLE:")
print("=" * 60)
print(year_stats.to_string(index=False))

print("\nMONTH SUMMARY TABLE:")
print("=" * 60)
print(month_stats.to_string(index=False))


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# ── Style (matching your figure exactly) ─────────────────────────────────────
plt.rcParams.update({
    'font.family':      'serif',
    'font.serif':       ['Times New Roman', 'DejaVu Serif'],
    'font.size':        14,
    'axes.labelsize':   18,
    'axes.titlesize':   18,
    'xtick.labelsize':  14,
    'ytick.labelsize':  16,
    'axes.linewidth':   1.5,
    'axes.edgecolor':   '#333333',
    'axes.labelcolor':  '#333333',
    'text.color':       '#333333',
    'xtick.color':      '#333333',
    'ytick.color':      '#333333',
    'figure.facecolor': 'white',
    'axes.facecolor':   'white',
    'axes.grid':        True,
    'figure.dpi':       300,
    'savefig.dpi':      300,
    'savefig.bbox':     'tight',
})

# ── Colors (matching your figure) ────────────────────────────────────────────
color_total = '#1f4e79'   # Dark navy blue  — Total Papers
color_ai    = '#c0392b'   # Dark red        — Papers Using AI

# ── Build monthly trend using event_year_month ────────────────────────────────
df_papers_event = (
    df.groupby(['paper_id', 'volume_id', 'event_year_month'], as_index=False)['ai_used']
    .max()
)
df_papers_event['year_month'] = df_papers_event['event_year_month'].astype(str)

# Total papers per month
monthly_total = (
    df_papers_event.groupby('year_month')['paper_id']
    .nunique().reset_index()
    .rename(columns={'paper_id': 'Total Papers'})
)

# AI papers per month
monthly_ai = (
    df_papers_event[df_papers_event['ai_used'] == True]
    .groupby('year_month')['paper_id']
    .nunique().reset_index()
    .rename(columns={'paper_id': 'Papers Using AI'})
)

monthly = (
    monthly_total
    .merge(monthly_ai, on='year_month', how='left')
)
monthly['Papers Using AI'] = monthly['Papers Using AI'].fillna(0).astype(int)
monthly = monthly.sort_values('year_month').set_index('year_month')

print(monthly)

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 6))

x_pos = np.arange(len(monthly))
x_labels = monthly.index.tolist()

total_vals = monthly['Total Papers'].values
ai_vals    = monthly['Papers Using AI'].values

# Lines with markers
ax.plot(x_pos, total_vals,
        marker='o', linewidth=2.5, markersize=7,
        color=color_total, label='Total Papers', zorder=3)

ax.plot(x_pos, ai_vals,
        marker='s', linewidth=2.5, markersize=7,
        color=color_ai, label='Papers Using AI', zorder=3)

# ── Data labels on every point (matching your figure style) ──────────────────
for i, (t, a) in enumerate(zip(total_vals, ai_vals)):
    # Total Papers label — above the point
    ax.text(i, t + max(total_vals) * 0.015,
            str(t), ha='center', va='bottom',
            fontsize=11, fontweight='bold', color=color_total)
    # Papers Using AI label — below the point (avoid overlap)
    offset = max(total_vals) * 0.04
    va = 'top' if a < t else 'bottom'
    ax.text(i, a - offset,
            str(a), ha='center', va='top',
            fontsize=11, fontweight='bold', color=color_ai)

# ── Axes formatting ───────────────────────────────────────────────────────────
ax.set_xticks(x_pos)
ax.set_xticklabels(x_labels, rotation=45, ha='right')
ax.set_xlabel('Time Period (Year-Month)', fontsize=18, labelpad=10)
ax.set_ylabel('Number of Papers',        fontsize=18, labelpad=10)

# Y-axis: give headroom for labels
ax.set_ylim(0, max(total_vals) * 1.18)

# Grid (light, matching figure)
ax.yaxis.grid(True, linestyle='--', alpha=0.5, color='gray')
ax.xaxis.grid(False)
ax.set_axisbelow(True)

# Spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(1.0)
ax.spines['bottom'].set_linewidth(1.0)

# Legend (matching position and style from your figure)
ax.legend(loc='upper left', frameon=True, fancybox=False,
          edgecolor='#333333', fontsize=16,
          borderpad=0.8, labelspacing=0.5)

plt.tight_layout()

# ── Export ────────────────────────────────────────────────────────────────────
os.makedirs(output_folder, exist_ok=True)
out_path = os.path.join(output_folder, 'fig_monthly_ai_trend_event_date.pdf')
plt.savefig(out_path, bbox_inches='tight', facecolor='white')
plt.show()
print(f"\n✓ Saved to '{out_path}'")

# ── Summary ───────────────────────────────────────────────────────────────────
total_p  = df_papers_event['paper_id'].nunique()
total_ai = df_papers_event[df_papers_event['ai_used'] == True]['paper_id'].nunique()
print(f"\nSUMMARY (by Event Date):")
print("=" * 60)
print(monthly[['Total Papers', 'Papers Using AI']].to_string())
print("-" * 60)
print(f"Total unique papers:      {total_p}")
print(f"Total papers using AI:    {total_ai}  ({total_ai/total_p*100:.1f}%)")
print(f"Total papers without AI:  {total_p - total_ai}  ({(total_p-total_ai)/total_p*100:.1f}%)")


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':      'serif',
    'font.serif':       ['Times New Roman', 'DejaVu Serif'],
    'font.size':        16,
    'axes.labelsize':   18,
    'axes.titlesize':   18,
    'xtick.labelsize':  14,
    'ytick.labelsize':  18,
    'axes.linewidth':   1.5,
    'axes.edgecolor':   '#333333',
    'axes.labelcolor':  '#333333',
    'text.color':       '#333333',
    'xtick.color':      '#333333',
    'ytick.color':      '#333333',
    'figure.facecolor': 'white',
    'axes.facecolor':   'white',
    'axes.grid':        False,
    'figure.dpi':       300,
    'savefig.dpi':      300,
    'savefig.bbox':     'tight',
})

# ── Derive year_month string from event_year_month ────────────────────────────
df['year_month_str'] = df['event_year_month'].astype(str)

# ── Filter: keep only actual AI tools ────────────────────────────────────────
df_ai_only = df[
    (df['tool'] != 'No AI tool') &
    (df['tool'] != 'AI (unspecified)')
].copy()

print(f"Rows after filtering: {len(df_ai_only)}")
print(f"Unique tools remaining: {df_ai_only['tool'].nunique()}")

# ── Top 20 tools overall (by raw mention count) ───────────────────────────────
top_20_tools = df_ai_only['tool'].value_counts().head(20).index.tolist()
print(f"\nTop 20 tools:\n{top_20_tools}")

# ── Group by year_month × tool, count mentions ────────────────────────────────
tool_evolution = (
    df_ai_only
    .groupby(['year_month_str', 'tool'])
    .size()
    .unstack(fill_value=0)
)

# ── Keep only top 20 tools, fill missing with 0 ───────────────────────────────
# Some top tools may not appear in every month — reindex ensures all columns exist
tool_evolution = tool_evolution.reindex(columns=top_20_tools, fill_value=0)

# ── Sort chronologically ──────────────────────────────────────────────────────
tool_evolution = tool_evolution.sort_index()

print(f"\nMonths in evolution table: {len(tool_evolution)}")
print(f"Shape: {tool_evolution.shape}")

# ── Figure: Stacked Area ──────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 8))

colors = plt.cm.tab20(np.linspace(0, 1, 20))

ax.stackplot(
    tool_evolution.index,
    [tool_evolution[tool] for tool in top_20_tools],
    labels=top_20_tools,
    colors=colors,
    alpha=0.85
)

# ── Axes ──────────────────────────────────────────────────────────────────────
ax.set_xlabel('Time Period (Year-Month)', fontsize=18, labelpad=10)
ax.set_ylabel('Number of Mentions',      fontsize=18, labelpad=10)
ax.set_title('Top 20 AI Tools Evolution Over Time (by Event Date)',
             fontweight='bold', pad=15, fontsize=18)

plt.xticks(rotation=45, ha='right', fontsize=13)

# ── Legend outside plot to avoid overlap ─────────────────────────────────────
ax.legend(
    loc='upper left',
    bbox_to_anchor=(1.01, 1),
    frameon=True,
    fancybox=False,
    edgecolor='#333333',
    fontsize=13,
    title='AI Tools',
    title_fontsize=14,
)

# ── Spines & grid ─────────────────────────────────────────────────────────────
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(0.8)
ax.spines['bottom'].set_linewidth(0.8)
ax.yaxis.grid(True, linestyle='--', alpha=0.4, color='gray')
ax.set_axisbelow(True)

plt.tight_layout()

# ── Export ────────────────────────────────────────────────────────────────────
os.makedirs(output_folder, exist_ok=True)
out_path = os.path.join(output_folder, 'fig_tools_evolution_stacked.pdf')
plt.savefig(out_path, bbox_inches='tight', facecolor='white')
plt.show()
print(f"\n✓ Saved to '{out_path}'")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\nTOP 20 TOOLS MONTHLY USAGE:")
print("=" * 80)
print(tool_evolution.to_string())

print("\nTOTAL MENTIONS PER TOOL (across all months):")
print("-" * 40)
print(tool_evolution.sum().sort_values(ascending=False).to_string())


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Set professional style for research papers - optimized for double column
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 18,
    'axes.labelsize': 18,
    'axes.titlesize': 18,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'axes.linewidth': 1.5,
    'axes.edgecolor': '#333333',
    'axes.labelcolor': '#333333',
    'text.color': '#333333',
    'xtick.color': '#333333',
    'ytick.color': '#333333',
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.grid': False,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

# Filter out "No AI tool" entries
df_ai_only = df[df['tool'] != 'No AI tool'].copy()
df_ai_only = df_ai_only[df_ai_only['tool'] != 'AI (unspecified)']

# Ensure year_month is string
df_ai_only['year_month'] = df_ai_only['year_month'].astype(str)

# Get top 10 tools overall (reduced from 20 for clarity)
top_10_tools = df_ai_only['tool'].value_counts().head(20).index.tolist()

# Group by year_month and tool, count occurrences
tool_evolution = df_ai_only.groupby(['year_month', 'tool']).size().unstack(fill_value=0)

# Filter to only top 10 tools
tool_evolution = tool_evolution[top_10_tools]

# Sort by month
tool_evolution = tool_evolution.sort_index()

# ============================================================
# Figure: Top 10 Tools Evolution - Stacked Bar Chart
# ============================================================
# Double column width is typically 3.33 inches (8.45 cm)
fig, ax = plt.subplots(figsize=(11, 6))  # Width for double column

# Professional distinct color palette
colors = plt.cm.tab20(np.linspace(0, 1, 20))

# Create stacked bar chart
x = np.arange(len(tool_evolution.index))
bottom = np.zeros(len(tool_evolution.index))

for i, tool in enumerate(top_10_tools):
    ax.bar(x, tool_evolution[tool], bottom=bottom, 
           label=tool, color=colors[i], edgecolor='white', linewidth=0.3)
    bottom += tool_evolution[tool].values

# Labels
ax.set_xlabel('Month', fontsize=20)
ax.set_ylabel('Number of Usage', fontsize=20)
ax.set_xticks(x)
ax.set_xticklabels(tool_evolution.index, rotation=45, ha='right', fontsize=18)

# Legend - positioned outside to avoid overlapping
# ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), frameon=True, 
#           fancybox=False, edgecolor='#333333', fontsize=16, ncol=1)

ax.legend(loc='upper left', frameon=True, fancybox=False, 
          edgecolor='#333333', fontsize=16.5, ncol=2)

# Spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(0.8)
ax.spines['bottom'].set_linewidth(0.8)

# Add subtle grid lines
ax.yaxis.grid(True, linestyle='--', alpha=0.3, color='gray', linewidth=0.5)
ax.set_axisbelow(True)

plt.tight_layout()

# Create output folder if needed
os.makedirs(output_folder, exist_ok=True)

# Save
plt.savefig(os.path.join(output_folder, 'fig_tools_evolution_stacked_bars.pdf'), 
            bbox_inches='tight', facecolor='white', dpi=300)
plt.show()

# Print summary
print("\nTOP 10 TOOLS MONTHLY USAGE:")
print("="*80)
print(tool_evolution.to_string())



import pandas as pd
import numpy as np
import os

# ── Ensure correct types ──────────────────────────────────────────────────────
df['ai_used'] = df['ai_used'].astype(bool)

# ── Total unique papers ───────────────────────────────────────────────────────
total_papers = df['paper_id'].nunique()

# ── Papers with AI (deduplicated at paper level) ──────────────────────────────
df_papers = (
    df.groupby(['paper_id', 'volume_id'], as_index=False)['ai_used']
    .max()
)
papers_with_ai    = df_papers[df_papers['ai_used'] == True]['paper_id'].nunique()
papers_without_ai = total_papers - papers_with_ai
adoption_rate     = (papers_with_ai / total_papers) * 100

# ── Tool metrics ──────────────────────────────────────────────────────────────
df_ai_tools = df[
    ~df['tool'].isin(['No AI tool', 'AI (unspecified)']) &
    df['tool'].notna()
].copy()

# Unique tools across dataset
total_tools = df_ai_tools['tool'].nunique()

# Total uses = unique (paper, tool) pairs — avoids inflation from role rows
total_uses = df_ai_tools.groupby(['paper_id', 'tool']).ngroups

# Distinct tools per paper (nunique handles multiple role rows for same tool)
tools_per_paper = df_ai_tools.groupby('paper_id')['tool'].nunique()

papers_multi_tools = (tools_per_paper > 1).sum()
multi_tool_rate    = papers_multi_tools / len(tools_per_paper) * 100

# Most used tool
tool_counts    = df_ai_tools['tool'].value_counts()
most_used_tool = tool_counts.index[0] if len(tool_counts) > 0 else 'N/A'

# ── Role metrics ──────────────────────────────────────────────────────────────
# ✅ FIXED: correct column name is 'contribution_roles'
all_roles = []
for roles in df[df['contribution_roles'] != 'No contribution mentioned']['contribution_roles']:
    if pd.notna(roles):
        all_roles.extend([r.strip() for r in roles.split(',')])
role_counts      = pd.Series(all_roles).value_counts()
most_common_role = role_counts.index[0] if len(role_counts) > 0 else 'N/A'

# ── Date range (using event_date_parsed) ──────────────────────────────────────
date_range = (
    f"{df['event_date_parsed'].min().strftime('%Y-%m')} to "
    f"{df['event_date_parsed'].max().strftime('%Y-%m')}"
)

# ── Summary table ─────────────────────────────────────────────────────────────
summary_export = pd.DataFrame({
    'Metric': [
        'Total Papers',
        'Papers Using AI',
        'Papers Not Using AI',
        'AI Adoption Rate (%)',
        'Unique AI Tools',
        'Total AI Tool Uses (unique paper×tool pairs)',
        'Average Distinct Tools per Paper (AI users)',
        'Median Distinct Tools per Paper (AI users)',
        'Maximum Tools per Paper',
        'Papers Using Multiple Tools',
        'Multi-tool Rate (%)',
        'Most Used Tool',
        'Most Common Contribution Role',
        'Event Date Range',
    ],
    'Value': [
        f"{total_papers:,}",
        f"{papers_with_ai:,}",
        f"{papers_without_ai:,}",
        f"{adoption_rate:.2f}",
        str(total_tools),
        f"{total_uses:,}",
        f"{tools_per_paper.mean():.2f}",
        f"{tools_per_paper.median():.0f}",
        str(tools_per_paper.max()),
        str(papers_multi_tools),
        f"{multi_tool_rate:.2f}",
        most_used_tool,
        most_common_role,
        date_range,
    ]
})

# ── Display ───────────────────────────────────────────────────────────────────
print("SUMMARY TABLE FOR EXPORT:")
print("=" * 60)
print(summary_export.to_string(index=False))

print("\n" + "=" * 60)
print("KEY STATISTICS")
print("=" * 60)
print(f"Total papers analyzed:             {total_papers:,}")
print(f"Papers using AI tools:             {papers_with_ai:,} ({adoption_rate:.1f}%)")
print(f"Papers without AI tools:           {papers_without_ai:,} ({100-adoption_rate:.1f}%)")
print(f"Unique AI tools identified:        {total_tools}")
print(f"Total AI tool uses (deduplicated): {total_uses:,}")
print()
print("Tools per Paper (AI users only):")
print(f"  Average:                         {tools_per_paper.mean():.2f}")
print(f"  Median:                          {tools_per_paper.median():.0f}")
print(f"  Maximum:                         {tools_per_paper.max()}")
print(f"  Papers using multiple tools:     {papers_multi_tools} ({multi_tool_rate:.1f}%)")

# ── Export CSV ────────────────────────────────────────────────────────────────
os.makedirs(output_folder, exist_ok=True)
summary_export.to_csv(os.path.join(output_folder, 'ai_usage_summary.csv'), index=False)
print(f"\n✓ Summary exported to '{output_folder}/ai_usage_summary.csv'")


# Tool counts (excluding "No AI tool")
tool_counts = df[df['tool'] != 'No AI tool']['tool'].value_counts()

# Total unique AI tools
total_tools = tool_counts.nunique()
total_tools
tool_counts
