

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ── Paths ──────────────────────────────────────────────────────────────────────
FILE_PATH     = r"path_to_your_json_file.json"
output_folder = r"path_to_output_folder"
os.makedirs(output_folder, exist_ok=True)

# ── Load JSON ──────────────────────────────────────────────────────────────────
with open(FILE_PATH, encoding='utf-8') as f:
    raw = json.load(f)

print(f'Total records in JSON: {len(raw)}')


# ──────────────────────────────────────────────────────────────────────────
# ## 2. Build Flat DataFrames
# ──────────────────────────────────────────────────────────────────────────


# ── df_papers: one row per paper ───────────────────────────────────────────────
paper_rows = []
for record in raw:                          # raw is a list, iterate directly
    if record is None:
        continue
    result = record.get('LLM_output_AI', {}) or {}
    paper_rows.append({
        'index_id':           record.get('index_id'),
        'paper_id':           record.get('paper_id'),
        'text':               record.get('Text-Ai-declaration', ''),
        'declaration_status': result.get('declaration_status', 'no_declaration'),
        'tools_and_models':   result.get('tools_and_models', []),
        'num_tools':          len(result.get('tools_and_models', [])),
        'status_inferred':    result.get('_status_inferred', False),
    })

df_papers = pd.DataFrame(paper_rows)
df_papers['ai_used'] = df_papers['declaration_status'] == 'declared_use'

print(f'df_papers shape: {df_papers.shape}')
print(df_papers['declaration_status'].value_counts())
df_papers.head()


df_papers


df_papers["paper_id"].value_counts()[lambda x: x > 1]


df_papers


# ── df_tools: one row per (index_id, tool) ─────────────────────────────────────
tool_rows = []
for record in raw:
    if record is None:
        continue
    result   = record.get('LLM_output_AI', {}) or {}
    index_id = record.get('index_id')
    paper_id = record.get('paper_id')
    status   = result.get('declaration_status', 'no_declaration')
    tools    = result.get('tools_and_models', [])
    usage    = result.get('usage', {})
    roles    = result.get('contribution_roles', {})

    if status == 'declared_no_use':
        tool_rows.append({
            'index_id':           index_id,
            'paper_id':           paper_id,
            'declaration_status': status,
            'tool':               'No AI tool',
            'usage_text':         '',
            'contribution_roles': 'No contribution mentioned',
        })
    elif tools:
        for tool in tools:
            tool_rows.append({
                'index_id':           index_id,
                'paper_id':           paper_id,
                'declaration_status': status,
                'tool':               tool,
                'usage_text':         usage.get(tool, ''),
                'contribution_roles': ', '.join(roles.get(tool, [])) if roles.get(tool) else 'No contribution mentioned',
            })

df_tools = pd.DataFrame(tool_rows)

print(f'df_tools shape: {df_tools.shape}')
df_tools.head(10)


df_tools


# ──────────────────────────────────────────────────────────────────────────
# ## 3. Overall Statistics
# ──────────────────────────────────────────────────────────────────────────


# ── Key counts ────────────────────────────────────────────────────────────────
total_papers       = df_papers['paper_id'].nunique()
declared_use       = (df_papers['declaration_status'] == 'declared_use').sum()
declared_no_use    = (df_papers['declaration_status'] == 'declared_no_use').sum()
no_declaration     = (df_papers['declaration_status'] == 'no_declaration').sum()
adoption_rate      = declared_use / total_papers * 100

# Tool stats — only from declared_use papers
df_ai = df_tools[
    (df_tools['tool'] != 'No AI tool') &
    (df_tools['tool'] != 'AI (unspecified)') &
    df_tools['tool'].notna()
].copy()

total_tools        = df_ai['tool'].nunique()
total_uses         = df_ai.groupby(['paper_id', 'tool']).ngroups
tools_per_paper    = df_ai.groupby('paper_id')['tool'].nunique()
papers_multi_tools = (tools_per_paper > 1).sum()
most_used_tool     = df_ai['tool'].value_counts().index[0] if len(df_ai) > 0 else 'N/A'

# Role stats
all_roles = []
for roles in df_tools[df_tools['contribution_roles'] != 'No contribution mentioned']['contribution_roles']:
    if pd.notna(roles):
        all_roles.extend([r.strip() for r in roles.split(',') if r.strip()])
role_counts      = pd.Series(all_roles).value_counts()
most_common_role = role_counts.index[0] if len(role_counts) > 0 else 'N/A'

print('KEY STATISTICS')
print('=' * 60)
print(f'Total papers analyzed:             {total_papers:,}')
print(f'Papers declaring AI use:           {declared_use:,} ({adoption_rate:.1f}%)')
print(f'Papers declaring NO AI use:        {declared_no_use:,} ({declared_no_use/total_papers*100:.1f}%)')
print(f'Papers with no declaration found:  {no_declaration:,} ({no_declaration/total_papers*100:.1f}%)')
print(f'Unique AI tools identified:        {total_tools}')
print(f'Total AI tool uses (deduplicated): {total_uses:,}')
print(f'Most used tool:                    {most_used_tool}')
print(f'Most common contribution role:     {most_common_role}')
print()
print('Tools per Paper (AI-declaring papers only):')
print(f'  Average:                         {tools_per_paper.mean():.2f}')
print(f'  Median:                          {tools_per_paper.median():.0f}')
print(f'  Maximum:                         {tools_per_paper.max()}')
print(f'  Papers using multiple tools:     {papers_multi_tools} ({papers_multi_tools/len(tools_per_paper)*100:.1f}%)')


# ── Papers with more than 5 tools ─────────────────────────────────────────────
papers_gt5 = tools_per_paper[tools_per_paper > 5]
print(f'Papers with more than 5 AI tools: {len(papers_gt5)}')
for pid, cnt in papers_gt5.items():
    print(f'  {pid}: {cnt} tools')


# ── Export summary CSV ─────────────────────────────────────────────────────────
summary_export = pd.DataFrame({
    'Metric': [
        'Total Papers',
        'Papers Declaring AI Use',
        'Papers Declaring No AI Use',
        'Papers with No Declaration Found',
        'AI Adoption Rate (%)',
        'Unique AI Tools',
        'Total AI Tool Uses (unique paper x tool pairs)',
        'Average Distinct Tools per Paper (AI users)',
        'Median Distinct Tools per Paper (AI users)',
        'Maximum Tools per Paper',
        'Papers Using Multiple Tools',
        'Multi-tool Rate (%)',
        'Most Used Tool',
        'Most Common Contribution Role',
    ],
    'Value': [
        f'{total_papers:,}',
        f'{declared_use:,}',
        f'{declared_no_use:,}',
        f'{no_declaration:,}',
        f'{adoption_rate:.2f}',
        str(total_tools),
        f'{total_uses:,}',
        f'{tools_per_paper.mean():.2f}',
        f'{tools_per_paper.median():.0f}',
        str(tools_per_paper.max()),
        str(papers_multi_tools),
        f'{papers_multi_tools/len(tools_per_paper)*100:.2f}',
        most_used_tool,
        most_common_role,
    ]
})

summary_export.to_csv(os.path.join(output_folder, 'ai_usage_summary.csv'), index=False)
print('Summary saved.')
print(summary_export.to_string(index=False))


# ──────────────────────────────────────────────────────────────────────────
# ## 4. Declaration Status Distribution
# ──────────────────────────────────────────────────────────────────────────


# ── Professional rcParams ──────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':      'serif',
    'font.serif':       ['Times New Roman', 'DejaVu Serif'],
    'font.size':        16,
    'axes.labelsize':   18,
    'axes.titlesize':   18,
    'xtick.labelsize':  16,
    'ytick.labelsize':  16,
    'axes.linewidth':   1.5,
    'axes.edgecolor':   '#333333',
    'axes.labelcolor':  '#333333',
    'text.color':       '#333333',
    'figure.facecolor': 'white',
    'axes.facecolor':   'white',
    'axes.grid':        False,
    'figure.dpi':       300,
    'savefig.dpi':      300,
    'savefig.bbox':     'tight',
})

status_counts = df_papers['declaration_status'].value_counts()
labels_map = {
    'declared_use':    'Declared AI Use',
    'declared_no_use': 'Declared No AI Use',
    'no_declaration':  'No Declaration Found',
}
labels   = [labels_map.get(k, k) for k in status_counts.index]
colors   = ['#2E5A87', '#c0392b', '#95a5a6']

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(labels, status_counts.values,
              color=colors[:len(labels)],
              edgecolor='#333333', linewidth=0.8, width=0.5)

max_val = status_counts.values.max()
for bar, val in zip(bars, status_counts.values):
    pct = val / total_papers * 100
    ax.text(bar.get_x() + bar.get_width()/2, val + max_val*0.02,
            f'{val}\n({pct:.1f}%)', ha='center', va='bottom',
            fontsize=14, fontweight='bold', color='#333333')

ax.set_ylabel('Number of Papers', fontsize=18)
ax.set_ylim(0, max_val * 1.20)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.grid(True, linestyle='--', alpha=0.4, color='gray')
ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'fig_declaration_status.png'), bbox_inches='tight', facecolor='white',dpi=600)
plt.show()
print('Figure saved.')


# ──────────────────────────────────────────────────────────────────────────
# ## 5. Tool Popularity Analysis
# ──────────────────────────────────────────────────────────────────────────


tool_counts = df_ai['tool'].value_counts()

print('TOP 15 MOST USED AI TOOLS:')
print('-' * 60)
for i, (tool, count) in enumerate(tool_counts.head(15).items(), 1):
    pct    = count / len(df_ai) * 100
    papers = df_ai[df_ai['tool'] == tool]['paper_id'].nunique()
    print(f'{i:2}. {tool:<30} {count:4} uses ({pct:5.1f}%) in {papers} papers')


# ── Figure: Top 15 AI Tools ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 6), dpi=300)

top_15      = tool_counts.head(15)
y_pos       = np.arange(len(top_15))
bar_colors  = plt.cm.Blues(np.linspace(0.4, 0.85, 15))

bars = ax.barh(y_pos, top_15.values,
               color=bar_colors, edgecolor='#333333',
               linewidth=0.7, height=0.75)

ax.set_yticks(y_pos)
ax.set_yticklabels(top_15.index, fontsize=16)
ax.set_xlabel('Number of Papers', fontsize=18)
ax.invert_yaxis()

max_val = top_15.values.max()
for bar, val in zip(bars, top_15.values):
    ax.text(val + max_val*0.02, bar.get_y() + bar.get_height()/2,
            str(val), va='center', fontsize=14, fontweight='bold', color='#333333')

ax.set_xlim(0, max_val * 1.18)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.xaxis.grid(True, linestyle='--', alpha=0.4, color='gray')
ax.set_axisbelow(True)
ax.tick_params(axis='x', labelsize=16)

plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'fig_top_ai_tools.png'), bbox_inches='tight', facecolor='white',dpi=600)
plt.show()
print('Figure saved.')


# ── Figure: Distribution of Papers by Number of Tools ─────────────────────────
tools_grouped = tools_per_paper.apply(lambda x: 6 if x >= 6 else x)
tool_dist     = tools_grouped.value_counts().sort_index()
tick_labels   = [str(int(x)) if x < 6 else '6+' for x in tool_dist.index]

fig, ax = plt.subplots(figsize=(8, 5))
bar_colors = plt.cm.GnBu(np.linspace(0.3, 0.8, len(tool_dist)))
bars = ax.bar(tick_labels, tool_dist.values,
              color=bar_colors, edgecolor='#333333',
              linewidth=0.8, width=0.6)

max_val = tool_dist.values.max()
for i, (x, y) in enumerate(zip(range(len(tool_dist)), tool_dist.values)):
    ax.text(x, y + max_val*0.02, str(y),
            ha='center', va='bottom', fontsize=14, fontweight='bold', color='#333333')

ax.set_xlabel('Number of Tools', fontsize=20)
ax.set_ylabel('Number of Papers', fontsize=20)
ax.set_ylim(0, max_val * 1.12)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.grid(True, linestyle='--', alpha=0.4, color='gray')
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'fig_papers_by_tool_count.png'), bbox_inches='tight', facecolor='white',dpi=600)
plt.show()
print('Figure saved.')


# ──────────────────────────────────────────────────────────────────────────
# ## 6. Contribution Roles Analysis
# ──────────────────────────────────────────────────────────────────────────


total_unique_papers = df_papers['paper_id'].nunique()

# ── Extract roles with paper + tool context ────────────────────────────────────
all_roles_list = []
for _, row in df_tools[df_tools['contribution_roles'] != 'No contribution mentioned'].iterrows():
    roles    = row['contribution_roles']
    paper_id = row['paper_id']
    tool     = row['tool']
    if pd.notna(roles):
        for role in [r.strip() for r in roles.split(',') if r.strip()]:
            all_roles_list.append({'role': role, 'paper_id': paper_id, 'tool': tool})

roles_df    = pd.DataFrame(all_roles_list)
role_counts = roles_df['role'].value_counts()
total_count = len(all_roles_list)

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
    'Fact-checking':               'Quality Assurance',
    'Generate images':             'Visual/Media',
    'Formatting assistance':       'Technical',
    'Citation management':         'Technical',
    'Code assistance':             'Technical',
}

# ── Build enhanced table ───────────────────────────────────────────────────────
table_data = []
cumulative = 0
for i, (role, count) in enumerate(role_counts.head(15).items(), 1):
    pct_instances  = count / total_count * 100
    cumulative    += pct_instances
    unique_papers  = roles_df[roles_df['role'] == role]['paper_id'].nunique()
    pct_papers     = unique_papers / total_unique_papers * 100
    avg_tools      = (roles_df[roles_df['role'] == role]
                      .groupby('paper_id')['tool'].nunique().mean())
    top5_tools     = roles_df[roles_df['role'] == role]['tool'].value_counts().head(5)
    top5_str       = ', '.join([f'{t} ({c})' for t, c in top5_tools.items()])
    category       = category_mapping.get(role, 'Other')

    table_data.append({
        'Rank':               i,
        'Contribution Role':  role,
        'Mentions':           count,
        '% of Mentions':      round(pct_instances, 1),
        'Cumulative (%)':     round(cumulative, 1),
        'Unique Papers':      unique_papers,
        '% of Papers':        round(pct_papers, 1),
        'Avg Tools/Paper':    round(avg_tools, 2),
        'Category':           category,
        'Top 5 Tools':        top5_str,
    })

enhanced_table = pd.DataFrame(table_data)
print('ENHANCED CONTRIBUTION ROLES TABLE:')
print('=' * 120)
print(enhanced_table.to_string(index=False))

enhanced_table.to_csv(os.path.join(output_folder, 'contribution_roles_table.csv'), index=False)
print(f'\n✓ CSV saved.')

print(f'\nKEY INSIGHTS:')
print(f'  Total role-mention instances:    {total_count:,}')
print(f'  Total unique papers:             {total_unique_papers:,}')
print(f'  Top 3 roles account for:         {enhanced_table["Cumulative (%)"].iloc[2]:.1f}% of all mentions')
print(f'  Most common role covers:         {enhanced_table["% of Papers"].iloc[0]:.1f}% of all papers')


# ── Figure: Top 15 Contribution Roles ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 6))

top_roles  = role_counts.head(15)
y_pos      = np.arange(len(top_roles))
bar_colors = plt.cm.GnBu(np.linspace(0.3, 0.8, len(top_roles)))

bars = ax.barh(y_pos, top_roles.values,
               color=bar_colors, edgecolor='#333333',
               linewidth=0.8, height=0.7)

ax.set_yticks(y_pos)
ax.set_yticklabels([r[:35] + '...' if len(r) > 35 else r for r in top_roles.index], fontsize=14)
ax.set_xlabel('Frequency', fontsize=18)
ax.invert_yaxis()

max_val = top_roles.values.max()
for bar, val in zip(bars, top_roles.values):
    ax.text(val + max_val*0.02, bar.get_y() + bar.get_height()/2,
            str(val), va='center', fontsize=13, fontweight='bold', color='#333333')

ax.set_xlim(0, max_val * 1.15)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.xaxis.grid(True, linestyle='--', alpha=0.4, color='gray')
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'fig_contribution_roles.png'), bbox_inches='tight', facecolor='white',dpi=600)
plt.show()
print('Figure saved.')


# ── Detailed tool breakdown per role ──────────────────────────────────────────
print('DETAILED TOOL BREAKDOWN BY CONTRIBUTION ROLE (TOP 5 TOOLS PER ROLE):')
print('=' * 70)
for role in role_counts.head(15).index:
    print(f'\n{role}:')
    breakdown  = roles_df[roles_df['role'] == role]['tool'].value_counts().head(5)
    role_total = role_counts[role]
    for tool, cnt in breakdown.items():
        pct = cnt / role_total * 100
        print(f'  • {tool}: {cnt} ({pct:.1f}%)')


# ──────────────────────────────────────────────────────────────────────────
# ## 7. Category-Level Analysis
# ──────────────────────────────────────────────────────────────────────────


# ── Category aggregation ───────────────────────────────────────────────────────
roles_df['category'] = roles_df['role'].map(category_mapping).fillna('Other')
cat_counts = roles_df['category'].value_counts()
print('ROLE CATEGORIES:')
print('-' * 50)
for cat, cnt in cat_counts.items():
    pct = cnt / total_count * 100
    print(f'  {cat:<25} {cnt:4} ({pct:.1f}%)')

# ── Figure: Category pie chart ────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 7))
colors_cat = plt.cm.Set2(np.linspace(0, 1, len(cat_counts)))

wedges, texts, autotexts = ax.pie(
    cat_counts.values,
    labels=None,                  # ← remove inline labels
    autopct='%1.1f%%',
    colors=colors_cat,
    startangle=90,
    pctdistance=0.75,
    wedgeprops={'edgecolor': 'white', 'linewidth': 1.5},
)
for autotext in autotexts:
    autotext.set_fontsize(11)
    autotext.set_fontweight('bold')

# Build legend labels: "Category name (N, X%)"
total = cat_counts.sum()
legend_labels = [
    f"{cat}  ({cnt}, {cnt/total*100:.1f}%)"
    for cat, cnt in cat_counts.items()
]
ax.legend(
    wedges,
    legend_labels,
    title="Categories",
    title_fontsize=12,
    fontsize=11,
    loc="center left",
    bbox_to_anchor=(1.02, 0.5),   # place legend to the right of the pie
    frameon=True,
    framealpha=0.9,
)

plt.tight_layout()
plt.savefig(
    os.path.join(output_folder, 'fig_role_categories_pie.png'),
    bbox_inches='tight',
    facecolor='white',
    dpi=600
)
plt.show()
print('Figure saved.')


# ──────────────────────────────────────────────────────────────────────────
# ## 8. Tool × Contribution Role Heatmap
# ──────────────────────────────────────────────────────────────────────────


top_tools_list = df_ai['tool'].value_counts().head(10).index.tolist()
top_roles_list = role_counts.head(10).index.tolist()

heatmap_data = (
    roles_df[
        roles_df['tool'].isin(top_tools_list) &
        roles_df['role'].isin(top_roles_list)
    ]
    .groupby(['tool', 'role'])
    .size()
    .unstack(fill_value=0)
    .reindex(index=top_tools_list, columns=top_roles_list, fill_value=0)
)

fig, ax = plt.subplots(figsize=(12, 6))
im = ax.imshow(heatmap_data.values, cmap='Blues', aspect='auto')

ax.set_xticks(range(len(top_roles_list)))
ax.set_xticklabels([r[:20] + '...' if len(r) > 20 else r for r in top_roles_list],
                   rotation=40, ha='right', fontsize=11)
ax.set_yticks(range(len(top_tools_list)))
ax.set_yticklabels(top_tools_list, fontsize=12)

for i in range(len(top_tools_list)):
    for j in range(len(top_roles_list)):
        val = heatmap_data.values[i, j]
        if val > 0:
            ax.text(j, i, str(val), ha='center', va='center',
                    fontsize=10, fontweight='bold',
                    color='white' if val > heatmap_data.values.max() * 0.6 else '#333333')

plt.colorbar(im, ax=ax, label='Co-occurrence Count')
ax.set_xlabel('Contribution Role', fontsize=14)
ax.set_ylabel('AI Tool', fontsize=14)

plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'fig_tool_role_heatmap.png'), bbox_inches='tight', facecolor='white',dpi=600)
plt.show()
print('Figure saved.')
