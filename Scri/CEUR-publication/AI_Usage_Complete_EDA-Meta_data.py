
import matplotlib.pyplot as plt
print(plt.style.available)


# ──────────────────────────────────────────────────────────────────────────
# ## 1. Setup and Data Preparation
# ──────────────────────────────────────────────────────────────────────────


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


output_dir=r"path/to/output/directory"


fpath=(r"path/to/your/file.csv")

df_meta=pd.read_csv(fpath)
df_meta


df_meta.columns


df_meta = df_meta.rename(columns={
    'index_id':    'paper_id',      
    'paper_id':    'volume_id',     
    
    
})
df_meta


# ──────────────────────────────────────────────────────────────────────────
# # AI_data
# ──────────────────────────────────────────────────────────────────────────



fpath=(r"path/to/your/file.csv")


df_tools_filtered=pd.read_csv(fpath)
df_tools_filtered




import pandas as pd
import os

# ── Reset index & rename columns to correct convention ───────────────────────

# Rename to clear, correct convention
df_tools_filtered = df_tools_filtered.rename(columns={
    'index_id':    'paper_id',      # actual paper identifier
    'paper_id':    'volume_id',     # CEUR-WS volume (Vol-XXXX)
    'AI-tool':     'tool',          # AI tool used
    'Tool-usage':  'contribution_roles',  # how the tool was used
})
df_tools_filtered


df_tools_filtered.columns


# do with event-date
print("Dataset Overview:")
print("="*50)
print(f"Shape: {df_tools_filtered.shape}")
print(f"Columns: {', '.join(df_tools_filtered.columns)}")

# Convert event_date to datetime and extract temporal features
# First, parse the month-year format (e.g., "November-2024")
df_tools_filtered['event_date_parsed'] = pd.to_datetime(df_tools_filtered['event_date'], format='%B-%Y')
df_tools_filtered['year'] = df_tools_filtered['event_date_parsed'].dt.year
df_tools_filtered['month'] = df_tools_filtered['event_date_parsed'].dt.month
df_tools_filtered['year_month'] = df_tools_filtered['event_date_parsed'].dt.to_period('M')

# Create binary AI usage indicator
df_tools_filtered['ai_used'] = ~df_tools_filtered['tool'].isin(['No AI tool'])

# Count number of roles
df_tools_filtered['num_roles'] = df_tools_filtered['contribution_roles'].apply(
    lambda x: 0 if x == 'No contribution mentioned' else len(x.split(','))
)


# print("\n✓ Features added: year, ai_used, num_roles, tool_category")
print("\nFirst 5 rows:")
df_tools_filtered


df_meta


df_meta.columns


df_meta=df_meta[['paper_id','institution', 'country',  'city','department']]
df_meta


df_meta['country'].isnull().sum()


df_meta.columns


institute_country = {
    "University of Wuppertal": "Germany",
    "Maria Sklodowska-Curie Warsaw Higher School": "Poland",
    "National University of Life and Environmental Sciences of Ukraine": "Ukraine",
    "University of North Bengal": "India",
    "Leiden University": "Netherlands",
    "Utrecht University": "Netherlands",
    "National Police of Ukraine": "Ukraine",
    "Sapienza University of Rome": "Italy",
    "Sony": "Japan",
    "ENS Paris-Saclay": "France",
    "Amazon": "United States",
    "Norwegian Labour and Welfare Administration": "Norway",
    "University of Pisa": "Italy",
    "Institute of Informatics and Telematics": "Italy",
    "Cy4gate S.p.A.": "Italy",
    "Italian Navy": "Italy",
    "University of Rome": "Italy",
    "University of Cagliari": "Italy",
    "Institut de Recherche en Informatique de Toulouse (IRIT)": "France",
    "CNIT National Network Assurance and Monitoring Lab": "Italy",
    "Politecnico di Milano": "Italy",
    "Micron Technology": "United States",
    "CS Group": "France",
    "Toulouse INP": "France",
    "Munich University of Applied Sciences": "Germany",
    "Lviv Polytechnic National University": "Ukraine",
    "Unilever": "United Kingdom/Netherlands",
    "Universidad Politécnica de Madrid": "Spain",
    "King's College London": "United Kingdom",
    "ASI - Agenzia Spaziale Italiana": "Italy",
    "University of Bari": "Italy",
    "Universidad Carlos III de Madrid": "Spain",
    "University of Hertfordshire": "United Kingdom",
    "City University of Hong Kong": "Hong Kong",
    "National and Kapodistrian University of Athens": "Greece",
    "Robert Gordon University": "United Kingdom",
    "Carnegie Mellon University": "United States",
    "Learning Engineering Virtual Institute": "United States",
    "Datapreneur Services Pvt. Ltd.": "India",
    "WinLighter": "France",
    "The Education University of Hong Kong": "Hong Kong",
    "University of California, Merced": "United States",
    "Gardner-Webb University": "United States",
    "Arizona State University": "United States",
    "Holon Institute of Technology": "Israel",
    "University of Hamburg": "Germany",
    "Nicolaus Copernicus University": "Poland",
    "Paris-Saclay University": "France",
    "Institute of Diabetology": "Germany",
    "University of Applied Sciences of the Grisons": "Switzerland",
    "Royal College of Surgeons in Ireland": "Ireland",
    "University of Bonn": "Germany",
    "Karlsruhe Institute of Technology": "Germany",
    "University of Udine": "Italy",
    "University of Calabria": "Italy",
    "University of Freiburg": "Germany",
    "University of Michigan": "United States",
    "Aalto University": "Finland",
    "University of Helsinki": "Finland",
    "Huygens Institute": "Netherlands",
    "National Technical University KhPI": "Ukraine",
    "Kharkiv Polytechnic Institute": "Ukraine",
    "Kyiv National University of Trade and Economics": "Ukraine",
    "University of Siegen": "Germany",
    "National Technical University of Athens": "Greece",
    "University of Washington": "United States",
    "University of Sarajevo": "Bosnia and Herzegovina",
    "Polytechnic University of Bari": "Italy",
    "University of Milano-Bicocca": "Italy",
    "North Carolina State University": "United States",
    "University of Massachusetts Amherst": "United States",
    "University of Pittsburgh": "United States",
    "University of Memphis": "United States",
    "University of Florida": "United States",
    "University of Angers": "France",
    "Leipzig University": "Germany",
    "University of Kassel": "Germany",
    "University of Brescia": "Italy",
    "University of Perugia": "Italy",
    "University of Turin": "Italy",
    "Politecnico di Torino": "Italy",
    "University of Milan": "Italy",
    "University of Catania": "Italy",
    "University Mediterranea of Reggio Calabria": "Italy",
    "Ada University": "Azerbaijan",
    "University of Padua": "Italy",
    "University of Potsdam": "Germany",
    "National Technical University of Ukraine 'Igor Sikorsky Kyiv Polytechnic Institute'": "Ukraine",
    "Linnaeus University": "Sweden",
    "Blekinge Institute of Technology": "Sweden",
    "University of Skövde": "Sweden",
    "TCS Research": "India",
    "Indian Institute of Science Education and Research Kolkata": "India",
    "Georgia Institute of Technology": "United States",
    "Ben-Gurion University of the Negev": "Israel",
    "Harbin Institute of Technology": "China",
    "Universidad de Jaén": "Spain",
    "Universitat de Barcelona": "Spain",
    "Vellore Institute of Technology": "India",
    "University of California, San Diego": "United States",
    "KLE Technological University": "India",
    "National Institute of Metrology, Technology and Quality": "Brazil",
    "Xidian University": "China",
    "University of Innsbruck": "Austria",
    "University of Applied Sciences BFI Vienna": "Austria",
    "Tarbiat Modares University": "Iran",
    "Sharif University of Technology": "Iran",
    "University of Tehran": "Iran",
    "University of the Basque Country": "Spain",
    "GESIS - Leibniz Institute for the Social Sciences": "Germany",
    "University of Tübingen": "Germany",
    "Bauhaus-Universität Weimar": "Germany",
    "Friedrich Schiller University Jena": "Germany",
    "University of Amsterdam": "Netherlands",
    "Jožef Stefan Institute": "Slovenia",
    "Charles University": "Czech Republic",
    "Kaunas University of Technology": "Lithuania",
    "Cornell University": "United States",
    "University of Bologna": "Italy",
    "Tallinn University of Technology": "Estonia",
    "University of Southern Denmark": "Denmark",
    "University of Florence": "Italy",
    "IMT School for Advanced Studies Lucca": "Italy",
    "Scuola Normale Superiore di Pisa": "Italy",
    "University of Salerno": "Italy",
    "Amsterdam University of Applied Sciences": "Netherlands",
    "Waseda University": "Japan",
    "University of Galway": "Ireland",
    "Technische Universität Dresden": "Germany",
    "Polytechnique Montréal": "Canada",
    "Ithaca College": "United States",
    "Edinburgh Napier University": "United Kingdom",
    "Ghent University": "Belgium",
    "University of Geneva": "Switzerland",
    "University of the Bundeswehr Munich": "Germany",
    "The Hong Kong Polytechnic University": "Hong Kong",
    "Wroclaw University": "Poland",
    "Technical University of Munich": "Germany",
    "National Institute of Technology Patna": "India",
    "Birla Institute of Technology and Science, Pilani": "India",
    "IT University of Copenhagen": "Denmark",
    "Jadavpur University": "India",
    "Indian Institute of Technology Madras": "India",
    "Indian Institute of Technology Kharagpur": "India",
    "Vrije Universiteit Amsterdam": "Netherlands",
    "Centrum Wiskunde en Informatica": "Netherlands",
    "University of Augsburg": "Germany",
    "University of Minnesota": "United States",
    "Drexel University": "United States",
    "Clemson University": "United States",
    "Trinity College Dublin": "Ireland",
    "Czech Technical University in Prague": "Czech Republic",
    "Fraunhofer FOKUS": "Germany",
    "Old Dominion University": "United States",
    "The Pennsylvania State University": "United States",
    "University of Naples Federico II": "Italy",
    "Universitat Autònoma de Barcelona": "Spain",
    "Vienna University of Technology": "Austria",
    "University of Western Australia": "Australia",
    "Barcelona Supercomputing Center": "Spain",
    "Universitat Politècnica de Catalunya": "Spain",
    "University of Edinburgh": "United Kingdom",
    "University of Maribor": "Slovenia",
    "Slovak University of Technology in Bratislava": "Slovakia",
    "University of Hagen": "Germany",
    "University of Luxembourg": "Luxembourg",
    "University of Oxford": "United Kingdom",
    "Rensselaer Polytechnic Institute": "United States",
    "The University of Tokyo": "Japan",
    "IBM Research": "United States",
    "Pontificia Universidad Católica de Chile": "Chile",
    "European University Institute": "Italy",
    "Northwestern University": "United States",
    "Northeastern University": "United States",
    "Queen Mary University of London": "United Kingdom",
    "Johannes Kepler University": "Austria",
    "Swarthmore College": "United States",
    "Pomona College": "United States",
    "New York University": "United States",
    "University of Oslo": "Norway",
    "Comenius University": "Slovakia",
    "Pasteur Institute": "France",
    "Bloomberg": "United States",
    "Brigham Young University": "United States",
    "Instituto Politécnico Nacional": "Mexico",
    "CIMAT": "Mexico",
    "Federal University of Ceará": "Brazil",
    "NOVA University of Lisbon": "Portugal",
    "University of Alicante": "Spain",
    "University of Southern California": "United States",
    "Syracuse University": "United States",
    "National Research Council of Italy": "Italy",
    "University of Passau": "Germany",
    "University of Chicago": "United States",
    "University of Mannheim": "Germany",
    "University of Trento": "Italy",
    "Graz University of Technology": "Austria",
    "University of Trieste": "Italy",
    "Singapore Management University": "Singapore",
    "National University Singapore": "Singapore",
    "Simon Fraser University": "Canada",
    "Swansea University": "United Kingdom",
    "University of Genoa": "Italy",
    "University of Zaragoza": "Spain",
    "University College London (UCL)": "United Kingdom",
    "Tel Aviv University": "Israel",
    "Boston University": "United States",
    "University of Barcelona": "Spain",
    "University of Groningen": "Netherlands",
    "Eötvös Loránd University": "Hungary",
    "Université Grenoble Alpes": "France",
    "University of Manchester": "United Kingdom",
    "University of Coimbra": "Portugal",
    "Hubei University of Technology": "China",
    "University of Pennsylvania": "United States",
    "Indian Institute of Technology Bombay": "India",
    "University of California, Irvine": "United States",
    "Stanford University": "United States",
    "Nanyang Technological University": "Singapore",
}


# Fill country where it is NaN using institution → country mapping
df_meta['country'] = df_meta['country'].fillna(
    df_meta['institution'].map(institute_country)
)


df_meta['country'].isnull().sum()


institutes_country_null = df_meta.loc[df_meta['country'].isnull(), 'institution'].dropna().unique().tolist()
institutes_country_null


missing_map = {
    "SER&Practices": "Italy",
    "Datapreneur Services Pvt. Ltd": "India",
    "Faculty Science Limited": "United Kingdom",
    "Fondazione Clément Fillietroz ONLUS": "Italy",
    "ESP": "France",  
    "NCIT": "Nrpal",  
    "Fusemachine": "United States",
    "Logictronix Technologies": "Nepal",
    "Pune Institute of Computer Technology": "India",
    "Ecole Nationale d'Ingénieurs de Brest": "France",
    "Université Paul Sabatier": "France",
    "Workifi": "Denmark",
    "Tifin": "United States",
    "Askmyfi": "United States",
    "IIIT Kottayam": "India",
    "University G. d'Annunzio of Chieti-Pescara": "Italy",
    "Institute for Research in Fundamental Sciences (IPM)": "Iran",
    "VNU University of Engineering and Technology": "Vietnam",
    "Indeed.com": "United States",
    "University Center of Defense": "Spain",
    "Amadeus": "Spain",
    "Sri Sivasubramaniya Nadar College of Engineering": "India",
    "Birla Institute of Technology and Science": "India",
    "Motilal Nehru National Institute of Technology Allahabad": "India",
    "Meenakshi Sundararajan Engineering College": "India",
    "Kongu Engineering College": "India",
    "St.Joseph's College of Engineering": "India",
    "Universidad Michoacana de San Nicolas de Hidalgo": "Mexico",
    "CONAHCyT-INFOTEC": "Mexico",
    "Haldia Institute of Technology": "India",
    "National Institute of Technology, Agartala": "India",
    "Institute of Engineering & Management, Kolkata": "India",
    "Indian Institute of Engineering Science and Technology": "India",
    "RCC Institute of Information Technology": "India",
    "Institute of Engineering and Management": "India",
    "Indian Institute of Information Technology Ranchi": "India",
    "Kalinga Institute of Industrial Technology": "India",
    "Indian Institute of Technology Goa": "India",
    "Indian Association of Cultivation of Science": "India",
    "Indian Institute of Technology": "India",
    "Institute of Telecommunications and Global Information Space": "Ukraine",
    "Central Research Institute of the Armed Forces of Ukraine": "Ukraine",
    "Bosch Center for AI": "Germany",
    "University of Colorado": "United States",
    "Graphwise": "United States",
    "Center for International Forestry Research": "Indonesia",
    "Austrian Center for Digital Humanities and Cultural Heritage": "Austria",
    "5T SRL": "Italy",
    "Siemens AG Österreich": "Austria",
    "ZBMED – Information Centre for Life Sciences": "Germany",
    "Artificial Intelligence Research Institute (IIIA - CSIC)": "Spain",
    "Hensun Innovation, LLC": "United States",
    "ServiceNow, Inc.": "United States",
    "Kirklareli University": "Turkey",
    "University of Camerino": "Italy",
    "Vienna University of Economics and Business": "Austria",
    "University of Messina": "Italy",
    "University of Tsukuba": "Japan",
    "Middle East College Oman": "Oman",
    "Thales": "France",
    "Paris Descartes University": "France",
    "University of Paris": "France",
    "University of St. Gallen": "Switzerland",
    "Infomart Corporation": "Japan",
    "Database Center for Life Science": "Japan",
    "USC Information Sciences Institute": "United States",
    "Osaka Electro-Communication University": "Japan",
    "Smart Information Flow Technologies": "United States",
    "Pace University": "United States",
    "Northeastern": "United States",
    "Technische Universität Dortmund": "Germany",
    "University of Opole": "Poland",
    "The Czech Academy of Sciences": "Czech Republic",
    "Instituto Federal do Triângulo Mineiro": "Brazil",
    "CICESE": "Mexico",
    "University Carlos III of Madrid (UC3M)": "Spain",
    "Instituto Nacional de Astrofísica, Óptica y Electrónica (INAOE)": "Mexico",
    "UTP": 'Malaysia',  
    "MetaMetrics Inc.": "United States",
    "Università degli Studi di Camerino": "Italy",
    "Università Ca' Foscari di Venezia": "Italy",
    "Zuyd University of Applied Sciences": "Netherlands",
    "IBM Research Brazil": "Brazil",
    "Universidad de Chile": "Chile",
    "Universidad Loyola": "Spain",
    "Università di Roma La Sapienza": "Italy",
    "Free University of Bozen-Bolzano": "Italy",
    "Istituto di Linguistica Computazionale \"A. Zampolli\" (CNR-ILC)": "Italy",
    "Università degli Studi di Pavia": "Italy",
    "Università Cattolica del Sacro Cuore": "Italy",
    "Università degli Studi di Roma \"Tor Vergata\"": "Italy",
    "LINKS Foundation": "Italy",
    "University School for Advanced Studies IUSS Pavia": "Italy",
    "IUSS Pavia": "Italy",
    "Ca' Foscari University of Venice": "Italy",
    "University of Verona": "Italy",
    "Brandenburg University of Technology Cottbus-Senftenberg": "Germany",
    "University of the Basque Country - UPV/EHU": "Spain",
    "Orai NLP Technologies": "Vietnam",
    "Université Toulouse-Jean Jaurès": "France",
    "Lo Congrès permanent de la lenga occitana": "France",
    "Université Perpignan Via Domitia": "France",
    "University of Lleida (UdL)": "Spain",
    "University Parthenope of Naples": "Italy",
    "Sant'Anna School of Advanced Studies": "Italy",
    "INTELLIGENZA AUMENTATA SRL": "Italy",
    "Shenzhen University": "China",
    "University of Rome, Tor Vergata": "Italy",
    "University of L'Aquila": "Italy",
    "Mälardalen University": "Sweden",
    "University of Bremen": "Germany",
    "eBay": "United States",
    "Coupang": "South Korea",
    "Etsy": "United States",
    "eBay Inc.": "United States",
    "University of Naples Parthenope": "Italy",
    "Ternopil Ivan Pului National Technical University": "Ukraine",
    "Deutsches Zentrum fur Luft- and Raumfahrt e.V.": "Germany",
    "Fraunhofer-Institute for open Communication Systems": "Germany",
    "University of South-Eastern Norway": "Norway",
    "University of Gothenburg": "Sweden",
    "University College, Copenhagen": "Denmark",
    "Gothenburg University": "Sweden",
    "Technical University of Denmark (DTU)": "Denmark",
    "DXC Technology": "United States",
    "zerodivision": "Germany",
    "AMADEUS France": "France",
    "Combitech": "Sweden",
    "Digital Safety CIC": "United Kingdom",
    "University of Modena and Reggio Emilia": "Italy",
    "Berger-Levrault": "France",
    "La Sapienza University of Rome": "Italy",
    "Airbus CyberSecurity": "France",
    "Università degli Studi di Milano Bicocca": "Italy",
    "University Iba Der Thiam of Thiès": "Senegal",
    "Uzhnu": "Ukraine",
    "Lntu": "Ukraine",
    "National Institute for Nuclear Physics (INFN)": "Italy",
    "University of Murcia": "Spain",
    "Universität Bielefeld": "Germany",
    "Università Politecnica delle Marche": "Italy",
    "Università Parthenope": "Italy",
    "Università Guglielmo Marconi": "Italy",
    "Sapienza, University of Rome": "Italy",
    "Flexify.AI": "Germany",
    "Bern University of Applied Sciences": "Switzerland",
    "University of Zurich": "Switzerland",
    "ExpertAI-Lux S.à r.l": "Luxembourg",
    "FBK": "Italy",
    "Rostock University": "Germany",
    "Maynooth University": "Ireland",
    "University of Lisbon": "Portugal",
    "Wageningen University & Research": "Netherlands",
    "Fraunhofer": "Germany",
    "Polytechnic University of Milan": "Italy",
    "Hamburg University of Applied Sciences (HAW Hamburg)": "Germany",
    "Mangalore University": "India",
    "Birla Institute of Technology and Science Pilani": "India",
    "Indian Institute of Information Technology Surat": "India",
    "Indian Institute of Technology Gandhinggar": "India",
    "National Institute of Technology": "India",
    "Sikkim Manipal Institute of Technology": "India",
    "National Institute of Techonology Karnataka": "India",
    "Huawei": "China",
    "H-Partners": "Switzerland",
    "Taipei Medical University": "Taiwan",
    "National Taiwan University": "Taiwan",
    "Medical University of Graz": "Austria",
    "Accenture": "Ireland",
    "Federal University of Ceará (UFC)": "Brazil",
    "Pontifical Catholic University of Rio de Janeiro": "Brazil",
    "University of Jyväskylä": "Finland",
    "Polytechnic University of Marche": "Italy",
    "Aristotle University of Thessaloniki": "Greece",
    "Albatross AI": "France",
    "Tokyo Gakugei University": "Japan",
    "Lossfunk": "Germany",
    "Ashoka University": "India",
    "Poznan University of Technology": "Poland",
    "Ministry of the Economy and Finance": "Italy",
    "Sogei": "Italy",
    "IBM": "United States",
    "Lyon 1 University": "France",
    "EPITA": "France",
    "INSA Lyon": "France",
}



df_meta['country'] = df_meta['country'].fillna(
    df_meta['institution'].map(missing_map)
)
df_meta


df_meta['institution'].isnull().sum()


institutes_country_null = df_meta.loc[df_meta['country'].isnull(), 'institution'].dropna().unique().tolist()
institutes_country_null


missing_map_1= {
     "The University of Western Australia": "Australia"}


df_meta['country'] = df_meta['country'].fillna(
    df_meta['institution'].map(missing_map_1)
)
df_meta

df_meta['country'].isnull().sum()


institutes_country_null = df_meta.loc[df_meta['country'].isnull(), 'institution'].dropna().unique().tolist()
institutes_country_null


df_meta


df_meta['country'].isnull().sum()


# Rows where country is NaN — show all columns to diagnose
df_meta[df_meta['country'].isnull()][['institution', 'country']].value_counts(dropna=False)


df_meta.columns


df_meta_filtered = df_meta.drop_duplicates()
df_meta_filtered.reset_index(drop=True, inplace=True)
df_meta


df_merged1 = pd.merge(df_tools_filtered, df_meta_filtered, on='paper_id', how='left')
df_merged1


df_merged = df_merged1.dropna(subset=['country', 'institution']).reset_index(drop=True)
df_merged


df_merged['country'].isnull().sum(), df_merged['institution'].isnull().sum()


df_merged['institution'].nunique(), df_merged['country'].nunique()


df_merged['institution'].value_counts()


count_unique = df_merged[
    (df_merged['institution'] == "Lviv Polytechnic National University") &
    (df_merged['ai_used'] == True)
]['paper_id'].nunique()

print(count_unique)

count_unique = df_merged[
    (df_merged['institution'] == "Lviv Polytechnic National University") &
    (df_merged['ai_used'] == False)
]['paper_id'].nunique()

print(count_unique)


count_unique = df_merged[
    (df_merged['institution'] == "Lviv Polytechnic National University") &
    (df_merged['ai_used'] == False)
]['paper_id'].nunique()

print(count_unique)


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

# ============================================================
# PROFESSIONAL STYLE CONFIGURATION FOR RESEARCH PAPERS
# ============================================================
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 18,
    'axes.labelsize': 20,
    'axes.titlesize': 20,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
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

# ============================================================
# SETUP AND DATA LOADING
# ============================================================

# Create output directory
output_dir = r"path/to/output/directory"
os.makedirs(output_dir, exist_ok=True)

# Merge the dataframes for comprehensive analysis
# NOTE: Make sure df_tools_filtered and df_meta_filtered1 are loaded before running this code


print("="*70)
print("DATASET OVERVIEW")
print("="*70)
print(f"Merged dataset shape: {df_merged.shape}")
print(f"Total unique papers: {df_merged['paper_id'].nunique()}")

# ============================================================
# CRITICAL FIX: Proper Paper-Level Aggregation
# ============================================================
# Problem: df_merged has multiple rows per paper (one per AI tool used)
# Solution: Aggregate properly at paper level for each analysis dimension

# Create paper-level dataset for temporal analysis
paper_level_temporal = df_merged.groupby('paper_id').agg({
    'ai_used': 'max',  # If ANY row has ai_used=True, the paper uses AI
    'year_month': 'first'
}).reset_index()

print(f"Papers using AI: {paper_level_temporal['ai_used'].sum()}")
print(f"Papers not using AI: {(~paper_level_temporal['ai_used']).sum()}")
print(f"Overall AI usage rate: {paper_level_temporal['ai_used'].sum() / len(paper_level_temporal) * 100:.1f}%")

# ============================================================
# ANALYSIS 1: Country-wise AI Usage Distribution
# ============================================================
print("\n" + "="*70)
print("ANALYSIS 1: Country-wise AI Usage Distribution")
print("="*70)

# For country analysis: papers can have multiple countries
# We need to count unique papers per country while respecting AI usage
country_paper_level = df_merged[['paper_id', 'country', 'ai_used']].drop_duplicates(['paper_id', 'country'])

# For each country-paper combination, use max of ai_used (handles any duplicate rows)
country_paper_agg = country_paper_level.groupby(['country', 'paper_id'])['ai_used'].max().reset_index()

# Now count papers by country
country_ai_counts = country_paper_agg.groupby('country').agg({
    'paper_id': 'count',
    'ai_used': 'sum'
}).reset_index()
country_ai_counts.columns = ['country', 'total_papers', 'ai_papers']
country_ai_counts['ai_usage_rate'] = (country_ai_counts['ai_papers'] / country_ai_counts['total_papers'] * 100).round(1)

print(f"\nTotal countries represented: {len(country_ai_counts)}")
print(f"\nTop 5 countries by total papers:")
print(country_ai_counts.nlargest(5, 'total_papers')[['country', 'total_papers', 'ai_papers', 'ai_usage_rate']].to_string(index=False))

# Verification with Ukraine
ukraine_data = country_ai_counts[country_ai_counts['country'] == 'Ukraine']
if len(ukraine_data) > 0:
    print(f"\n✓ VERIFICATION - Ukraine:")
    print(f"  Total papers: {int(ukraine_data['total_papers'].values[0])}")
    print(f"  AI papers: {int(ukraine_data['ai_papers'].values[0])}")
    print(f"  Non-AI papers: {int(ukraine_data['total_papers'].values[0] - ukraine_data['ai_papers'].values[0])}")
    print(f"  AI usage rate: {ukraine_data['ai_usage_rate'].values[0]:.1f}%")

# Get top 20 countries by AI papers
top_countries = country_ai_counts.nlargest(20, 'ai_papers')

# ============================================================
# FIGURE 1A: TOP 20 COUNTRIES BY AI PAPER COUNT
# ============================================================
fig1a, ax1a = plt.subplots(figsize=(14, 10), dpi=300)

colors_blue_gradient = plt.cm.Blues(np.linspace(0.4, 0.85, len(top_countries)))
y_pos = np.arange(len(top_countries))

bars = ax1a.barh(y_pos, top_countries['ai_papers'], 
                 color=colors_blue_gradient,
                 edgecolor='#333333', 
                 linewidth=0.7,
                 height=0.75)

ax1a.set_yticks(y_pos)
ax1a.set_yticklabels(top_countries['country'], fontsize=25)
ax1a.set_xlabel('Number of Papers Using AI', fontsize=25)
ax1a.invert_yaxis()

# Add value labels (AI papers / Total papers)
max_val = top_countries['ai_papers'].max()
for bar, ai_count, total in zip(bars, top_countries['ai_papers'], top_countries['total_papers']):
    width = bar.get_width()
    ax1a.text(width + max_val*0.02, bar.get_y() + bar.get_height()/2, 
              f'{int(ai_count)}/{int(total)}', 
              ha='left', va='center', fontsize=20, fontweight='bold', color='#333333')

# Set x-axis limit to accommodate labels
ax1a.set_xlim(0, max_val * 1.25)

# Styling
ax1a.spines['top'].set_visible(False)
ax1a.spines['right'].set_visible(False)
ax1a.spines['left'].set_linewidth(1.0)
ax1a.spines['bottom'].set_linewidth(1.0)
ax1a.xaxis.grid(True, linestyle='--', alpha=0.4, color='gray')
ax1a.set_axisbelow(True)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'fig1a_country_ai_count.pdf'), bbox_inches='tight', facecolor='white')
plt.show()

print(f"\n✓ Figure 1A saved: Top 20 Countries by AI Paper Count")

# ============================================================
# FIGURE 1B: TOP 20 COUNTRIES BY AI USAGE RATE
# ============================================================

# Get top 20 countries by AI usage rate (minimum 10 papers)
top_countries_rate = country_ai_counts[country_ai_counts['total_papers'] >= 10].nlargest(20, 'ai_usage_rate')

fig1b, ax1b = plt.subplots(figsize=(10, 9), dpi=300)

# Create light green to blue fading gradient
n_bars = len(top_countries_rate)
colors_green_blue_gradient = []
for i in range(n_bars):
    # Interpolate from light green to blue
    ratio = i / (n_bars - 1) if n_bars > 1 else 0
    r = (144 * (1 - ratio) + 46 * ratio) / 255   # 144->46 (green to blue red channel)
    g = (238 * (1 - ratio) + 90 * ratio) / 255   # 238->90 (green to blue green channel)
    b = (144 * (1 - ratio) + 135 * ratio) / 255  # 144->135 (green to blue blue channel)
    colors_green_blue_gradient.append((r, g, b))

y_pos = np.arange(len(top_countries_rate))

bars = ax1b.barh(y_pos, top_countries_rate['ai_usage_rate'], 
                 color=colors_green_blue_gradient,
                 edgecolor='#333333', 
                 linewidth=0.7,
                 height=0.75)

ax1b.set_yticks(y_pos)
ax1b.set_yticklabels(top_countries_rate['country'], fontsize=25)
ax1b.set_xlabel('AI Usage Rate (%)', fontsize=25)
ax1b.invert_yaxis()

# Add value labels
max_val = top_countries_rate['ai_usage_rate'].max()
for bar, rate in zip(bars, top_countries_rate['ai_usage_rate']):
    width = bar.get_width()
    ax1b.text(width + max_val*0.02, bar.get_y() + bar.get_height()/2, 
              f'{rate:.1f}%', 
              ha='left', va='center', fontsize=20, fontweight='bold', color='#333333')

# Set x-axis limit
ax1b.set_xlim(0, max_val * 1.2)

# Styling
ax1b.spines['top'].set_visible(False)
ax1b.spines['right'].set_visible(False)
ax1b.spines['left'].set_linewidth(1.0)
ax1b.spines['bottom'].set_linewidth(1.0)
ax1b.xaxis.grid(True, linestyle='--', alpha=0.4, color='gray')
ax1b.set_axisbelow(True)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'fig1b_country_ai_rate.pdf'), bbox_inches='tight', facecolor='white')
plt.show()

print(f"✓ Figure 1B saved: Top 20 Countries by AI Usage Rate (Min. 10 Papers)")

# Print insights
print(f"\nCountry Analysis Insights:")
print(f"  - Total countries represented: {country_ai_counts['country'].nunique()}")
print(f"  - Top country by AI papers: {top_countries.iloc[0]['country']} ({int(top_countries.iloc[0]['ai_papers'])} papers)")
print(f"  - Top country by AI rate: {top_countries_rate.iloc[0]['country']} ({top_countries_rate.iloc[0]['ai_usage_rate']:.1f}%)")


# ============================================================
# EXTRACT COUNTRIES WITH MINIMUM 10 PAPERS - COMPREHENSIVE TABLE
# ============================================================

print("\n" + "="*70)
print("COUNTRIES WITH MINIMUM 10 PAPERS - DETAILED TABLE")
print("="*70)

# Filter countries with at least 10 papers
countries_min_10 = country_ai_counts[country_ai_counts['total_papers'] >= 50].copy()

# Calculate non-AI papers
countries_min_10['non_ai_papers'] = countries_min_10['total_papers'] - countries_min_10['ai_papers']

# Reorder columns for better readability
countries_table = countries_min_10[['country', 'ai_papers', 'non_ai_papers', 'total_papers', 'ai_usage_rate']].copy()

# Rename columns for clarity
countries_table.columns = ['Country', 'Papers Using AI', 'Non-AI Papers', 'Total Papers', 'AI Usage Rate (%)']

# Sort by AI usage rate (descending)
countries_table = countries_table.sort_values('AI Usage Rate (%)', ascending=False).reset_index(drop=True)

# Display the table
print(f"\nTotal countries with ≥50 papers: {len(countries_table)}")
print("\n" + countries_table.to_string(index=False))

# Save to CSV
output_file = os.path.join(output_dir, 'countries_min10_papers_detailed.csv')
countries_table.to_csv(output_file, index=False)
print(f"\n✓ Table saved to: {output_file}")

# Summary statistics
print("\n" + "="*70)
print("SUMMARY STATISTICS FOR COUNTRIES WITH ≥50 PAPERS")
print("="*70)
print(f"Total countries: {len(countries_table)}")
print(f"Total papers across these countries: {countries_table['Total Papers'].sum()}")
print(f"Total AI papers: {countries_table['Papers Using AI'].sum()}")
print(f"Total Non-AI papers: {countries_table['Non-AI Papers'].sum()}")
print(f"Average AI usage rate: {countries_table['AI Usage Rate (%)'].mean():.1f}%")
print(f"Median AI usage rate: {countries_table['AI Usage Rate (%)'].median():.1f}%")
print(f"Highest AI usage rate: {countries_table['AI Usage Rate (%)'].max():.1f}% ({countries_table.iloc[0]['Country']})")
print(f"Lowest AI usage rate: {countries_table['AI Usage Rate (%)'].min():.1f}% ({countries_table.iloc[-1]['Country']})")


# ============================================================
# ANALYSIS 2: Institutional AI Usage Patterns
# ============================================================
print("\n" + "="*70)
print("ANALYSIS 2: Institutional AI Usage Patterns")
print("="*70)

# Similar approach for institutions: papers can have multiple institutions
institution_paper_level = df_merged[['paper_id', 'institution', 'ai_used']].drop_duplicates(['paper_id', 'institution'])

# For each institution-paper combination, use max of ai_used
institution_paper_agg = institution_paper_level.groupby(['institution', 'paper_id'])['ai_used'].max().reset_index()

# Count papers by institution
institution_ai_counts = institution_paper_agg.groupby('institution').agg({
    'paper_id': 'count',
    'ai_used': 'sum'
}).reset_index()
institution_ai_counts.columns = ['institution', 'total_papers', 'ai_papers']
institution_ai_counts['ai_usage_rate'] = (institution_ai_counts['ai_papers'] / institution_ai_counts['total_papers'] * 100).round(1)

print(f"\nTotal institutions represented: {len(institution_ai_counts)}")

# Verification with Lviv Polytechnic
lviv_data = institution_ai_counts[institution_ai_counts['institution'] == 'Lviv Polytechnic National University']
if len(lviv_data) > 0:
    print(f"\n✓ VERIFICATION - Lviv Polytechnic National University:")
    print(f"  Total papers: {int(lviv_data['total_papers'].values[0])}")
    print(f"  AI papers: {int(lviv_data['ai_papers'].values[0])}")
    print(f"  Non-AI papers: {int(lviv_data['total_papers'].values[0] - lviv_data['ai_papers'].values[0])}")
    print(f"  AI usage rate: {lviv_data['ai_usage_rate'].values[0]:.1f}%")

# Get top 10 institutions by AI papers (reduced from 15 for better readability)
top_institutions = institution_ai_counts.nlargest(10, 'ai_papers')

# Truncate long institution names for better display
top_institutions['institution_short'] = top_institutions['institution'].apply(
    lambda x: x[:45] + '...' if len(x) > 45 else x
)

# ============================================================
# FIGURE 2: TOP 10 INSTITUTIONS BY AI USAGE (SINGLE COLUMN)
# ============================================================
# ACM single column width: ~3.33 inches, double column: ~7 inches
# For readable single-column figure in ACM format
fig2, ax2 = plt.subplots(figsize=(8.2, 5), dpi=300)

# Create blue to coral/orange fading gradient
n_bars = len(top_institutions)
colors_blue_coral_gradient = []
for i in range(n_bars):
    # Interpolate from blue to coral
    ratio = i / (n_bars - 1) if n_bars > 1 else 0
    r = (46 * (1 - ratio) + 240 * ratio) / 255    # 46->240 (blue to coral red channel)
    g = (90 * (1 - ratio) + 128 * ratio) / 255    # 90->128 (blue to coral green channel)
    b = (135 * (1 - ratio) + 128 * ratio) / 255   # 135->128 (blue to coral blue channel)
    colors_blue_coral_gradient.append((r, g, b))

y_pos = np.arange(len(top_institutions))

bars = ax2.barh(y_pos, top_institutions['ai_papers'], 
                color=colors_blue_coral_gradient,
                edgecolor='#333333', 
                linewidth=0.5,
                height=0.7)

ax2.set_yticks(y_pos)
ax2.set_yticklabels(top_institutions['institution_short'], fontsize=17)
ax2.set_xlabel('Number of Papers Using AI', fontsize=17, fontweight='bold')
ax2.invert_yaxis()
ax2.tick_params(axis='x', labelsize=17)
ax2.tick_params(axis='y', labelsize=17)

# Add value labels (count and percentage) - smaller and inside bars when possible
max_val = top_institutions['ai_papers'].max()
for bar, ai_count, rate in zip(bars, top_institutions['ai_papers'], top_institutions['ai_usage_rate']):
    width = bar.get_width()
    label = f'{int(ai_count)} ({rate:.1f}%)'
    
    # Place label inside bar if there's enough space, otherwise outside
    if width > max_val * 0.3:
        # Inside the bar (right-aligned)
        ax2.text(width - max_val*0.02, bar.get_y() + bar.get_height()/2, 
                 label, 
                 ha='right', va='center', fontsize=10.5, fontweight='bold', color='white')
    else:
        # Outside the bar
        ax2.text(width + max_val*0.02, bar.get_y() + bar.get_height()/2, 
                 label, 
                 ha='left', va='center', fontsize=10.5, fontweight='bold', color='#333333')

# Set x-axis limit
ax2.set_xlim(0, max_val * 1.25)

# Styling
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['left'].set_linewidth(1.0)
ax2.spines['bottom'].set_linewidth(1.0)
ax2.xaxis.grid(True, linestyle='--', alpha=0.3, color='gray', linewidth=0.5)
ax2.set_axisbelow(True)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'fig2_institution_ai_usage.pdf'), bbox_inches='tight', facecolor='white')
plt.show()

print(f"✓ Figure 2 saved: Top 10 Institutions by AI Usage")
print(f"\nInstitution Analysis Insights:")
print(f"  - Total institutions represented: {institution_ai_counts['institution'].nunique()}")
print(f"  - Top institution: {top_institutions.iloc[0]['institution'][:50]} ({int(top_institutions.iloc[0]['ai_papers'])} papers)")


paper_level_temporal


paper_level_temporal['year_month'].value_counts()


# ============================================================
print("\n" + "="*70)
print("ANALYSIS 3: Temporal Trends in AI Usage")
print("="*70)

# Use paper_level_temporal for temporal analysis (already aggregated correctly)
temporal_data = paper_level_temporal.groupby('year_month').agg({
    'paper_id': 'count',
    'ai_used': 'sum'
}).reset_index()
temporal_data.columns = ['year_month', 'total_papers', 'ai_papers']
temporal_data['ai_usage_rate'] = (temporal_data['ai_papers'] / temporal_data['total_papers'] * 100).round(1)

# Convert year_month to datetime for proper plotting
temporal_data['date'] = pd.to_datetime(temporal_data['year_month'].astype(str) + '-01')
temporal_data = temporal_data.sort_values('date')


temporal_data['date'].value_counts()


# ============================================================
# ANALYSIS 3: TEMPORAL ANALYSIS OF AI USAGE
# ============================================================
print("\n" + "="*70)
print("ANALYSIS 3: Temporal Trends in AI Usage")
print("="*70)

# Use paper_level_temporal for temporal analysis (already aggregated correctly)
temporal_data = paper_level_temporal.groupby('year_month').agg({
    'paper_id': 'count',
    'ai_used': 'sum'
}).reset_index()

temporal_data.columns = ['year_month', 'total_papers', 'ai_papers']
temporal_data['ai_usage_rate'] = (temporal_data['ai_papers'] / temporal_data['total_papers'] * 100).round(1)

# Convert year_month to datetime for proper plotting
temporal_data['date'] = pd.to_datetime(temporal_data['year_month'].astype(str) + '-01')
temporal_data = temporal_data.sort_values('date')

# ============================================================
# FIGURE 3A: TEMPORAL TREND (ABSOLUTE NUMBERS)
# ============================================================
fig3a, ax3a = plt.subplots(figsize=(12, 6), dpi=300)

ax3a.plot(temporal_data['date'], temporal_data['total_papers'], 
         marker='o', linewidth=2.5, markersize=8, 
         label='Total Papers', color='#2E5A87', markeredgecolor='#333333', markeredgewidth=0.5)
ax3a.plot(temporal_data['date'], temporal_data['ai_papers'], 
         marker='s', linewidth=2.5, markersize=8, 
         label='Papers Using AI', color='#C73E1D', markeredgecolor='#333333', markeredgewidth=0.5)

ax3a.set_ylabel('Number of Papers', fontsize=30)
ax3a.set_xlabel('Time Period (Year-Month)', fontsize=30)
ax3a.legend(fontsize=30, frameon=True, fancybox=False, shadow=False, 
           edgecolor='#333333', loc='upper left')

# Format x-axis
ax3a.tick_params(axis='x', rotation=45,labelsize=25)
import matplotlib.dates as mdates
ax3a.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax3a.xaxis.set_major_locator(mdates.MonthLocator(interval=1))

# Set x-axis limits to match actual data range (removes extra months)
ax3a.set_xlim(temporal_data['date'].min(), temporal_data['date'].max())

# Add value annotations
for date, total, ai in zip(temporal_data['date'], temporal_data['total_papers'], temporal_data['ai_papers']):
    ax3a.annotate(f'{ai}', (date, ai), textcoords="offset points", xytext=(0,8), 
                ha='center', fontsize=16, fontweight='bold', color='#C73E1D')
    ax3a.annotate(f'{total}', (date, total), textcoords="offset points", xytext=(0,8), 
                ha='center', fontsize=16, fontweight='bold', color='#2E5A87')

# Styling
ax3a.spines['top'].set_visible(False)
ax3a.spines['right'].set_visible(False)
ax3a.spines['left'].set_linewidth(1.0)
ax3a.spines['bottom'].set_linewidth(1.0)
ax3a.grid(True, linestyle='--', alpha=0.4, color='gray')
ax3a.set_axisbelow(True)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'fig3a_temporal_absolute.pdf'), bbox_inches='tight', facecolor='white')
plt.show()

print(f"✓ Figure 3A saved: Temporal Trend - Paper Volume and AI Usage")

# ============================================================
# FIGURE 3B: TEMPORAL TREND (AI USAGE RATE)
# ============================================================
fig3b, ax3b = plt.subplots(figsize=(12, 6), dpi=300)

ax3b.plot(temporal_data['date'], temporal_data['ai_usage_rate'], 
         marker='D', linewidth=2.5, markersize=8, 
         color='#F18F01', markeredgecolor='#333333', markeredgewidth=0.5)

ax3b.set_ylabel('AI Usage Rate (%)', fontsize=25)
ax3b.set_xlabel('Time Period (Year-Month)', fontsize=25)

# Format x-axis
ax3b.tick_params(axis='x', rotation=45)
ax3b.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax3b.xaxis.set_major_locator(mdates.MonthLocator(interval=1))

# Set x-axis limits to match actual data range (removes extra months)
ax3b.set_xlim(temporal_data['date'].min(), temporal_data['date'].max())

# Add value annotations
for date, rate in zip(temporal_data['date'], temporal_data['ai_usage_rate']):
    ax3b.annotate(f'{rate:.1f}%', (date, rate), 
                textcoords="offset points", xytext=(0,10), 
                ha='center', fontsize=18, fontweight='bold', color='#333333')

# Styling
ax3b.spines['top'].set_visible(False)
ax3b.spines['right'].set_visible(False)
ax3b.spines['left'].set_linewidth(1.0)
ax3b.spines['bottom'].set_linewidth(1.0)
ax3b.grid(True, linestyle='--', alpha=0.4, color='gray')
ax3b.set_axisbelow(True)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'fig3b_temporal_rate.pdf'), bbox_inches='tight', facecolor='white')
plt.show()

print(f"✓ Figure 3B saved: Temporal Trend - AI Usage Rate")

# Print temporal insights
earliest_rate = temporal_data['ai_usage_rate'].iloc[0]
latest_rate = temporal_data['ai_usage_rate'].iloc[-1]

print(f"\nTemporal Analysis Insights:")
print(f"  - Time period: {temporal_data['year_month'].iloc[0]} to {temporal_data['year_month'].iloc[-1]}")
print(f"  - Number of time periods: {len(temporal_data)}")
print(f"  - AI usage rate at start: {earliest_rate:.1f}%")
print(f"  - AI usage rate at end: {latest_rate:.1f}%")

if len(temporal_data) > 1:
    growth_rate = ((latest_rate - earliest_rate) / earliest_rate * 100)
    print(f"  - Growth in AI usage rate: {growth_rate:+.1f}%")

peak_period = temporal_data.loc[temporal_data['ai_usage_rate'].idxmax()]
print(f"  - Peak AI usage: {peak_period['year_month']} ({peak_period['ai_usage_rate']:.1f}%)")

# ============================================================
# GEOGRAPHIC INSIGHTS
# ============================================================
print("\n" + "="*70)
print("ANALYSIS 4: Geographic Patterns in AI Adoption")
print("="*70)

country_summary = country_ai_counts[country_ai_counts['total_papers'] >= 5].copy()

print("\nCountries with highest AI usage rates (≥50%):")
high_ai_countries = country_summary[country_summary['ai_usage_rate'] >= 50].sort_values('ai_usage_rate', ascending=False)
if len(high_ai_countries) > 0:
    print(high_ai_countries[['country', 'total_papers', 'ai_papers', 'ai_usage_rate']].head(10).to_string(index=False))
else:
    print("  No countries with ≥50% AI usage rate")

print("\nCountries with lowest AI usage rates (<50%):")
low_ai_countries = country_summary[country_summary['ai_usage_rate'] < 50].sort_values('ai_usage_rate')
if len(low_ai_countries) > 0:
    print(low_ai_countries[['country', 'total_papers', 'ai_papers', 'ai_usage_rate']].head(10).to_string(index=False))

# ============================================================
# SAVE SUMMARY STATISTICS
# ============================================================
print("\n" + "="*70)
print("SAVING SUMMARY STATISTICS")
print("="*70)

summary_stats = {
    'Metric': [
        'Total Papers', 
        'Papers Using AI', 
        'AI Usage Rate', 
        'Countries Represented', 
        'Institutions Represented',
        'Time Period Start', 
        'Time Period End',
        'AI Usage Start', 
        'AI Usage End'
    ],
    'Value': [
        len(paper_level_temporal), 
        int(paper_level_temporal['ai_used'].sum()), 
        f"{paper_level_temporal['ai_used'].sum()/len(paper_level_temporal)*100:.1f}%",
        country_ai_counts['country'].nunique(), 
        institution_ai_counts['institution'].nunique(),
        str(temporal_data['year_month'].min()), 
        str(temporal_data['year_month'].max()),
        f"{earliest_rate:.1f}%", 
        f"{latest_rate:.1f}%"
    ]
}

# Save all summary files
pd.DataFrame(summary_stats).to_csv(os.path.join(output_dir, 'summary_statistics.csv'), index=False)
country_ai_counts.to_csv(os.path.join(output_dir, 'country_ai_statistics.csv'), index=False)
institution_ai_counts.to_csv(os.path.join(output_dir, 'institution_ai_statistics.csv'), index=False)
temporal_data.to_csv(os.path.join(output_dir, 'temporal_trends.csv'), index=False)

print(f"\n✓ Summary statistics saved")
print(f"✓ Country statistics saved")
print(f"✓ Institution statistics saved")
print(f"✓ Temporal trends saved")

print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)
print(f"\nAll analysis files and visualizations saved to:")
print(f"  {output_dir}")
print("\nGenerated figures:")
print("  - fig1a_country_ai_count.pdf")
print("  - fig1b_country_ai_rate.pdf")
print("  - fig2_institution_ai_usage.pdf")
print("  - fig3a_temporal_absolute.pdf")
print("  - fig3b_temporal_rate.pdf")


df_merged


# Top tool and role by insitute
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

# ============================================================
# PROFESSIONAL STYLE CONFIGURATION FOR RESEARCH PAPERS
# ============================================================
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 16,
    'axes.labelsize': 20,
    'axes.titlesize': 20,
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

# ============================================================
# SETUP AND DATA LOADING
# ============================================================

# Create output directory
output_dir = r"path_to_output_directory"  # <-- UPDATE THIS PATH
os.makedirs(output_dir, exist_ok=True)

# NOTE: Make sure df_tools_filtered and df_meta_filtered1 are loaded before running this code
# df_merged = pd.merge(df_tools_filtered, df_meta_filtered1, on='paper_id', how='left')

print("="*80)
print("INSTITUTIONAL AI TOOL AND CONTRIBUTION ROLE ANALYSIS")
print("="*80)
print(f"Total rows in merged dataset: {len(df_merged)}")
print(f"Total unique papers: {df_merged['paper_id'].nunique()}")
print(f"Total institutions: {df_merged['institution'].nunique()}")
print(f"Total unique tools: {df_merged['tool'].nunique()}")

# ============================================================
# ANALYSIS 1: TOP TOOLS BY INSTITUTION
# ============================================================
print("\n" + "="*80)
print("ANALYSIS 1: Top AI Tools by Institution")
print("="*80)

# Get institutions with significant AI usage (minimum threshold)
MIN_PAPERS_THRESHOLD = 20  # Institutions with at least 20 papers

# Count papers per institution
institution_paper_counts = df_merged.groupby('institution')['paper_id'].nunique().reset_index()
institution_paper_counts.columns = ['institution', 'paper_count']
top_institutions_list = institution_paper_counts[
    institution_paper_counts['paper_count'] >= MIN_PAPERS_THRESHOLD
].sort_values('paper_count', ascending=False)['institution'].tolist()

print(f"\nAnalyzing top {len(top_institutions_list)} institutions with ≥{MIN_PAPERS_THRESHOLD} papers")
print(f"Top 10 institutions by paper count:")
print(institution_paper_counts.nlargest(10, 'paper_count').to_string(index=False))

# For each institution, get top 5 tools
institution_tool_analysis = []

for inst in top_institutions_list[:15]:  # Analyze top 15 institutions
    # Filter data for this institution
    inst_data = df_merged[df_merged['institution'] == inst]
    
    # Count tool usage (by unique papers using each tool)
    tool_counts = inst_data.groupby('tool')['paper_id'].nunique().reset_index()
    tool_counts.columns = ['tool', 'paper_count']
    tool_counts = tool_counts.sort_values('paper_count', ascending=False)
    
    # Get top 5 tools
    top_5_tools = tool_counts.head(5)
    
    # Calculate percentage
    total_papers = inst_data['paper_id'].nunique()
    
    for idx, row in top_5_tools.iterrows():
        institution_tool_analysis.append({
            'institution': inst,
            'tool': row['tool'],
            'paper_count': row['paper_count'],
            'total_papers': total_papers,
            'percentage': (row['paper_count'] / total_papers * 100)
        })

df_inst_tools = pd.DataFrame(institution_tool_analysis)

# ============================================================
# FIGURE 1: TOP TOOLS FOR TOP 10 INSTITUTIONS (HEATMAP)
# ============================================================
print("\nCreating heatmap of top tools by institution...")

# Select top 10 institutions by total papers
top_10_insts = top_institutions_list[:10]

# Get top 10 most used tools overall
top_tools_overall = df_merged.groupby('tool')['paper_id'].nunique().nlargest(10).index.tolist()

# Create matrix for heatmap
heatmap_data = []
for inst in top_10_insts:
    inst_row = []
    inst_data = df_merged[df_merged['institution'] == inst]
    total_papers = inst_data['paper_id'].nunique()
    
    for tool in top_tools_overall:
        tool_papers = inst_data[inst_data['tool'] == tool]['paper_id'].nunique()
        percentage = (tool_papers / total_papers * 100) if total_papers > 0 else 0
        inst_row.append(percentage)
    
    heatmap_data.append(inst_row)

# Truncate long institution names
inst_labels = [inst[:45] + '...' if len(inst) > 45 else inst for inst in top_10_insts]

# Create heatmap
fig1, ax1 = plt.subplots(figsize=(14, 10), dpi=300)

im = ax1.imshow(heatmap_data, cmap='Blues', aspect='auto', vmin=0, vmax=50)

# Set ticks and labels
ax1.set_xticks(np.arange(len(top_tools_overall)))
ax1.set_yticks(np.arange(len(top_10_insts)))
ax1.set_xticklabels(top_tools_overall, rotation=45, ha='right', fontsize=16)
ax1.set_yticklabels(inst_labels, fontsize=14)

# Add colorbar
cbar = plt.colorbar(im, ax=ax1)
cbar.set_label('Usage Rate (%)', rotation=270, labelpad=25, fontsize=18)
cbar.ax.tick_params(labelsize=16)

# Add percentage values on heatmap
for i in range(len(top_10_insts)):
    for j in range(len(top_tools_overall)):
        value = heatmap_data[i][j]
        if value > 0:
            text_color = 'white' if value > 25 else 'black'
            text = ax1.text(j, i, f'{value:.1f}', ha='center', va='center',
                          color=text_color, fontsize=11, fontweight='bold')

ax1.set_xlabel('AI Tool', fontsize=20, fontweight='bold')
ax1.set_ylabel('Institution', fontsize=20, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'fig1_institution_tool_heatmap.pdf'), 
            bbox_inches='tight', facecolor='white')
plt.show()

print("✓ Figure 1 saved: Institution-Tool Usage Heatmap")

# ============================================================
# ANALYSIS 2: TOP CONTRIBUTION ROLES BY INSTITUTION
# ============================================================
print("\n" + "="*80)
print("ANALYSIS 2: Top Contribution Roles by Institution")
print("="*80)

# Check if contribution_role column exists
if 'contribution_role' in df_merged.columns:
    
    # Overall top contribution roles
    top_roles_overall = df_merged.groupby('contribution_role')['paper_id'].nunique().nlargest(8).index.tolist()
    print(f"\nTop contribution roles overall:")
    role_counts = df_merged.groupby('contribution_role')['paper_id'].nunique().sort_values(ascending=False)
    print(role_counts.head(8))
    
    # For each institution, analyze contribution roles
    institution_role_analysis = []
    
    for inst in top_institutions_list[:15]:  # Analyze top 15 institutions
        inst_data = df_merged[df_merged['institution'] == inst]
        
        # Count role usage
        role_counts = inst_data.groupby('contribution_role')['paper_id'].nunique().reset_index()
        role_counts.columns = ['contribution_role', 'paper_count']
        role_counts = role_counts.sort_values('paper_count', ascending=False)
        
        # Get top 3 roles
        top_3_roles = role_counts.head(3)
        
        total_papers = inst_data['paper_id'].nunique()
        
        for idx, row in top_3_roles.iterrows():
            institution_role_analysis.append({
                'institution': inst,
                'contribution_role': row['contribution_role'],
                'paper_count': row['paper_count'],
                'total_papers': total_papers,
                'percentage': (row['paper_count'] / total_papers * 100)
            })
    
    df_inst_roles = pd.DataFrame(institution_role_analysis)
    
    # ============================================================
    # FIGURE 2: TOP CONTRIBUTION ROLES FOR TOP 10 INSTITUTIONS (HEATMAP)
    # ============================================================
    print("\nCreating heatmap of contribution roles by institution...")
    
    # Create matrix for heatmap
    heatmap_data_roles = []
    for inst in top_10_insts:
        inst_row = []
        inst_data = df_merged[df_merged['institution'] == inst]
        total_papers = inst_data['paper_id'].nunique()
        
        for role in top_roles_overall:
            role_papers = inst_data[inst_data['contribution_role'] == role]['paper_id'].nunique()
            percentage = (role_papers / total_papers * 100) if total_papers > 0 else 0
            inst_row.append(percentage)
        
        heatmap_data_roles.append(inst_row)
    
    # Truncate long role names for display
    role_labels = [role[:40] + '...' if len(role) > 40 else role for role in top_roles_overall]
    
    # Create heatmap
    fig2, ax2 = plt.subplots(figsize=(14, 10), dpi=300)
    
    im2 = ax2.imshow(heatmap_data_roles, cmap='Greens', aspect='auto', vmin=0, vmax=50)
    
    # Set ticks and labels
    ax2.set_xticks(np.arange(len(top_roles_overall)))
    ax2.set_yticks(np.arange(len(top_10_insts)))
    ax2.set_xticklabels(role_labels, rotation=45, ha='right', fontsize=16)
    ax2.set_yticklabels(inst_labels, fontsize=14)
    
    # Add colorbar
    cbar2 = plt.colorbar(im2, ax=ax2)
    cbar2.set_label('Usage Rate (%)', rotation=270, labelpad=25, fontsize=18)
    cbar2.ax.tick_params(labelsize=16)
    
    # Add percentage values
    for i in range(len(top_10_insts)):
        for j in range(len(top_roles_overall)):
            value = heatmap_data_roles[i][j]
            if value > 0:
                text_color = 'white' if value > 25 else 'black'
                text = ax2.text(j, i, f'{value:.1f}', ha='center', va='center',
                              color=text_color, fontsize=11, fontweight='bold')
    
    ax2.set_xlabel('Contribution Role', fontsize=20, fontweight='bold')
    ax2.set_ylabel('Institution', fontsize=20, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig2_institution_role_heatmap.pdf'), 
                bbox_inches='tight', facecolor='white')
    plt.show()
    
    print("✓ Figure 2 saved: Institution-Contribution Role Heatmap")

else:
    print("\n⚠ Warning: 'contribution_role' column not found in dataset")
    print("Skipping contribution role analysis")

# ============================================================
# FIGURE 3: TOP 5 TOOLS FOR EACH TOP 10 INSTITUTIONS (BAR CHARTS)
# ============================================================
print("\n" + "="*80)
print("Creating detailed tool usage charts for top institutions...")
print("="*80)

# Create subplots for top 10 institutions
fig3, axes = plt.subplots(5, 2, figsize=(18, 24), dpi=300)
axes = axes.flatten()

for idx, inst in enumerate(top_10_insts):
    ax = axes[idx]
    
    # Get top 5 tools for this institution
    inst_tools = df_inst_tools[df_inst_tools['institution'] == inst].nlargest(5, 'paper_count')
    
    # Create color gradient
    colors = plt.cm.Blues(np.linspace(0.4, 0.85, len(inst_tools)))
    
    # Create horizontal bar chart
    y_pos = np.arange(len(inst_tools))
    bars = ax.barh(y_pos, inst_tools['percentage'], color=colors, 
                   edgecolor='#333333', linewidth=0.7, height=0.7)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(inst_tools['tool'], fontsize=14)
    ax.set_xlabel('Usage Rate (%)', fontsize=16)
    
    # Truncate institution name for title
    inst_short = inst[:50] + '...' if len(inst) > 50 else inst
    ax.set_title(f'{inst_short}\n({inst_tools["total_papers"].iloc[0]} papers)', 
                fontsize=14, fontweight='bold', pad=10)
    
    ax.invert_yaxis()
    
    # Add value labels
    max_val = inst_tools['percentage'].max()
    for bar, pct, count in zip(bars, inst_tools['percentage'], inst_tools['paper_count']):
        width = bar.get_width()
        ax.text(width + max_val*0.02, bar.get_y() + bar.get_height()/2,
               f'{pct:.1f}% ({int(count)})', ha='left', va='center',
               fontsize=11, fontweight='bold')
    
    # Styling
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim(0, max_val * 1.35)
    ax.grid(True, axis='x', linestyle='--', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'fig3_institution_top5_tools.pdf'), 
            bbox_inches='tight', facecolor='white')
plt.show()

print("✓ Figure 3 saved: Top 5 Tools per Institution")

# ============================================================
# DETAILED TEXT REPORTS
# ============================================================
print("\n" + "="*80)
print("GENERATING DETAILED REPORTS")
print("="*80)

# Report 1: Top tools for each top 15 institutions
print("\n" + "-"*80)
print("REPORT 1: Top 5 AI Tools by Institution")
print("-"*80)

for inst in top_institutions_list[:15]:
    inst_tools = df_inst_tools[df_inst_tools['institution'] == inst].nlargest(5, 'paper_count')
    total = inst_tools['total_papers'].iloc[0] if len(inst_tools) > 0 else 0
    
    print(f"\n{inst}")
    print(f"Total papers: {total}")
    print("Top 5 tools:")
    for idx, row in inst_tools.iterrows():
        print(f"  {row['tool']:20s} : {int(row['paper_count']):4d} papers ({row['percentage']:.1f}%)")

# Report 2: Top contribution roles for each institution
if 'contribution_role' in df_merged.columns:
    print("\n" + "-"*80)
    print("REPORT 2: Top 3 Contribution Roles by Institution")
    print("-"*80)
    
    for inst in top_institutions_list[:15]:
        inst_roles = df_inst_roles[df_inst_roles['institution'] == inst].nlargest(3, 'paper_count')
        total = inst_roles['total_papers'].iloc[0] if len(inst_roles) > 0 else 0
        
        print(f"\n{inst}")
        print(f"Total papers: {total}")
        print("Top 3 contribution roles:")
        for idx, row in inst_roles.iterrows():
            print(f"  {row['contribution_role']:40s} : {int(row['paper_count']):4d} papers ({row['percentage']:.1f}%)")

# ============================================================
# SAVE DATA TO CSV
# ============================================================
print("\n" + "="*80)
print("SAVING DATA FILES")
print("="*80)

# Save institution-tool analysis
df_inst_tools.to_csv(os.path.join(output_dir, 'institution_tool_analysis.csv'), index=False)
print("✓ Saved: institution_tool_analysis.csv")

# Save institution-role analysis
if 'contribution_role' in df_merged.columns:
    df_inst_roles.to_csv(os.path.join(output_dir, 'institution_role_analysis.csv'), index=False)
    print("✓ Saved: institution_role_analysis.csv")

# Save summary statistics
summary_data = []
for inst in top_institutions_list[:15]:
    inst_data = df_merged[df_merged['institution'] == inst]
    total_papers = inst_data['paper_id'].nunique()
    total_tools = inst_data['tool'].nunique()
    
    # Get most common tool
    top_tool = inst_data.groupby('tool')['paper_id'].nunique().idxmax()
    top_tool_count = inst_data.groupby('tool')['paper_id'].nunique().max()
    
    summary_data.append({
        'institution': inst,
        'total_papers': total_papers,
        'unique_tools_used': total_tools,
        'most_common_tool': top_tool,
        'most_common_tool_papers': top_tool_count,
        'most_common_tool_percentage': (top_tool_count / total_papers * 100)
    })

df_summary = pd.DataFrame(summary_data)
df_summary.to_csv(os.path.join(output_dir, 'institution_summary.csv'), index=False)
print("✓ Saved: institution_summary.csv")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
print(f"\nAll files saved to: {output_dir}")
print("\nGenerated files:")
print("  Figures:")
print("    - fig1_institution_tool_heatmap.pdf")
print("    - fig2_institution_role_heatmap.pdf")
print("    - fig3_institution_top5_tools.pdf")
print("  Data:")
print("    - institution_tool_analysis.csv")
print("    - institution_role_analysis.csv")
print("    - institution_summary.csv")


# all 28 countries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ============================================================
# COUNTRY SELECTION - All 28 countries from the table
# ============================================================
countries = [
    "Norway", "India", "Brazil", "Japan", "Sweden",
    "Austria", "China", "Belgium", "United States", "Spain",
    "Switzerland", "Algeria", "Germany", "Netherlands",
    "Mexico", "Slovakia", "Portugal", "France", "Finland",
    "United Kingdom", "Czech Republic", "Ireland", "Italy",
    "Canada", "Poland", "Ukraine", "Greece", "Kazakhstan"
]

country_abbreviations = {
    "Norway": "NOR", "India": "IND", "Brazil": "BRA",
    "Japan": "JPN", "Sweden": "SWE", "Austria": "AUT",
    "China": "CHN", "Belgium": "BEL", "United States": "USA",
    "Spain": "ESP", "Switzerland": "CHE", "Algeria": "DZA",
    "Germany": "DEU", "Netherlands": "NLD", "Mexico": "MEX",
    "Slovakia": "SVK", "Portugal": "PRT", "France": "FRA",
    "Finland": "FIN", "United Kingdom": "GBR", "Czech Republic": "CZE",
    "Ireland": "IRL", "Italy": "ITA", "Canada": "CAN",
    "Poland": "POL", "Ukraine": "UKR", "Greece": "GRC",
    "Kazakhstan": "KAZ"
}

# ============================================================
# PROFESSIONAL STYLE CONFIGURATION
# ============================================================
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.titlesize': 16,
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
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

# ============================================================
# SETUP
# ============================================================
output_dir = r"path_to_output_directory"  # <-- UPDATE THIS PATH
os.makedirs(output_dir, exist_ok=True)

print("="*80)
print("TOP 10 TOOLS BY ALL 28 COUNTRIES - STACKED BAR CHART")
print("="*80)

# ============================================================
# FILTER DATA BY SELECTED COUNTRIES
# ============================================================
df_merged_filtered = df_merged[df_merged['country'].isin(countries)].copy()
print(f"\nFiltered to {len(df_merged_filtered)} rows from selected countries")

country_papers = df_merged_filtered[['paper_id', 'country']].drop_duplicates()
country_counts = country_papers['country'].value_counts()

# Order countries by AI usage rate (descending), matching the table
country_order_by_rate = [
    "Norway", "India", "Brazil", "Japan", "Sweden",
    "Austria", "China", "Belgium", "United States", "Spain",
    "Switzerland", "Algeria", "Germany", "Netherlands", "Mexico",
    "Slovakia", "Portugal", "France", "Finland", "United Kingdom",
    "Czech Republic", "Ireland", "Italy", "Canada", "Poland",
    "Ukraine", "Greece", "Kazakhstan"
]
selected_countries_ordered = [c for c in country_order_by_rate if c in country_counts.index]

print(f"\nSelected countries and paper counts:")
for i, country in enumerate(selected_countries_ordered, 1):
    count = country_counts.get(country, 0)
    print(f"  {i:2d}. {country:25s} : {count:4d} papers")

# ============================================================
# FILTER OUT AI (unspecified) and No AI tool
# ============================================================
df_filtered = df_merged_filtered[
    ~df_merged_filtered['tool'].isin(['No AI tool', 'AI (unspecified)'])
].copy()
print(f"\nRemoved {len(df_merged_filtered) - len(df_filtered)} rows with 'AI (unspecified)' or 'No AI tool'")

# ============================================================
# GET TOP 10 TOOLS OVERALL
# ============================================================
overall_tool_counts = df_filtered.groupby('tool')['paper_id'].nunique().sort_values(ascending=False)
top_10_tools = overall_tool_counts.head(10).index.tolist()

print(f"\nTop 10 tools overall (across all 28 countries):")
for i, tool in enumerate(top_10_tools, 1):
    print(f"  {i:2d}. {tool:25s} : {overall_tool_counts[tool]:4d} papers")

# ============================================================
# PREPARE DATA FOR VISUALIZATION
# ============================================================
plot_data = []
for country in selected_countries_ordered:
    country_data = df_filtered[df_filtered['country'] == country]
    for tool in top_10_tools:
        tool_papers = country_data[country_data['tool'] == tool]['paper_id'].nunique()
        plot_data.append({'country': country, 'tool': tool, 'papers': tool_papers})

df_plot = pd.DataFrame(plot_data)
df_pivot = df_plot.pivot(index='country', columns='tool', values='papers').fillna(0)
df_pivot = df_pivot.reindex(selected_countries_ordered)

# ============================================================
# CREATE STACKED BAR PLOT
# ============================================================
print("\nCreating stacked bar plot...")

# Wide figure to accommodate 28 countries
fig, ax = plt.subplots(figsize=(22, 9), dpi=300)

x = np.arange(len(selected_countries_ordered))
width = 0.65

colors = [
    '#2E5A87',  # Blue
    '#C73E1D',  # Red
    '#F18F01',  # Orange
    '#6A994E',  # Green
    '#BC4B51',  # Dark Pink
    '#8C4D9E',  # Purple
    '#FFCD00',  # Yellow
    '#00A6A6',  # Teal
    '#FF7EC9',  # Light Pink
    '#4B3621'   # Brown
]

bottom = np.zeros(len(selected_countries_ordered))

for i, tool in enumerate(top_10_tools):
    values = df_pivot[tool].values
    bars = ax.bar(x, values, width, label=tool, color=colors[i],
                  bottom=bottom, edgecolor='#333333', linewidth=0.5)

    # Value labels inside segments — threshold raised since bars are narrower
    for j, (bar, val) in enumerate(zip(bars, values)):
        if val > 30:
            y_position = bottom[j] + bar.get_height() / 2
            ax.text(bar.get_x() + bar.get_width() / 2., y_position,
                    f'{int(val)}', ha='center', va='center',
                    fontsize=7, fontweight='bold', color='white')

    bottom += values

# Totals on top of each bar
for j, (bar_x, total) in enumerate(zip(x, bottom)):
    ax.text(bar_x, total + 3, f'{int(total)}', ha='center', va='bottom',
            fontsize=8, fontweight='bold', color='#333333')

# X-axis abbreviations
country_abbrevs = [country_abbreviations[c] for c in selected_countries_ordered]
ax.set_xlabel('Country', fontsize=18, fontweight='bold', labelpad=10)
ax.set_ylabel('Number of Tool Instances', fontsize=18, fontweight='bold', labelpad=10)
ax.set_xticks(x)
ax.set_xticklabels(country_abbrevs, rotation=45, ha='right', fontsize=12)

# Legend — 2 columns outside plot to save horizontal space
ax.legend(loc='upper right', fontsize=11, frameon=True,
          edgecolor='#333333', fancybox=False, shadow=False, ncol=2)

# Styling
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(1.5)
ax.spines['bottom'].set_linewidth(1.5)
ax.yaxis.grid(True, linestyle='--', alpha=0.4, color='gray', linewidth=1)
ax.set_axisbelow(True)

max_val = bottom.max()
ax.set_ylim(0, max_val * 1.15)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'top10_tools_28countries_stacked.pdf'),
            bbox_inches='tight', facecolor='white')
print("✓ Figure saved: top10_tools_28countries_stacked.pdf")
plt.show()


# all 28 countries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ============================================================
# COUNTRY SELECTION - All 28 countries from the table
# ============================================================
countries = [
    "Norway", "India", "Brazil", "Japan", "Sweden",
    "Austria", "China", "Belgium", "United States", "Spain",
    "Switzerland", "Algeria", "Germany", "Netherlands",
    "Mexico", "Slovakia", "Portugal", "France", "Finland",
    "United Kingdom", "Czech Republic", "Ireland", "Italy",
    "Canada", "Poland", "Ukraine", "Greece", "Kazakhstan"
]

country_abbreviations = {
    "Norway": "NOR", "India": "IND", "Brazil": "BRA",
    "Japan": "JPN", "Sweden": "SWE", "Austria": "AUT",
    "China": "CHN", "Belgium": "BEL", "United States": "USA",
    "Spain": "ESP", "Switzerland": "CHE", "Algeria": "DZA",
    "Germany": "DEU", "Netherlands": "NLD", "Mexico": "MEX",
    "Slovakia": "SVK", "Portugal": "PRT", "France": "FRA",
    "Finland": "FIN", "United Kingdom": "GBR", "Czech Republic": "CZE",
    "Ireland": "IRL", "Italy": "ITA", "Canada": "CAN",
    "Poland": "POL", "Ukraine": "UKR", "Greece": "GRC",
    "Kazakhstan": "KAZ"
}

# ============================================================
# PROFESSIONAL STYLE CONFIGURATION
# ============================================================
# ============================================================
# PROFESSIONAL STYLE CONFIGURATION  (replace the whole block)
# ============================================================
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 34,
    'axes.labelsize': 38,
    'axes.titlesize': 38,
    'xtick.labelsize': 38,
    'ytick.labelsize': 38,
    'axes.linewidth': 1.5,
    'axes.edgecolor': '#333333',
    'axes.labelcolor': '#333333',
    'text.color': '#333333',
    'xtick.color': '#333333',
    'ytick.color': '#333333',
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.grid': False,
    'figure.dpi': 600,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight'
})
# ============================================================
# SETUP
# ============================================================
output_dir = r"path_to_output_directory"  # <-- UPDATE THIS PATH
os.makedirs(output_dir, exist_ok=True)

print("="*80)
print("TOP 10 TOOLS BY ALL 28 COUNTRIES - STACKED BAR CHART")
print("="*80)

# ============================================================
# FILTER DATA BY SELECTED COUNTRIES
# ============================================================
df_merged_filtered = df_merged[df_merged['country'].isin(countries)].copy()
print(f"\nFiltered to {len(df_merged_filtered)} rows from selected countries")

country_papers = df_merged_filtered[['paper_id', 'country']].drop_duplicates()
country_counts = country_papers['country'].value_counts()

eligible_countries = country_counts[country_counts >= 100].index.tolist()

# Order countries by AI usage rate (descending), matching the table
country_order_by_rate = [
    "Norway", "India", "Brazil", "Japan", "Sweden",
    "Austria", "China", "Belgium", "United States", "Spain",
    "Switzerland", "Algeria", "Germany", "Netherlands", "Mexico",
    "Slovakia", "Portugal", "France", "Finland", "United Kingdom",
    "Czech Republic", "Ireland", "Italy", "Canada", "Poland",
    "Ukraine", "Greece", "Kazakhstan"
]
selected_countries_ordered = [c for c in country_order_by_rate 
                               if c in eligible_countries]

print(f"\nSelected countries and paper counts:")
for i, country in enumerate(selected_countries_ordered, 1):
    count = country_counts.get(country, 0)
    print(f"  {i:2d}. {country:25s} : {count:4d} papers")

# ============================================================
# FILTER OUT AI (unspecified) and No AI tool
# ============================================================
df_filtered = df_merged_filtered[
    ~df_merged_filtered['tool'].isin(['No AI tool', 'AI (unspecified)'])
].copy()
print(f"\nRemoved {len(df_merged_filtered) - len(df_filtered)} rows with 'AI (unspecified)' or 'No AI tool'")


# ============================================================
# GET TOP 10 TOOLS OVERALL
# ============================================================
overall_tool_counts = df_filtered.groupby('tool')['paper_id'].nunique().sort_values(ascending=False)
top_10_tools = overall_tool_counts.head(10).index.tolist()

print(f"\nTop 10 tools overall (across all 28 countries):")
for i, tool in enumerate(top_10_tools, 1):
    print(f"  {i:2d}. {tool:25s} : {overall_tool_counts[tool]:4d} papers")

# ============================================================
# PREPARE DATA FOR VISUALIZATION
# ============================================================
plot_data = []
for country in selected_countries_ordered:
    country_data = df_filtered[df_filtered['country'] == country]
    for tool in top_10_tools:
        tool_papers = country_data[country_data['tool'] == tool]['paper_id'].nunique()
        plot_data.append({'country': country, 'tool': tool, 'papers': tool_papers})

df_plot = pd.DataFrame(plot_data)
df_pivot = df_plot.pivot(index='country', columns='tool', values='papers').fillna(0)
df_pivot = df_pivot.reindex(selected_countries_ordered)

# ============================================================
# CREATE STACKED BAR PLOT
# ============================================================
print("\nCreating stacked bar plot...")

fig, ax = plt.subplots(figsize=(22, 11), dpi=300)

x = np.arange(len(selected_countries_ordered))
width = 0.65

colors = [
    '#2E5A87', '#C73E1D', '#F18F01', '#6A994E', '#BC4B51',
    '#8C4D9E', '#FFCD00', '#00A6A6', '#FF7EC9', '#4B3621'
]

bottom = np.zeros(len(selected_countries_ordered))

for i, tool in enumerate(top_10_tools):
    values = df_pivot[tool].values
    bars = ax.bar(x, values, width, label=tool, color=colors[i],
                  bottom=bottom, edgecolor='#333333', linewidth=0.7)

    # ── numbers inside segments ──
    for j, (bar, val) in enumerate(zip(bars, values)):
        if val > 25:
            y_position = bottom[j] + bar.get_height() / 2
            ax.text(bar.get_x() + bar.get_width() / 2., y_position,
                    f'{int(val)}', ha='center', va='center',
                    fontsize=28, fontweight='bold', color='white')  # ← was 7

    bottom += values

# ── totals on top of bars ──
for j, (bar_x, total) in enumerate(zip(x, bottom)):
    ax.text(bar_x, total + 3, f'{int(total)}', ha='center', va='bottom',
            fontsize=32, fontweight='bold', color='#333333')  # ← was 8

# ── axis labels and ticks ──
country_abbrevs = [country_abbreviations[c] for c in selected_countries_ordered]
ax.set_xlabel('Country', fontsize=38, fontweight='bold', labelpad=12)
ax.set_ylabel('Number of Tool Instances', fontsize=38, fontweight='bold', labelpad=12)
ax.set_xticks(x)
ax.set_xticklabels(country_abbrevs, rotation=45, ha='right', fontsize=36)  # ← was 12

# ── legend: upper left, large ──
ax.legend(loc='upper left', fontsize=30, frameon=True,        # ← was upper right, 11
          edgecolor='#333333', fancybox=False, shadow=False,
          ncol=2)                                              # ← was ncol=2

# Styling
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(1.7)
ax.spines['bottom'].set_linewidth(1.7)
ax.yaxis.grid(True, linestyle='--', alpha=0.4, color='gray', linewidth=1)
ax.set_axisbelow(True)

max_val = bottom.max()
ax.set_ylim(0, max_val * 1.18)   # ← slightly more headroom for legend + totals

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'top10_tools_28countries_stacked.pdf'),
            bbox_inches='tight', facecolor='white')
print("✓ Figure saved: top10_tools_28countries_stacked.pdf")
plt.show()


# Combined all varient of chatgpt


# all 28 countries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ============================================================
# COUNTRY SELECTION - All 28 countries from the table
# ============================================================
countries = [
    "Norway", "India", "Brazil", "Japan", "Sweden",
    "Austria", "China", "Belgium", "United States", "Spain",
    "Switzerland", "Algeria", "Germany", "Netherlands",
    "Mexico", "Slovakia", "Portugal", "France", "Finland",
    "United Kingdom", "Czech Republic", "Ireland", "Italy",
    "Canada", "Poland", "Ukraine", "Greece", "Kazakhstan"
]

country_abbreviations = {
    "Norway": "NOR", "India": "IND", "Brazil": "BRA",
    "Japan": "JPN", "Sweden": "SWE", "Austria": "AUT",
    "China": "CHN", "Belgium": "BEL", "United States": "USA",
    "Spain": "ESP", "Switzerland": "CHE", "Algeria": "DZA",
    "Germany": "DEU", "Netherlands": "NLD", "Mexico": "MEX",
    "Slovakia": "SVK", "Portugal": "PRT", "France": "FRA",
    "Finland": "FIN", "United Kingdom": "GBR", "Czech Republic": "CZE",
    "Ireland": "IRL", "Italy": "ITA", "Canada": "CAN",
    "Poland": "POL", "Ukraine": "UKR", "Greece": "GRC",
    "Kazakhstan": "KAZ"
}

# ============================================================
# PROFESSIONAL STYLE CONFIGURATION
# ============================================================
# ============================================================
# PROFESSIONAL STYLE CONFIGURATION  (replace the whole block)
# ============================================================
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 34,
    'axes.labelsize': 38,
    'axes.titlesize': 38,
    'xtick.labelsize': 38,
    'ytick.labelsize': 38,
    'axes.linewidth': 1.5,
    'axes.edgecolor': '#333333',
    'axes.labelcolor': '#333333',
    'text.color': '#333333',
    'xtick.color': '#333333',
    'ytick.color': '#333333',
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.grid': False,
    'figure.dpi': 600,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight'
})
# ============================================================
# SETUP
# ============================================================
output_dir = r"path_to_output_directory"  # <-- UPDATE THIS PATH
os.makedirs(output_dir, exist_ok=True)

print("="*80)
print("TOP 10 TOOLS BY ALL 28 COUNTRIES - STACKED BAR CHART")
print("="*80)

# ============================================================
# FILTER DATA BY SELECTED COUNTRIES
# ============================================================
df_merged_filtered = df_merged[df_merged['country'].isin(countries)].copy()
print(f"\nFiltered to {len(df_merged_filtered)} rows from selected countries")

country_papers = df_merged_filtered[['paper_id', 'country']].drop_duplicates()
country_counts = country_papers['country'].value_counts()

eligible_countries = country_counts[country_counts >= 100].index.tolist()

# Order countries by AI usage rate (descending), matching the table
country_order_by_rate = [
    "Norway", "India", "Brazil", "Japan", "Sweden",
    "Austria", "China", "Belgium", "United States", "Spain",
    "Switzerland", "Algeria", "Germany", "Netherlands", "Mexico",
    "Slovakia", "Portugal", "France", "Finland", "United Kingdom",
    "Czech Republic", "Ireland", "Italy", "Canada", "Poland",
    "Ukraine", "Greece", "Kazakhstan"
]
selected_countries_ordered = [c for c in country_order_by_rate 
                               if c in eligible_countries]

print(f"\nSelected countries and paper counts:")
for i, country in enumerate(selected_countries_ordered, 1):
    count = country_counts.get(country, 0)
    print(f"  {i:2d}. {country:25s} : {count:4d} papers")

# ============================================================
# FILTER OUT AI (unspecified) and No AI tool
# ============================================================
df_filtered = df_merged_filtered[
    ~df_merged_filtered['tool'].isin(['No AI tool', 'AI (unspecified)'])
].copy()
print(f"\nRemoved {len(df_merged_filtered) - len(df_filtered)} rows with 'AI (unspecified)' or 'No AI tool'")

# ============================================================
# MERGE ALL CHATGPT VARIANTS INTO ONE
# ============================================================
chatgpt_variants = ['ChatGPT', 'ChatGPT-4', 'ChatGPT-4o',]
df_filtered['tool'] = df_filtered['tool'].replace(
    {v: 'ChatGPT' for v in chatgpt_variants}
)
print(f"\nMerged ChatGPT variants {chatgpt_variants} → 'ChatGPT (all variants)'")


# ============================================================
# GET TOP 10 TOOLS OVERALL
# ============================================================
overall_tool_counts = df_filtered.groupby('tool')['paper_id'].nunique().sort_values(ascending=False)
top_10_tools = overall_tool_counts.head(10).index.tolist()

print(f"\nTop 10 tools overall (across all 28 countries):")
for i, tool in enumerate(top_10_tools, 1):
    print(f"  {i:2d}. {tool:25s} : {overall_tool_counts[tool]:4d} papers")

# ============================================================
# PREPARE DATA FOR VISUALIZATION
# ============================================================
plot_data = []
for country in selected_countries_ordered:
    country_data = df_filtered[df_filtered['country'] == country]
    for tool in top_10_tools:
        tool_papers = country_data[country_data['tool'] == tool]['paper_id'].nunique()
        plot_data.append({'country': country, 'tool': tool, 'papers': tool_papers})

df_plot = pd.DataFrame(plot_data)
df_pivot = df_plot.pivot(index='country', columns='tool', values='papers').fillna(0)
df_pivot = df_pivot.reindex(selected_countries_ordered)

# ============================================================
# CREATE STACKED BAR PLOT
# ============================================================
print("\nCreating stacked bar plot...")

fig, ax = plt.subplots(figsize=(22, 11), dpi=300)

x = np.arange(len(selected_countries_ordered))
width = 0.65

colors = [ '#2E5A87', '#C73E1D', '#F18F01', '#3A86FF', '#1A936F', '#8C4D9E', '#FFCD00', '#00A6A6', '#FF7EC9', '#4B3621' ]

bottom = np.zeros(len(selected_countries_ordered))

for i, tool in enumerate(top_10_tools):
    values = df_pivot[tool].values
    bars = ax.bar(x, values, width, label=tool, color=colors[i],
                  bottom=bottom, edgecolor='#333333', linewidth=0.7)

    # ── numbers inside segments ──
    for j, (bar, val) in enumerate(zip(bars, values)):
        if val > 25:
            y_position = bottom[j] + bar.get_height() / 2
            ax.text(bar.get_x() + bar.get_width() / 2., y_position,
                    f'{int(val)}', ha='center', va='center',
                    fontsize=28, fontweight='bold', color='white')  # ← was 7

    bottom += values

# ── totals on top of bars ──
for j, (bar_x, total) in enumerate(zip(x, bottom)):
    ax.text(bar_x, total + 3, f'{int(total)}', ha='center', va='bottom',
            fontsize=32, fontweight='bold', color='#333333')  # ← was 8

# ── axis labels and ticks ──
country_abbrevs = [country_abbreviations[c] for c in selected_countries_ordered]
ax.set_xlabel('Country', fontsize=38, fontweight='bold', labelpad=12)
ax.set_ylabel('Number of Tool Instances', fontsize=38, fontweight='bold', labelpad=12)
ax.set_xticks(x)
ax.set_xticklabels(country_abbrevs, rotation=45, ha='right', fontsize=36)  # ← was 12

# ── legend: upper left, large ──
ax.legend(loc='upper left', fontsize=30, frameon=True,        # ← was upper right, 11
          edgecolor='#333333', fancybox=False, shadow=False,
          ncol=2)                                              # ← was ncol=2

# Styling
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(1.7)
ax.spines['bottom'].set_linewidth(1.7)
ax.yaxis.grid(True, linestyle='--', alpha=0.4, color='gray', linewidth=1)
ax.set_axisbelow(True)

max_val = bottom.max()
ax.set_ylim(0, max_val * 1.18)   # ← slightly more headroom for legend + totals

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'top10_tools_28countries_stacked.pdf'),
            bbox_inches='tight', facecolor='white')
print("✓ Figure saved: top10_tools_28countries_stacked.pdf")
plt.show()


# all 28 countries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ============================================================
# COUNTRY SELECTION - All 28 countries from the table
# ============================================================
countries = [
    "Norway", "India", "Brazil", "Japan", "Sweden",
    "Austria", "China", "Belgium", "United States", "Spain",
    "Switzerland", "Algeria", "Germany", "Netherlands",
    "Mexico", "Slovakia", "Portugal", "France", "Finland",
    "United Kingdom", "Czech Republic", "Ireland", "Italy",
    "Canada", "Poland", "Ukraine", "Greece", "Kazakhstan"
]

country_abbreviations = {
    "Norway": "NOR", "India": "IND", "Brazil": "BRA",
    "Japan": "JPN", "Sweden": "SWE", "Austria": "AUT",
    "China": "CHN", "Belgium": "BEL", "United States": "USA",
    "Spain": "ESP", "Switzerland": "CHE", "Algeria": "DZA",
    "Germany": "DEU", "Netherlands": "NLD", "Mexico": "MEX",
    "Slovakia": "SVK", "Portugal": "PRT", "France": "FRA",
    "Finland": "FIN", "United Kingdom": "GBR", "Czech Republic": "CZE",
    "Ireland": "IRL", "Italy": "ITA", "Canada": "CAN",
    "Poland": "POL", "Ukraine": "UKR", "Greece": "GRC",
    "Kazakhstan": "KAZ"
}

# ============================================================
# PROFESSIONAL STYLE CONFIGURATION
# ============================================================
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 34,
    'axes.labelsize': 38,
    'axes.titlesize': 38,
    'xtick.labelsize': 38,
    'ytick.labelsize': 38,
    'axes.linewidth': 1.5,
    'axes.edgecolor': '#333333',
    'axes.labelcolor': '#333333',
    'text.color': '#333333',
    'xtick.color': '#333333',
    'ytick.color': '#333333',
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.grid': False,
    'figure.dpi': 600,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight'
})

# ============================================================
# SETUP
# ============================================================
output_dir = r"path_to_output_directory"  # <-- UPDATE THIS PATH
os.makedirs(output_dir, exist_ok=True)

print("="*80)
print("TOP 10 TOOLS BY ALL 28 COUNTRIES - STACKED BAR CHART")
print("="*80)

# ============================================================
# FILTER DATA BY SELECTED COUNTRIES
# ============================================================
df_merged_filtered = df_merged[df_merged['country'].isin(countries)].copy()
print(f"\nFiltered to {len(df_merged_filtered)} rows from selected countries")

country_papers = df_merged_filtered[['paper_id', 'country']].drop_duplicates()
country_counts = country_papers['country'].value_counts()

eligible_countries = country_counts[country_counts >= 100].index.tolist()

# Order countries by AI usage rate (descending), matching the table
country_order_by_rate = [
    "Norway", "India", "Brazil", "Japan", "Sweden",
    "Austria", "China", "Belgium", "United States", "Spain",
    "Switzerland", "Algeria", "Germany", "Netherlands", "Mexico",
    "Slovakia", "Portugal", "France", "Finland", "United Kingdom",
    "Czech Republic", "Ireland", "Italy", "Canada", "Poland",
    "Ukraine", "Greece", "Kazakhstan"
]
selected_countries_ordered = [c for c in country_order_by_rate
                               if c in eligible_countries]

print(f"\nSelected countries and paper counts:")
for i, country in enumerate(selected_countries_ordered, 1):
    count = country_counts.get(country, 0)
    print(f"  {i:2d}. {country:25s} : {count:4d} papers")

# ============================================================
# FILTER OUT AI (unspecified) and No AI tool
# ============================================================
df_filtered = df_merged_filtered[
    ~df_merged_filtered['tool'].isin(['No AI tool', 'AI (unspecified)'])
].copy()
print(f"\nRemoved {len(df_merged_filtered) - len(df_filtered)} rows with 'AI (unspecified)' or 'No AI tool'")

# ============================================================
# MERGE ALL CHATGPT VARIANTS INTO ONE
# ============================================================
chatgpt_variants = ['ChatGPT', 'ChatGPT-4', 'ChatGPT-4o']
df_filtered['tool'] = df_filtered['tool'].replace(
    {v: 'ChatGPT' for v in chatgpt_variants}
)
print(f"\nMerged ChatGPT variants {chatgpt_variants} → 'ChatGPT'")

# ============================================================
# GET TOP 10 TOOLS OVERALL
# ============================================================
overall_tool_counts = df_filtered.groupby('tool')['paper_id'].nunique().sort_values(ascending=False)
top_10_tools = overall_tool_counts.head(10).index.tolist()

print(f"\nTop 10 tools overall (across all 28 countries):")
for i, tool in enumerate(top_10_tools, 1):
    print(f"  {i:2d}. {tool:25s} : {overall_tool_counts[tool]:4d} papers")

# ============================================================
# PREPARE DATA FOR VISUALIZATION
# ============================================================
plot_data = []
for country in selected_countries_ordered:
    country_data = df_filtered[df_filtered['country'] == country]
    for tool in top_10_tools:
        tool_papers = country_data[country_data['tool'] == tool]['paper_id'].nunique()
        plot_data.append({'country': country, 'tool': tool, 'papers': tool_papers})

df_plot = pd.DataFrame(plot_data)
df_pivot = df_plot.pivot(index='country', columns='tool', values='papers').fillna(0)
df_pivot = df_pivot.reindex(selected_countries_ordered)

# ============================================================
# PART 1 — STACKED BAR FIGURE (unchanged)
# ============================================================
print("\nCreating stacked bar plot...")

fig, ax = plt.subplots(figsize=(22, 11), dpi=300)

x = np.arange(len(selected_countries_ordered))
width = 0.65

colors = [
    '#2E5A87',  # deep blue
    '#C73E1D',  # red-orange
    '#F18F01',  # orange
    '#6A4C93',  # purple
    '#1A936F',  # teal-green
    '#FF595E',  # bright red
    '#FFCA3A',  # yellow
    '#00B4D8',  # cyan
    '#FF7EC9',  # pink
    '#6B705C'   # olive/gray
]

bottom = np.zeros(len(selected_countries_ordered))

for i, tool in enumerate(top_10_tools):
    values = df_pivot[tool].values
    bars = ax.bar(x, values, width, label=tool, color=colors[i],
                  bottom=bottom, edgecolor='#333333', linewidth=0.7)

    # numbers inside segments
    for j, (bar, val) in enumerate(zip(bars, values)):
        if val > 25:
            y_position = bottom[j] + bar.get_height() / 2
            ax.text(bar.get_x() + bar.get_width() / 2., y_position,
                    f'{int(val)}', ha='center', va='center',
                    fontsize=28, fontweight='bold', color='white')

    bottom += values

# totals on top of bars
for j, (bar_x, total) in enumerate(zip(x, bottom)):
    ax.text(bar_x, total + 3, f'{int(total)}', ha='center', va='bottom',
            fontsize=32, fontweight='bold', color='#333333')

# axis labels and ticks
country_abbrevs = [country_abbreviations[c] for c in selected_countries_ordered]
ax.set_xlabel('Country', fontsize=38, fontweight='bold', labelpad=12)
ax.set_ylabel('Number of Tool Instances', fontsize=38, fontweight='bold', labelpad=12)
ax.set_xticks(x)
ax.set_xticklabels(country_abbrevs, rotation=45, ha='right', fontsize=36)

# legend
ax.legend(loc='upper left', fontsize=30, frameon=True,
          edgecolor='#333333', fancybox=False, shadow=False,
          ncol=2)

# styling
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(1.7)
ax.spines['bottom'].set_linewidth(1.7)
ax.yaxis.grid(True, linestyle='--', alpha=0.4, color='gray', linewidth=1)
ax.set_axisbelow(True)

max_val = bottom.max()
ax.set_ylim(0, max_val * 1.18)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'top10_tools_28countries_stacked.pdf'),
            bbox_inches='tight', facecolor='white')
print("✓ Figure saved: top10_tools_28countries_stacked.pdf")
plt.show()

# ============================================================
# PART 2 — DETAILED PER-COUNTRY PER-TOOL CSV
# ============================================================
print("\nBuilding detailed analysis table...")

# --- total papers per country (all tools, not just top 10) ---
total_papers_per_country = (
    df_merged_filtered[['paper_id', 'country']]
    .drop_duplicates()
    .groupby('country')['paper_id']
    .nunique()
)

# --- base: pivot already has country × tool paper counts ---
df_export = df_pivot[top_10_tools].copy()
df_export.index.name = 'country'

# --- add country abbreviation ---
df_export.insert(0, 'abbreviation',
                 [country_abbreviations[c] for c in df_export.index])

# --- add total papers in country (denominator for %) ---
df_export['total_papers_in_country'] = [
    total_papers_per_country.get(c, 0) for c in df_export.index
]

# --- add row total: sum of top-10 tool papers per country ---
df_export['top10_tool_total'] = df_export[top_10_tools].sum(axis=1).astype(int)

# --- % share columns: each tool's count as % of top10_tool_total ---
for tool in top_10_tools:
    pct_col = f'{tool}_pct_of_top10'
    df_export[pct_col] = (
        df_export[tool] / df_export['top10_tool_total'].replace(0, np.nan) * 100
    ).round(1)

# --- dominant tool per country ---
df_export['dominant_tool'] = df_pivot[top_10_tools].idxmax(axis=1)

# --- rank of each country by top10_tool_total ---
df_export['rank_by_total'] = df_export['top10_tool_total'].rank(
    ascending=False, method='min').astype(int)

# --- reorder columns for readability ---
count_cols  = top_10_tools
pct_cols    = [f'{t}_pct_of_top10' for t in top_10_tools]
meta_cols   = ['abbreviation', 'total_papers_in_country',
               'top10_tool_total', 'rank_by_total', 'dominant_tool']

df_export = df_export[meta_cols + count_cols + pct_cols]

# --- save ---
csv_path = os.path.join(output_dir, 'top10_tools_28countries_detailed.csv')
df_export.to_csv(csv_path)
print(f"✓ CSV saved: top10_tools_28countries_detailed.csv")

# --- print a quick summary in console ---
print("\n" + "="*80)
print("QUICK SUMMARY TABLE (counts only)")
print("="*80)
summary = df_export[['abbreviation', 'top10_tool_total', 'dominant_tool', 'rank_by_total'] + count_cols]
print(summary.to_string())
