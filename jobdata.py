import html
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# job category is made from keywords in the title (first match wins)
CATEGORY_PATTERNS = {
    'AI & Machine Learning': r"\bai\b|a\.i\.|chatgpt|\bgpt|machine learning|\bllm\b|artificial intelligence|openai|prompt engineer|deep learning|midjourney|stable diffusion|\bnlp\b|computer vision|chatbot|\bml\b",
    'Data & Analytics': r"\bdata\b|analytics|analyst|\bsql\b|power bi|tableau|excel|scraping|scraper|dashboard|google sheets|spreadsheet|\betl\b|database|looker",
    'Mobile App Development': r"\bapp\b|\bapps\b|mobile|android|\bios\b|flutter|react native|swift|kotlin|iphone",
    'E-commerce': r"shopify|amazon|ebay|etsy|dropship|woocommerce|e-commerce|ecommerce|\bfba\b|walmart|tiktok shop|online store",
    'Web & Software Development': r"web|wordpress|\bwix\b|webflow|developer|programmer|\bphp\b|python|react|node|javascript|laravel|full stack|fullstack|backend|front end|frontend|software|\bapi\b|\bcode|coder|django|\.net|\bjava\b|html|\bcss\b|bubble|devops|\baws\b|cloud|engineer|saas|plugin|zapier|automation|crm|gohighlevel|\bseo\b",
    '3D, Game & Architecture': r"\b3d\b|\bcad\b|architect|revit|blender|autocad|interior|render|\bgame|unity|unreal|solidworks|sketchup",
    'Audio & Music': r"music|audio|voice|singer|\bsong|sound|mixing|\bbeat|producer|vocal|podcast",
    'Video Editing & Animation': r"video|youtube|editor|editing|animat|reel|tiktok|motion graphic|after effects|premiere|\bvfx\b|shorts|capcut|videographer|filming|ugc|\bvsl\b",
    'Graphic Design': r"design|logo|graphic|illustrat|brand|canva|photoshop|figma|\bui\b|\bux\b|thumbnail|packaging|poster|flyer|artist|drawing|\bart\b|photo",
    'Writing & Translation': r"writer|writing|copywrit|content|blog|article|ghostwrit|proofread|translat|transcri|script|ebook|e-book|\bbook|resume|\bcv\b|author|editor|newsletter",
    'Sales & Lead Generation': r"\bsales|\blead|appointment|\bcold|outreach|closer|\bsdr\b|\bb2b\b|setter|prospect|linkedin",
    'Social Media & Marketing': r"marketing|marketer|social media|\bads\b|\bppc\b|facebook|google ads|email|growth|campaign|media buyer|funnel|affiliate|influencer|instagram|community manager|advertis",
    'Customer Support': r"customer|support|service|chat agent|call center|receptionist|\bagent\b",
    'Finance & Accounting': r"account|bookkeep|\btax\b|finance|financial|quickbooks|crypto|trading|forex|invest|payroll|\bxero\b|blockchain|nft|web3",
    'Virtual Assistant & Admin': r"virtual assistant|\bva\b|assistant|admin|data entry|secretary|manager|coordinator|scheduler|research|organizer|project management|\bpm\b|recruit",
}


def load_data(path):
    df = pd.read_csv(path)

    df['published_date'] = pd.to_datetime(df['published_date'])
    df['date'] = df['published_date'].dt.tz_convert(None).dt.normalize()

    df['title'] = df['title'].fillna('').apply(html.unescape)
    df['title_clean'] = df['title'].str.lower().str.strip()
    df = df.drop_duplicates(subset=['title_clean', 'published_date'])
    df = df[df['title_clean'] != '']

    df['hourly_rate'] = df[['hourly_low', 'hourly_high']].mean(axis=1)
    df.loc[df['hourly_rate'] > 150, 'hourly_rate'] = np.nan
    df.loc[df['budget'] > 5000, 'budget'] = np.nan

    # keep only the days which have real data (the file has a few old stray rows)
    day_count = df.groupby('date').size()
    good_days = day_count[day_count >= 500].index
    df = df[df['date'].isin(good_days)].copy()

    df['category'] = 'Other'
    for cat, pattern in CATEGORY_PATTERNS.items():
        match = df['title_clean'].str.contains(pattern, regex=True)
        df.loc[(df['category'] == 'Other') & match, 'category'] = cat

    return df.reset_index(drop=True)


class Recommender:
    def __init__(self, df):
        # one copy of every title, newest first
        jobs = df.sort_values('published_date', ascending=False)
        jobs = jobs.drop_duplicates('title_clean').reset_index(drop=True)
        jobs['job_type'] = np.where(jobs['is_hourly'], 'Hourly', 'Fixed')
        jobs['pay'] = np.where(jobs['is_hourly'], jobs['hourly_rate'], jobs['budget'])
        self.jobs = jobs

        self.tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), min_df=2)
        self.matrix = self.tfidf.fit_transform(jobs['title_clean'])

        self.countries = sorted(jobs['country'].dropna().unique())

    def recommend(self, skills, job_type='any', min_hourly=0, country=None, top_n=10):
        query = self.tfidf.transform([skills.lower()])
        scores = cosine_similarity(query, self.matrix).ravel()
        jobs = self.jobs

        keep = scores > 0
        if job_type == 'hourly':
            keep = keep & jobs['is_hourly'].values
        elif job_type == 'fixed':
            keep = keep & (~jobs['is_hourly'].values)
        if min_hourly > 0:
            keep = keep & (jobs['hourly_rate'].fillna(0).values >= min_hourly)
        if country:
            keep = keep & (jobs['country'] == country).values

        index = np.where(keep)[0]
        index = index[np.argsort(-scores[index], kind='stable')][:top_n]

        result = []
        for i in index:
            row = jobs.loc[i]
            pay = None if pd.isnull(row['pay']) else float(row['pay'])
            country_name = None if pd.isnull(row['country']) else row['country']
            result.append({
                'title': row['title'],
                'category': row['category'],
                'job_type': row['job_type'],
                'pay': pay,
                'country': country_name,
                'match_score': round(float(scores[i]), 3),
                'link': row['link'],
                'posted': row['published_date'].strftime('%Y-%m-%d'),
            })
        return result


def make_dashboard_data(df):
    df = df.copy()
    # calendar week (starts on Monday) and month
    df['week_start'] = df['date'] - pd.to_timedelta(df['date'].dt.dayofweek, unit='D')
    df['month'] = df['date'].dt.strftime('%Y-%m')

    def summary(group_col):
        out = df.groupby(group_col).agg(jobs=('title', 'size'),
                                        days=('date', 'nunique'),
                                        hourly_share=('is_hourly', 'mean'),
                                        median_hourly_rate=('hourly_rate', 'median'),
                                        median_budget=('budget', 'median'))
        out['jobs_per_day'] = out['jobs'] / out['days']
        out['hourly_share'] = out['hourly_share'] * 100
        return out

    weekly = summary('week_start')
    monthly = summary('month')
    monthly['jobs_per_day_change_%'] = monthly['jobs_per_day'].pct_change() * 100

    top_categories = df['category'].value_counts().drop('Other').head(5).index
    cat_week = pd.crosstab(df['week_start'], df['category'], normalize='index') * 100

    weeks = [d.strftime('%Y-%m-%d') for d in weekly.index]
    return {
        'weeks': weeks,
        'days_in_week': weekly['days'].tolist(),
        'jobs_per_day': weekly['jobs_per_day'].round(0).tolist(),
        'hourly_share': weekly['hourly_share'].round(1).tolist(),
        'median_hourly_rate': weekly['median_hourly_rate'].round(2).tolist(),
        'median_budget': weekly['median_budget'].round(1).tolist(),
        'category_share': {cat: cat_week[cat].round(2).tolist() for cat in top_categories},
        'monthly': monthly.round(2).reset_index().fillna(0).to_dict('records'),
        'total_jobs': int(len(df)),
        'first_date': df['date'].min().strftime('%Y-%m-%d'),
        'last_date': df['date'].max().strftime('%Y-%m-%d'),
    }
