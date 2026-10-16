"""Independent, bounded regression workspace; never writes research artifacts."""
from io import BytesIO
from itertools import combinations
from urllib.parse import urlparse
import urllib.request
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet
from sklearn.feature_selection import RFE, SequentialFeatureSelector, f_regression
from sklearn.model_selection import train_test_split, KFold, TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

MAX_BYTES = 5_000_000
ALLOWED_HOSTS = {'raw.githubusercontent.com'}
STARTUPS_URL = 'https://raw.githubusercontent.com/Avik-Jain/100-Days-Of-ML-Code/master/datasets/50_Startups.csv'
METHODS = ['Lasso', 'Forward SFS', 'RFE', 'ElasticNet', 'Exhaustive Search', 'F-test']

class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError('不接受重新導向；請使用原始 CSV HTTPS 網址。')

def fetch_csv(url):
    p = urlparse(url)
    if p.scheme != 'https' or p.hostname not in ALLOWED_HOSTS or p.username or p.password or p.port not in (None, 443) or p.fragment:
        raise ValueError('僅允許 https://raw.githubusercontent.com 的公開 CSV 網址。')
    if not p.path.lower().endswith('.csv'):
        raise ValueError('網址必須指向 .csv 檔案。')
    with urllib.request.build_opener(NoRedirect).open(url, timeout=15) as response:
        raw = response.read(MAX_BYTES + 1)
    return parse_csv(raw)

def parse_csv(raw):
    if len(raw) > MAX_BYTES:
        raise ValueError('CSV 上限為 5 MB。')
    df = pd.read_csv(BytesIO(raw), nrows=10001)
    if len(df) < 20 or len(df) > 10000 or not 2 <= len(df.columns) <= 40:
        raise ValueError('資料需有 20–10,000 列及 2–40 欄。')
    if any(str(c).startswith('Unnamed:') for c in df.columns):
        raise ValueError('請移除未命名欄位並提供完整欄名。')
    return df

def prepare(df, target, chronological=False):
    if target not in df or not pd.api.types.is_numeric_dtype(df[target]):
        raise ValueError('目標必須為數值欄位。')
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=[target]).copy()
    if len(df) < 20 or df[target].nunique() < 2:
        raise ValueError('有效資料不足或目標為常數。')
    if chronological:
        df = df.sort_values('date')
    X = df.drop(columns=[target, 'date'], errors='ignore')
    X = X.dropna(axis=1, how='all')
    if X.shape[1] == 0:
        raise ValueError('沒有有效預測欄位。')
    if len(df) > 20000 or X.shape[1] > 40:
        raise ValueError('超出訓練大小限制。')
    nums = X.select_dtypes(include='number').columns.tolist()
    cats = [c for c in X if c not in nums]
    if any(X[c].nunique() > 20 for c in cats):
        raise ValueError('類別欄位最多 20 種值，請移除識別碼或文字欄位。')
    if chronological:
        a, b = int(len(df)*.55), int(len(df)*.70)
        # Final 30% remains unavailable to the K-selection workspace.
        ti, vi = np.arange(a), np.arange(a, b)
        test = np.arange(int(len(df)*.85), len(df))
    else:
        trainval, test = train_test_split(np.arange(len(df)), test_size=.2, random_state=42)
        ti, vi = train_test_split(trainval, test_size=.25, random_state=42)
    transforms = []
    if nums:
        transforms.append(('numeric', make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True), StandardScaler()), nums))
    if cats:
        transforms.append(('category', make_pipeline(SimpleImputer(strategy='most_frequent', keep_empty_features=True), OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False)), cats))
    pre = ColumnTransformer(transforms, verbose_feature_names_out=False)
    train = pre.fit_transform(X.iloc[ti])
    val = pre.transform(X.iloc[vi])
    if train.shape[1] == 0:
        raise ValueError('編碼後沒有有效特徵；請加入數值或非固定類別欄位。')
    if train.shape[1] > 40:
        raise ValueError('編碼後特徵上限為 40。')
    return dict(X=X, y=df[target].to_numpy(float), ti=ti, vi=vi, test=test, pre=pre,
                train=train, val=val, names=pre.get_feature_names_out(), chronological=chronological)

def select_indices(ctx, method, k):
    X, y = ctx['train'], ctx['y'][ctx['ti']]
    p = X.shape[1]
    if not 1 <= k <= p:
        raise ValueError('K 超出有效範圍。')
    if k == p:
        return np.arange(p)
    if method in ('Lasso', 'ElasticNet'):
        reg = Lasso(alpha=.001, max_iter=10000) if method == 'Lasso' else ElasticNet(alpha=.001, l1_ratio=.5, max_iter=10000)
        scale = max(float(np.std(y)), 1.)
        reg.fit(X, y/scale)
        return np.sort(np.argsort(-np.abs(reg.coef_), kind='stable')[:k])
    if method == 'RFE':
        return np.flatnonzero(RFE(LinearRegression(), n_features_to_select=k).fit(X, y).support_)
    # Selection CV must refit preprocessing within each training fold.
    # Fast ranking and RFE above are fitted on training data only.
    if method in ('Forward SFS', 'Exhaustive Search'):
        if len(X) > 1000 or p > 8:
            raise ValueError('SFS／Exhaustive 僅適用於 ≤1,000 列、≤8 個編碼特徵。')
        cv = TimeSeriesSplit(3) if ctx['chronological'] else KFold(3, shuffle=True, random_state=42)
        raw = ctx['X'].iloc[ctx['ti']]
        splits = []
        for a, b in cv.split(raw):
            from sklearn.base import clone
            foldpre = clone(ctx['pre'])
            xa = foldpre.fit_transform(raw.iloc[a])
            xb = foldpre.transform(raw.iloc[b])
            # Fold-specific categories can change feature counts. Reject rather than mis-map.
            if not np.array_equal(foldpre.get_feature_names_out(), ctx['names']):
                raise ValueError('CV 折內類別不足，請選擇 Lasso、RFE 或 F-test。')
            splits.append((xa, xb, y[a], y[b]))
        def score(cols):
            return np.mean([mean_squared_error(yb, LinearRegression().fit(xa[:, cols], ya).predict(xb[:, cols])) for xa, xb, ya, yb in splits])
        if method == 'Exhaustive Search':
            return np.array(min(combinations(range(p), k), key=lambda cols: score(list(cols))))
        chosen = []
        for _ in range(k):
            chosen.append(min((j for j in range(p) if j not in chosen), key=lambda j: score(chosen+[j])))
        return np.sort(chosen)
    if method != 'F-test':
        raise ValueError('未知特徵選擇方法。')
    scores = np.nan_to_num(f_regression(X, y)[0])
    return np.sort(np.argsort(-scores, kind='stable')[:k])

def fit_model(ctx, method, k):
    cols = select_indices(ctx, method, k)
    model = LinearRegression().fit(ctx['train'][:, cols], ctx['y'][ctx['ti']])
    pred = model.predict(ctx['val'][:, cols])
    return dict(model=model, cols=cols, names=ctx['names'][cols], metrics=metrics(ctx['y'][ctx['vi']], pred, k), predictions=pred)

def metrics(y, pred, p):
    r2 = r2_score(y, pred)
    return {'RMSE':float(np.sqrt(mean_squared_error(y,pred))), 'MAE':float(mean_absolute_error(y,pred)), 'R²':float(r2),
            'Adjusted R²':float(1-(1-r2)*(len(y)-1)/(len(y)-p-1)) if len(y)>p+1 else np.nan}

def inference(X, y, names, model):
    design = np.column_stack([np.ones(len(X)), X])
    rank = np.linalg.matrix_rank(design)
    dof = len(y)-rank
    if rank != design.shape[1] or dof <= 0:
        raise ValueError('設計矩陣秩不足或自由度不足，無法提供經典 OLS 推論。')
    resid = y-model.predict(X)
    sse = float(resid@resid)
    cov = np.linalg.inv(design.T@design)*sse/dof
    se = np.sqrt(np.maximum(np.diag(cov),0))
    coef = np.r_[model.intercept_, model.coef_]
    t = np.divide(coef,se,out=np.full_like(coef,np.nan),where=se>0)
    critical = stats.t.ppf(.975,dof)
    vifs = [np.nan]
    for j in range(X.shape[1]):
        if X.shape[1] == 1: vifs.append(1.); continue
        r2 = LinearRegression().fit(np.delete(X,j,axis=1),X[:,j]).score(np.delete(X,j,axis=1),X[:,j])
        vifs.append(1/max(1-r2,1e-15))
    table = pd.DataFrame({'特徵':['Intercept']+list(names),'係數':coef,'標準誤':se,'t':t,'p-value':2*stats.t.sf(np.abs(t),dof),'95% 下界':coef-critical*se,'95% 上界':coef+critical*se,'VIF':vifs})
    sst = float(np.sum((y-np.mean(y))**2))
    f = max(0,(sst-sse)/X.shape[1])/(sse/dof) if sse>0 else np.inf
    return table, f, stats.f.sf(f,X.shape[1],dof)

def diagnostics(y, pred):
    d = pd.DataFrame({'實際值':y,'預測值':pred})
    d['殘差'] = d['實際值']-d['預測值']
    d['絕對誤差'] = d['殘差'].abs()
    d['百分比誤差 (%)'] = d['絕對誤差']/d['實際值'].abs().replace(0,np.nan)*100
    return d
