<div dir="rtl" align="right">

# راهنمای فارسی ارائه و یادگیری پایتون - آزمایش ۳

این فایل ترجمه و نسخهٔ راست‌به‌چپ فایل `PRESENTATION_GUIDE.md` است و دقیقاً از ترتیب نوت‌بوک `kmeans_rent_classification.ipynb` پیروی می‌کند.

هر بخش شامل دو قسمت است:

1. **توضیح مناسب برای TA** - خلاصه‌ای که می‌توانید در ارائه بیان کنید.
2. **توضیح خط‌به‌خط پایتون** - معنی هر دستور و دلیل استفاده از آن.

**روش شمارش خطوط:** شماره‌ها دقیقاً با شمارهٔ خطوط PyCharm در هر سلول کد نوت‌بوک مطابقت دارند. PyCharm برای هر سلول دوباره از خط ۱ شروع می‌کند. خطوط خالی و خطوطی که فقط توضیح یا `comment` هستند نیز شمرده می‌شوند. اگر چند دستور با `;` در یک خط نوشته شده باشند، همهٔ آن‌ها یک شماره خط دارند.

در این پروژه بررسی می‌کنیم که آیا ۲۱ استان سوئد را می‌توان بر اساس دو ویژگی زیر به گروه‌های طبیعی تقسیم کرد:

- اجارهٔ سالانه به‌ازای هر مترمربع؛
- میانگین درآمد سالانه برحسب هزار کرون سوئد یا KSEK.

الگوریتم K-means و محاسبهٔ Silhouette از ابتدا پیاده‌سازی شده‌اند. از Pandas برای داده‌های جدولی، از NumPy برای محاسبات عددی و از Matplotlib برای رسم نمودار استفاده می‌کنیم.

---

## بخش ۱: واردکردن کتابخانه‌ها و تنظیم اجرای تکرارپذیر

### توضیح مناسب برای TA

ابتدا کتابخانه‌های لازم برای مدیریت مسیر فایل، آرایه‌های عددی، جدول داده و رسم نمودار را وارد می‌کنم. یک seed تصادفی ثابت تعریف می‌کنم تا K-means در هر اجرا centroidهای اولیهٔ یکسانی انتخاب کند و نتیجه قابل تکرار باشد. سپس مسیر فایل داده را تعیین می‌کنم. مسیر جایگزین باعث می‌شود اگر CSV کنار نوت‌بوک قرار گرفت نیز برنامه اجرا شود.

### کد

</div>

```python
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-whitegrid')
pd.set_option('display.precision', 2)
RANDOM_STATE = 42
DATA_PATH = Path('data/inc_vs_rent.csv')
if not DATA_PATH.exists():
    DATA_PATH = Path('inc_vs_rent.csv')  # convenient fallback
```

<div dir="rtl" align="right">

### توضیح خط‌به‌خط

- **خط ۱:** کلاس `Path` را از `pathlib` وارد می‌کند تا با مسیر فایل‌ها به‌شکل شیءگرا کار کنیم.
- **خط ۲:** NumPy را با نام کوتاه `np` وارد می‌کند. از آن برای آرایه، فاصله، میانگین و انتخاب تصادفی استفاده می‌شود.
- **خط ۳:** Pandas را با نام `pd` وارد می‌کند تا CSV را به‌شکل DataFrame بخوانیم و تحلیل کنیم.
- **خط ۴:** بخش رسم نمودار Matplotlib را با نام `plt` وارد می‌کند.
- **خط ۵:** خط خالی برای خوانایی است.
- **خط ۶:** یک قالب سفید و شبکه‌دار برای نمودارها انتخاب می‌کند.
- **خط ۷:** نمایش اعداد اعشاری در جدول‌های Pandas را روی دو رقم اعشار تنظیم می‌کند.
- **خط ۸:** seed ثابت ۴۲ را ذخیره می‌کند. خود عدد مهم نیست؛ ثابت‌بودن آن نتیجه را تکرارپذیر می‌کند.
- **خط ۹:** مسیر اصلی CSV را در پوشهٔ `data` می‌سازد.
- **خط ۱۰:** بررسی می‌کند آیا فایل در مسیر اصلی وجود ندارد.
- **خط ۱۱:** اگر فایل وجود نداشت، مسیر جایگزین کنار نوت‌بوک را انتخاب می‌کند.

---

## بخش ۲: بارگذاری دادهٔ خام

### توضیح مناسب برای TA

فایل CSV را داخل یک DataFrame می‌خوانم. ستون اول فایل فقط index ذخیره‌شده است، بنابراین با `index_col=0` آن را به‌عنوان ویژگی استفاده نمی‌کنم. سپس ده ردیف اول را نمایش می‌دهم تا ساختار و مقادیر داده را بررسی کنم. در PDF نام `rent_vs_inc.csv` آمده، ولی فایل واقعی تحویل‌شده `inc_vs_rent.csv` است.

### کد

</div>

```python
df = pd.read_csv(DATA_PATH, index_col=0)
df.head(10)
```

<div dir="rtl" align="right">

### توضیح خط‌به‌خط

- **خط ۱:** `pd.read_csv` فایل CSV را می‌خواند. `DATA_PATH` محل فایل را مشخص می‌کند و `index_col=0` ستون اول را index قرار می‌دهد، نه یک ویژگی داده.
- **خط ۲:** `df.head(10)` ده ردیف اول را برمی‌گرداند. چون آخرین عبارت سلول است، Jupyter آن را به‌شکل جدول نمایش می‌دهد.

### اطلاعاتی که فقط با دیدن جدول به‌دست می‌آوریم

- هر ردیف نمایندهٔ یک استان است.
- همهٔ داده‌ها مربوط به سال ۲۰۲۰ هستند.
- ستون `region` متن توصیفی است و وارد محاسبهٔ فاصله نمی‌شود.
- دو ویژگی عددی خوشه‌بندی `Annual rent sqm` و `Avg yearly inc KSEK` هستند.
- بازهٔ عددی اجاره از درآمد بزرگ‌تر است؛ بنابراین در فاصلهٔ اقلیدسی خام، اجاره اثر بیشتری دارد.

---

## بخش ۳: بررسی کیفیت داده و آمار توصیفی

### توضیح مناسب برای TA

قبل از مدل‌سازی، اندازه و کیفیت داده را کنترل می‌کنم. مجموعه‌داده ۲۱ ردیف دارد، مقدار گمشده ندارد و نام استان تکراری نیست. آمار توصیفی مرکز، پراکندگی و بازهٔ ستون‌های عددی را نشان می‌دهد. این بررسی مهم است، چون K-means به مقادیر عددی و مقیاس ویژگی‌ها حساس است.

### کد

</div>

```python
print(f'Rows: {len(df)}, columns: {df.shape[1]}')
print('Missing values:', int(df.isna().sum().sum()))
print('Duplicate regions:', int(df['region'].duplicated().sum()))
display(df.describe(include=[np.number]))
```

<div dir="rtl" align="right">

### توضیح خط‌به‌خط

- **خط ۱:** `len(df)` تعداد ردیف‌ها و `df.shape[1]` تعداد ستون‌ها را می‌دهد. حرف `f` قبل از رشته اجازه می‌دهد عبارت‌های داخل `{}` در متن قرار بگیرند.
- **خط ۲:** `df.isna()` محل مقادیر گمشده را با `True` مشخص می‌کند. دو بار `.sum()` ابتدا برای هر ستون و سپس برای کل جدول جمع می‌زند. `int` نتیجه را به عدد صحیح معمولی تبدیل می‌کند.
- **خط ۳:** ستون `region` را انتخاب می‌کند، موارد تکراری را با `.duplicated()` پیدا می‌کند و تعدادشان را می‌شمارد.
- **خط ۴:** `describe` تعداد، میانگین، انحراف معیار، حداقل، چارک‌ها و حداکثر ستون‌های عددی را محاسبه می‌کند و `display` جدول را نشان می‌دهد.

### چرا داده را به train و validation تقسیم نمی‌کنیم؟

این مسئله یادگیری بدون ناظر است و برچسب صحیح خوشه برای استان‌ها نداریم. بنابراین validation set دارای target واقعی نیست که بتوان با آن accuracy را محاسبه کرد. از همهٔ نقاط برای یافتن ساختار استفاده می‌کنیم و کیفیت درونی خوشه‌ها را با Silhouette می‌سنجیم. اگر هدف بررسی پایداری روی سال‌های آینده بود، می‌شد از دادهٔ خارجی یا روش‌های بازنمونه‌گیری استفاده کرد.

---

## بخش ۴: رسم نمودار پراکندگی اولیه

### توضیح مناسب برای TA

دو ویژگی لازم را انتخاب و به ماتریس NumPy تبدیل می‌کنم، چون مدل دست‌نویس با آرایهٔ عددی کار می‌کند. محور x اجاره و محور y درآمد است و هر نقطه یک استان را نشان می‌دهد. نمودار قبل از خوشه‌بندی به ما اجازه می‌دهد شکل داده را ببینیم؛ بیشتر استان‌ها در ناحیهٔ اجارهٔ پایین‌تر قرار دارند و Stockholm و Uppsala مقادیر بالاتری دارند.

### کد

</div>

```python
FEATURES = ['Annual rent sqm', 'Avg yearly inc KSEK']
X = df[FEATURES].to_numpy(dtype=float)

fig, ax = plt.subplots(figsize=(9, 6))
ax.scatter(X[:, 0], X[:, 1], s=75, color='steelblue', edgecolor='white')
for _, row in df.iterrows():
    ax.annotate(row['region'].split(' ', 1)[1].replace(' county', ''),
                (row[FEATURES[0]], row[FEATURES[1]]), xytext=(4, 4),
                textcoords='offset points', fontsize=7)
ax.set(xlabel='Annual rent per m²', ylabel='Average yearly income (KSEK)',
       title='Swedish counties: rent versus income (2020)')
plt.tight_layout(); plt.show()
```

<div dir="rtl" align="right">

### توضیح خط‌به‌خط

- **خط ۱:** فهرست نام دقیق دو ستون ویژگی را می‌سازد.
- **خط ۲:** این دو ستون را به‌ترتیب انتخاب و با نوع `float` به آرایهٔ دوبعدی `X` تبدیل می‌کند.
- **خط ۳:** خط خالی است.
- **خط ۴:** یک figure و یک axes با اندازهٔ ۹ در ۶ اینچ می‌سازد.
- **خط ۵:** نقاط را رسم می‌کند. `X[:, 0]` یعنی همهٔ ردیف‌های ستون اول و `X[:, 1]` یعنی همهٔ ردیف‌های ستون دوم.
- **خط ۶:** با `iterrows` روی ردیف‌های DataFrame حرکت می‌کند. `_` همان index استفاده‌نشده و `row` دادهٔ ردیف است.
- **خط ۷:** برچسب استان را اضافه می‌کند؛ `split` کد عددی ابتدای نام و `replace` کلمهٔ `county` را حذف می‌کند.
- **خط ۸:** مختصات برچسب را تعیین و آن را چهار واحد نمایشی به راست و بالا جابه‌جا می‌کند.
- **خط ۹:** واحد offset و اندازهٔ فونت را مشخص می‌کند.
- **خطوط ۱۰ تا ۱۱:** نام محورهای x و y و عنوان نمودار را تعیین می‌کنند.
- **خط ۱۲:** `tight_layout` از بریده‌شدن اجزا جلوگیری می‌کند و `show` نمودار را نمایش می‌دهد. هر دو دستور با `;` در یک خط هستند.

---

## بخش ۵: تعریف کلاس K-means از ابتدا

### توضیح مناسب برای TA

K-means را از ابتدا و به‌شکل یک کلاس پیاده‌سازی می‌کنم. ابتدا تعدادی مشاهدهٔ واقعی به‌طور تصادفی به‌عنوان centroid انتخاب می‌شوند. سپس هر نقطه به نزدیک‌ترین centroid نسبت داده می‌شود و هر centroid با میانگین اعضای خوشهٔ خود به‌روزرسانی می‌شود. این فرایند تا زمانی ادامه دارد که بیشترین حرکت centroid از tolerance کمتر شود یا تعداد تکرار به حداکثر برسد. مدل centroidها، labelها، تعداد iteration و inertia را ذخیره می‌کند و می‌تواند برای نقطهٔ جدید نیز خوشه پیش‌بینی کند.

### کد

</div>

```python
class KMeansScratch:
    """K-means using Euclidean distance and sample-based random initialization."""
    def __init__(self, n_clusters=3, max_iter=100, tol=1e-6, random_state=None):
        if n_clusters < 1:
            raise ValueError('n_clusters must be at least 1')
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    @staticmethod
    def _distances(X, centroids):
        return np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or self.n_clusters > len(X):
            raise ValueError('X must be 2D and n_clusters cannot exceed sample count')
        rng = np.random.default_rng(self.random_state)
        self.centroids_ = X[rng.choice(len(X), self.n_clusters, replace=False)].copy()
        for iteration in range(self.max_iter):
            distances = self._distances(X, self.centroids_)
            labels = distances.argmin(axis=1)
            new_centroids = np.empty_like(self.centroids_)
            for cluster in range(self.n_clusters):
                members = X[labels == cluster]
                if len(members):
                    new_centroids[cluster] = members.mean(axis=0)
                else:
                    # Re-seed an empty cluster with the currently worst represented point.
                    new_centroids[cluster] = X[distances.min(axis=1).argmax()]
            shift = np.linalg.norm(new_centroids - self.centroids_, axis=1).max()
            self.centroids_ = new_centroids
            if shift <= self.tol:
                break
        self.n_iter_ = iteration + 1
        self.labels_ = self.predict(X)
        self.inertia_ = float(np.sum((X - self.centroids_[self.labels_]) ** 2))
        return self

    def predict(self, X):
        if not hasattr(self, 'centroids_'):
            raise RuntimeError('Call fit before predict')
        X = np.asarray(X, dtype=float)
        return self._distances(X, self.centroids_).argmin(axis=1)

    def fit_predict(self, X):
        return self.fit(X).labels_
```

<div dir="rtl" align="right">

### توضیح خط‌به‌خط

- **خط ۱:** کلاس جدید `KMeansScratch` را تعریف می‌کند. کلاس داده و رفتار مدل را در یک شیء قرار می‌دهد.
- **خط ۲:** docstring کوتاهی است که کلاس را توضیح می‌دهد.
- **خط ۳:** سازندهٔ کلاس است. `self` به شیء فعلی اشاره می‌کند و بقیهٔ پارامترها مقدار پیش‌فرض دارند.
- **خط ۴:** بررسی می‌کند تعداد خوشه کمتر از ۱ نباشد.
- **خط ۵:** اگر مقدار نامعتبر باشد، خطای واضح ایجاد می‌کند.
- **خطوط ۶ تا ۹:** پارامترهای سازنده را به‌عنوان attribute شیء ذخیره می‌کنند.
- **خط ۱۰:** خط خالی است.
- **خط ۱۱:** `@staticmethod` می‌گوید این تابع کمکی به یک شیء fitشده وابسته نیست.
- **خط ۱۲:** متد داخلی محاسبهٔ فاصله را تعریف می‌کند. `_` اول نام نشان می‌دهد متد کمکی است.
- **خط ۱۳:** با broadcasting اختلاف هر نقطه با هر centroid را می‌سازد و با `np.linalg.norm` فاصلهٔ اقلیدسی را روی محور ویژگی محاسبه می‌کند.
- **خط ۱۴:** خط خالی است.
- **خط ۱۵:** متد `fit` را تعریف می‌کند؛ همان خطی که فرایند یادگیری خوشه‌ها از آن آغاز می‌شود.
- **خط ۱۶:** ورودی را به آرایهٔ NumPy با نوع float تبدیل می‌کند.
- **خط ۱۷:** بررسی می‌کند `X` دوبعدی باشد و تعداد خوشه از تعداد نمونه بیشتر نباشد.
- **خط ۱۸:** در صورت نامعتبر بودن ورودی، خطا ایجاد می‌کند.
- **خط ۱۹:** مولد عدد تصادفی NumPy را با seed ذخیره‌شده می‌سازد.
- **خط ۲۰:** بدون تکرار، `n_clusters` ردیف تصادفی را به‌عنوان centroid اولیه انتخاب می‌کند. `copy` نسخه‌ای مستقل می‌سازد.
- **خط ۲۱:** حلقهٔ به‌روزرسانی را تا حداکثر `max_iter` شروع می‌کند.
- **خط ۲۲:** فاصلهٔ همهٔ نقاط تا همهٔ centroidها را محاسبه می‌کند.
- **خط ۲۳:** با `argmin(axis=1)` ستون کمترین فاصله را برای هر ردیف پیدا می‌کند؛ این ستون label خوشه است.
- **خط ۲۴:** آرایه‌ای هم‌اندازه و هم‌نوع centroidها برای مقادیر جدید آماده می‌کند.
- **خط ۲۵:** روی شمارهٔ همهٔ خوشه‌ها حلقه می‌زند.
- **خط ۲۶:** با Boolean indexing اعضای خوشهٔ فعلی را انتخاب می‌کند.
- **خط ۲۷:** بررسی می‌کند خوشه حداقل یک عضو داشته باشد.
- **خط ۲۸:** میانگین هر ویژگی اعضا را centroid جدید قرار می‌دهد.
- **خط ۲۹:** حالت نادر خوشهٔ خالی را مدیریت می‌کند.
- **خط ۳۰:** comment توضیح می‌دهد که خوشهٔ خالی چگونه دوباره مقداردهی می‌شود؛ PyCharm این خط را نیز می‌شمارد.
- **خط ۳۱:** نقطه‌ای را انتخاب می‌کند که از نزدیک‌ترین centroid فعلی بیشترین فاصله را دارد.
- **خط ۳۲:** فاصلهٔ حرکت هر centroid را محاسبه و بیشترین آن‌ها را در `shift` ذخیره می‌کند.
- **خط ۳۳:** centroidهای قدیمی را با centroidهای جدید جایگزین می‌کند.
- **خط ۳۴:** بررسی می‌کند بیشترین حرکت از tolerance کمتر یا مساوی باشد.
- **خط ۳۵:** در صورت همگرایی، حلقه را متوقف می‌کند.
- **خط ۳۶:** تعداد iteration اجراشده را ذخیره می‌کند. `+1` شمارندهٔ صفرمبنا را به تعداد انسانی تبدیل می‌کند.
- **خط ۳۷:** labelهای نهایی دادهٔ آموزشی را با centroidهای نهایی محاسبه و ذخیره می‌کند.
- **خط ۳۸:** inertia یعنی مجموع مربع فاصلهٔ هر نقطه تا centroid خوشهٔ خودش را محاسبه می‌کند.
- **خط ۳۹:** خود شیء fitشده را برمی‌گرداند تا بتوان `.fit(X)` را در همان عبارت استفاده کرد.
- **خط ۴۰:** خط خالی است.
- **خط ۴۱:** متد `predict` را برای نسبت‌دادن نقاط به centroidهای یادگرفته‌شده تعریف می‌کند.
- **خط ۴۲:** بررسی می‌کند attribute مربوط به centroidها پس از fit وجود داشته باشد.
- **خط ۴۳:** اگر قبل از fit پیش‌بینی کنیم، خطای واضح ایجاد می‌کند.
- **خط ۴۴:** دادهٔ جدید را به آرایهٔ float تبدیل می‌کند.
- **خط ۴۵:** فاصله تا centroidها را محاسبه و index نزدیک‌ترین centroid را برای هر نقطه برمی‌گرداند.
- **خط ۴۶:** خط خالی است.
- **خط ۴۷:** متد کمکی `fit_predict` را تعریف می‌کند.
- **خط ۴۸:** مدل را fit می‌کند و مستقیماً labelهای ذخیره‌شده را برمی‌گرداند.

فاصلهٔ اقلیدسی بین دو نقطهٔ دوبعدی چنین است:

\[
d = \sqrt{(x_1-x_2)^2 + (y_1-y_2)^2}
\]

---

## بخش ۶: برازش و نمایش مدل اولیه با سه خوشه

### توضیح مناسب برای TA

قبل از تنظیم hyperparameter، کار کلاس را با سه خوشه نمایش می‌دهم. رنگ نقاط label خوشه و علامت‌های X بزرگ centroidها را نشان می‌دهند. شمارهٔ خوشه فقط یک شناسهٔ دلخواه است و رتبه یا کیفیت را نشان نمی‌دهد.

### کد

</div>

```python
initial_model = KMeansScratch(n_clusters=3, max_iter=100, random_state=RANDOM_STATE).fit(X)
print(f'Converged in {initial_model.n_iter_} iterations; inertia = {initial_model.inertia_:.2f}')

fig, ax = plt.subplots(figsize=(9, 6))
points = ax.scatter(X[:, 0], X[:, 1], c=initial_model.labels_, cmap='tab10',
                    s=80, edgecolor='white')
ax.scatter(initial_model.centroids_[:, 0], initial_model.centroids_[:, 1],
           marker='X', s=240, c=np.arange(3), cmap='tab10', edgecolor='black',
           linewidth=1.2, label='Centroids')
ax.set(xlabel='Annual rent per m²', ylabel='Average yearly income (KSEK)',
       title='Initial K-means solution (k=3)')
ax.legend(); plt.tight_layout(); plt.show()
```

<div dir="rtl" align="right">

### توضیح خط‌به‌خط

- **خط ۱:** مدل سه‌خوشه‌ای را می‌سازد و بلافاصله روی `X` fit می‌کند.
- **خط ۲:** تعداد iteration و inertia را چاپ می‌کند. `:.2f` دو رقم اعشار نمایش می‌دهد.
- **خط ۳:** خط خالی است.
- **خط ۴:** figure و axes را می‌سازد.
- **خطوط ۵ تا ۶:** نقاط را بر اساس label با colormap رنگ می‌کند و شیء نمودار را در `points` می‌گذارد.
- **خطوط ۷ تا ۹:** centroidها را با علامت X بزرگ و رنگ متناظر رسم می‌کنند.
- **خطوط ۱۰ تا ۱۱:** نام محورها و عنوان را تعیین می‌کنند.
- **خط ۱۲:** legend را نشان می‌دهد، layout را اصلاح می‌کند و نمودار را نمایش می‌دهد.

---

## بخش ۷: محاسبهٔ Silhouette از ابتدا

### توضیح مناسب برای TA

Silhouette هم فشردگی داخل خوشه و هم جدایی بین خوشه‌ها را اندازه می‌گیرد. برای نقطهٔ `i`، مقدار `a(i)` میانگین فاصله تا سایر اعضای خوشهٔ خودش است. مقدار `b(i)` کمترین میانگین فاصله تا یکی از خوشه‌های دیگر است.

\[
s(i)=\frac{b(i)-a(i)}{\max(a(i),b(i))}
\]

مقدار نزدیک ۱ یعنی نقطه به‌خوبی جدا شده، مقدار نزدیک صفر یعنی نقطه روی مرز خوشه‌هاست و مقدار منفی می‌تواند نشان دهد نقطه به خوشهٔ دیگری نزدیک‌تر است. امتیاز کل، میانگین امتیاز همهٔ نقاط است.

### کد

</div>

```python
def average_intra_cluster_distance(X, labels, i):
    """a(i): mean distance from point i to other points in its cluster."""
    same = np.flatnonzero(labels == labels[i])
    same = same[same != i]
    if len(same) == 0:
        return 0.0
    return float(np.linalg.norm(X[same] - X[i], axis=1).mean())

def average_nearest_cluster_distance(X, labels, i):
    """b(i): smallest mean distance from point i to another cluster."""
    other_clusters = np.unique(labels[labels != labels[i]])
    means = [np.linalg.norm(X[labels == c] - X[i], axis=1).mean() for c in other_clusters]
    return float(min(means))

def silhouette_samples_scratch(X, labels):
    X, labels = np.asarray(X, float), np.asarray(labels)
    if len(np.unique(labels)) < 2:
        raise ValueError('Silhouette is undefined for a single cluster')
    scores = np.zeros(len(X))
    for i in range(len(X)):
        if np.sum(labels == labels[i]) == 1:
            scores[i] = 0.0  # standard convention for singleton clusters
            continue
        a_i = average_intra_cluster_distance(X, labels, i)
        b_i = average_nearest_cluster_distance(X, labels, i)
        scores[i] = (b_i - a_i) / max(a_i, b_i)
    return scores

def silhouette_score_scratch(X, labels):
    return float(silhouette_samples_scratch(X, labels).mean())

def grid_search_clusters(X, k_values=range(1, 11), random_state=42):
    results, models = [], {}
    for k in k_values:
        model = KMeansScratch(k, max_iter=100, random_state=random_state).fit(X)
        models[k] = model
        score = np.nan if k == 1 else silhouette_score_scratch(X, model.labels_)
        results.append({'k': k, 'silhouette': score, 'inertia': model.inertia_})
    return pd.DataFrame(results), models

grid_results, grid_models = grid_search_clusters(X)
display(grid_results.round(4))
```

<div dir="rtl" align="right">

### توضیح خطوط ۱ تا ۱۳: توابع `a(i)` و `b(i)`

- **خط ۱:** تابع فاصلهٔ درون‌خوشه‌ای را تعریف می‌کند.
- **خط ۲:** docstring تابع `a(i)` است.
- **خط ۳:** index نقاط هم‌خوشه با نقطهٔ `i` را پیدا می‌کند.
- **خط ۴:** خود نقطهٔ `i` را حذف می‌کند تا فاصلهٔ صفر با خودش وارد میانگین نشود.
- **خط ۵:** singleton بودن خوشه را بررسی می‌کند.
- **خط ۶:** برای singleton مقدار صفر برمی‌گرداند.
- **خط ۷:** میانگین فاصلهٔ اقلیدسی نقطه تا هم‌خوشه‌ای‌هایش را محاسبه می‌کند.
- **خط ۸:** خط خالی است.
- **خط ۹:** تابع فاصله تا نزدیک‌ترین خوشهٔ متفاوت را تعریف می‌کند.
- **خط ۱۰:** docstring مربوط به `b(i)` است.
- **خط ۱۱:** شناسهٔ خوشه‌های دیگر را بدون تکرار استخراج می‌کند.
- **خط ۱۲:** میانگین فاصلهٔ نقطهٔ `i` تا هر خوشهٔ دیگر را می‌سازد.
- **خط ۱۳:** کوچک‌ترین میانگین را به‌عنوان `b(i)` برمی‌گرداند.

### توضیح خطوط ۱۵ تا ۳۰: Silhouette هر نقطه و میانگین کل

- **خط ۱۴:** خط خالی است.
- **خط ۱۵:** تابع محاسبهٔ امتیاز برای تک‌تک نمونه‌ها را تعریف می‌کند.
- **خط ۱۶:** ورودی‌ها را به آرایهٔ NumPy تبدیل می‌کند.
- **خط ۱۷:** بررسی می‌کند حداقل دو خوشه وجود داشته باشد.
- **خط ۱۸:** برای یک خوشه خطا می‌دهد، چون `b(i)` قابل تعریف نیست.
- **خط ۱۹:** برای هر مشاهده یک محل امتیاز با مقدار اولیهٔ صفر می‌سازد.
- **خط ۲۰:** روی index تمام نقاط حلقه می‌زند.
- **خط ۲۱:** بررسی می‌کند نقطه تنها عضو خوشه است یا نه.
- **خط ۲۲:** برای singleton امتیاز استاندارد صفر را ثبت می‌کند.
- **خط ۲۳:** به نقطهٔ بعدی می‌رود.
- **خط ۲۴:** `a(i)` را محاسبه می‌کند.
- **خط ۲۵:** `b(i)` را محاسبه می‌کند.
- **خط ۲۶:** فرمول Silhouette را اجرا و نتیجه را ذخیره می‌کند.
- **خط ۲۷:** همهٔ امتیازهای نقطه‌ای را برمی‌گرداند.
- **خط ۲۸:** خط خالی است.
- **خط ۲۹:** تابع امتیاز کلی را تعریف می‌کند.
- **خط ۳۰:** میانگین امتیاز همهٔ نقاط را محاسبه می‌کند.

---

## بخش ۸: Grid search برای تعداد خوشه‌ها

### توضیح مناسب برای TA

تعداد خوشه یک hyperparameter است، چون K-means آن را خودکار یاد نمی‌گیرد. طبق صورت آزمایش، مقادیر ۱ تا ۱۰ را امتحان می‌کنم. برای هر `k` یک مدل جدید fit می‌شود و Silhouette و inertia آن ذخیره می‌شوند. برای `k=1`، Silhouette تعریف نشده است، پس `NaN` ذخیره و هنگام انتخاب بهترین مقدار کنار گذاشته می‌شود.

### توضیح خط‌به‌خط ادامهٔ همان سلول

- **خط ۳۱:** خط خالی است.
- **خط ۳۲:** تابع grid search را تعریف می‌کند. `range(1, 11)` اعداد ۱ تا ۱۰ را می‌سازد، چون حد بالا در Python وارد بازه نمی‌شود.
- **خط ۳۳:** یک list خالی برای نتیجه‌ها و یک dictionary خالی برای مدل‌ها می‌سازد.
- **خط ۳۴:** روی هر مقدار پیشنهادی `k` حلقه می‌زند.
- **خط ۳۵:** برای `k` فعلی مدل تازه‌ای می‌سازد و fit می‌کند.
- **خط ۳۶:** مدل fitشده را با کلید `k` در dictionary نگه می‌دارد.
- **خط ۳۷:** اگر `k=1` باشد `NaN` و در غیر این صورت Silhouette را ذخیره می‌کند.
- **خط ۳۸:** `k`، Silhouette و inertia را به نتایج اضافه می‌کند.
- **خط ۳۹:** نتایج را به DataFrame تبدیل و همراه مدل‌ها برمی‌گرداند.
- **خط ۴۰:** خط خالی است.
- **خط ۴۱:** grid search را اجرا و دو خروجی آن را جدا می‌کند.
- **خط ۴۲:** جدول نتیجه‌ها را با چهار رقم اعشار نمایش می‌دهد.

---

## بخش ۹: انتخاب و رسم بهترین `k`

### توضیح مناسب برای TA

ردیف دارای Silhouette نامشخص را حذف می‌کنم و `k` دارای بیشترین امتیاز را انتخاب می‌کنم. برای این داده و initialization، بهترین نتیجه `k=2` با امتیاز حدود `0.6521` است. نقطهٔ قرمز در نمودار مقدار انتخاب‌شده را مشخص می‌کند.

### کد

</div>

```python
valid = grid_results.dropna(subset=['silhouette'])
optimal_k = int(valid.loc[valid['silhouette'].idxmax(), 'k'])
optimal_score = float(valid['silhouette'].max())
print(f'Best k = {optimal_k}, silhouette coefficient = {optimal_score:.4f}')

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(valid['k'], valid['silhouette'], marker='o', linewidth=2)
ax.scatter([optimal_k], [optimal_score], s=130, color='crimson', zorder=3, label='Best k')
ax.set(xticks=range(1, 11), xlabel='Number of clusters (k)',
       ylabel='Mean silhouette coefficient', title='Grid search over k=1...10')
ax.legend(); plt.tight_layout(); plt.show()
```

<div dir="rtl" align="right">

### توضیح خط‌به‌خط

- **خط ۱:** ردیف‌های دارای Silhouette گمشده را حذف می‌کند؛ یعنی `k=1` کنار گذاشته می‌شود.
- **خط ۲:** `idxmax` ردیف بیشترین امتیاز را پیدا می‌کند و `.loc` مقدار `k` همان ردیف را برمی‌دارد.
- **خط ۳:** بیشترین Silhouette را ذخیره می‌کند.
- **خط ۴:** بهترین `k` و امتیاز را با چهار رقم اعشار چاپ می‌کند.
- **خط ۵:** خط خالی است.
- **خط ۶:** figure و axes می‌سازد.
- **خط ۷:** Silhouette را نسبت به `k` با خط و marker دایره‌ای رسم می‌کند.
- **خط ۸:** یک marker قرمز بزرگ روی بهترین نقطه قرار می‌دهد. `zorder=3` آن را روی خط رسم می‌کند.
- **خطوط ۹ تا ۱۰:** tickها، نام محورها و عنوان را تنظیم می‌کنند.
- **خط ۱۱:** legend، اصلاح layout و نمایش نمودار را در یک خط انجام می‌دهد.

---

## بخش ۱۰: نمایش و خلاصه‌سازی خوشه‌های بهینه

### توضیح مناسب برای TA

مدل fitشدهٔ مربوط به بهترین `k` را از dictionary می‌گیرم. رنگ نقاط عضویت خوشه و Xها centroid را نشان می‌دهند. جدول خلاصه تعداد اعضا، میانگین اجاره، میانگین درآمد و نام استان‌ها را برای هر خوشه ارائه می‌دهد. در این اجرا Stockholm، Uppsala و Skåne گروه اجارهٔ بالاتر را تشکیل می‌دهند و ۱۸ استان دیگر در گروه دوم هستند. شماره‌های خوشه ترتیب یا ارزش ذاتی ندارند.

### کد

</div>

```python
optimal_model = grid_models[optimal_k]
fig, ax = plt.subplots(figsize=(9, 6))
ax.scatter(X[:, 0], X[:, 1], c=optimal_model.labels_, cmap='tab10',
           s=85, edgecolor='white')
ax.scatter(optimal_model.centroids_[:, 0], optimal_model.centroids_[:, 1],
           c=np.arange(optimal_k), cmap='tab10', marker='X', s=260,
           edgecolor='black', linewidth=1.2, label='Centroids')
for _, row in df.iterrows():
    ax.annotate(row['region'].split(' ', 1)[1].replace(' county', ''),
                (row[FEATURES[0]], row[FEATURES[1]]), xytext=(4, 4),
                textcoords='offset points', fontsize=7)
ax.set(xlabel='Annual rent per m²', ylabel='Average yearly income (KSEK)',
       title=f'Optimal K-means clustering (k={optimal_k})')
ax.legend(); plt.tight_layout(); plt.show()

cluster_summary = (df.assign(cluster=optimal_model.labels_)
                   .groupby('cluster')
                   .agg(count=('region', 'size'),
                        mean_rent=('Annual rent sqm', 'mean'),
                        mean_income=('Avg yearly inc KSEK', 'mean'),
                        regions=('region', lambda s: ', '.join(s.str.replace(r'^\d+ ', '', regex=True)))))
display(cluster_summary)
```

<div dir="rtl" align="right">

### توضیح خط‌به‌خط

- **خط ۱:** مدل ذخیره‌شده با کلید بهترین `k` را می‌گیرد.
- **خط ۲:** figure و axes را ایجاد می‌کند.
- **خطوط ۳ تا ۴:** نقاط استان‌ها را با رنگ label خوشه رسم می‌کنند.
- **خطوط ۵ تا ۷:** centroidها را به‌شکل X بزرگ و با رنگ متناظر رسم می‌کنند.
- **خط ۸:** روی ردیف‌های استان‌ها حلقه می‌زند.
- **خط ۹:** کد استان و کلمهٔ `county` را برای ساخت برچسب کوتاه‌تر حذف می‌کند.
- **خط ۱۰:** برچسب را در مختصات اجاره و درآمد قرار می‌دهد.
- **خط ۱۱:** offset و اندازهٔ فونت را تنظیم می‌کند.
- **خطوط ۱۲ تا ۱۳:** نام محورها و عنوان دارای `optimal_k` را تعیین می‌کنند.
- **خط ۱۴:** legend، layout و نمایش نمودار را انجام می‌دهد.
- **خط ۱۵:** خط خالی است.
- **خط ۱۶:** یک DataFrame موقت می‌سازد و label خوشه را به آن اضافه می‌کند.
- **خط ۱۷:** ردیف‌ها را بر اساس `cluster` گروه‌بندی می‌کند.
- **خط ۱۸:** aggregateهای نام‌گذاری‌شده را شروع و تعداد اعضا را محاسبه می‌کند.
- **خط ۱۹:** میانگین اجارهٔ هر خوشه را محاسبه می‌کند.
- **خط ۲۰:** میانگین درآمد هر خوشه را محاسبه می‌کند.
- **خط ۲۱:** نام استان‌های هر خوشه را در یک رشتهٔ جداشده با ویرگول قرار می‌دهد؛ regex کد عددی ابتدای نام را حذف می‌کند.
- **خط ۲۲:** جدول خلاصه را نمایش می‌دهد.

---

## بخش ۱۱: پیش‌بینی خوشهٔ سه منطقهٔ جدید

### توضیح مناسب برای TA

سه منطقهٔ داده‌شده را با همان ترتیب ویژگی‌های آموزشی، یعنی ابتدا اجاره و سپس درآمد، وارد می‌کنم. `predict` الگوریتم را دوباره fit نمی‌کند؛ فاصلهٔ نقطهٔ جدید تا centroidهای یادگرفته‌شده را اندازه می‌گیرد و شناسهٔ نزدیک‌ترین centroid را برمی‌گرداند. Region A و Region C در خوشهٔ ۱ و Region B با اجارهٔ بالاتر در خوشهٔ ۰ قرار می‌گیرند.

### کد

</div>

```python
new_regions = np.array([[1010, 320.12], [1258, 320.00], [980, 292.40]])
new_labels = optimal_model.predict(new_regions)
predictions = pd.DataFrame(new_regions, columns=FEATURES)
predictions.insert(0, 'new_region', ['Region A', 'Region B', 'Region C'])
predictions['predicted_cluster'] = new_labels
display(predictions)

fig, ax = plt.subplots(figsize=(9, 6))
ax.scatter(X[:, 0], X[:, 1], c=optimal_model.labels_, cmap='tab10',
           s=75, alpha=.65, edgecolor='white', label='Original counties')
ax.scatter(optimal_model.centroids_[:, 0], optimal_model.centroids_[:, 1],
           c=np.arange(optimal_k), cmap='tab10', marker='X', s=250,
           edgecolor='black', linewidth=1.2, label='Centroids')
ax.scatter(new_regions[:, 0], new_regions[:, 1], c=new_labels, cmap='tab10',
           vmin=0, vmax=max(optimal_k - 1, 1), marker='*', s=330,
           edgecolor='black', linewidth=1.2, label='New regions')
for name, (x, y) in zip(predictions['new_region'], new_regions):
    ax.annotate(name, (x, y), xytext=(7, 7), textcoords='offset points', weight='bold')
ax.set(xlabel='Annual rent per m²', ylabel='Average yearly income (KSEK)',
       title='Optimal clusters and predictions for new regions')
ax.legend(); plt.tight_layout(); plt.show()
```

<div dir="rtl" align="right">

### توضیح خط‌به‌خط

- **خط ۱:** یک آرایهٔ ۳ در ۲ می‌سازد؛ هر لیست داخلی `[اجاره، درآمد]` یک منطقه است.
- **خط ۲:** هر منطقه را با centroidهای مدل به نزدیک‌ترین خوشه نسبت می‌دهد.
- **خط ۳:** مقادیر را با نام ویژگی‌ها به DataFrame تبدیل می‌کند.
- **خط ۴:** نام‌های Region A تا C را در ستون اول درج می‌کند.
- **خط ۵:** label پیش‌بینی‌شده را به جدول اضافه می‌کند.
- **خط ۶:** جدول پیش‌بینی را نمایش می‌دهد.
- **خط ۷:** خط خالی است.
- **خط ۸:** figure و axes می‌سازد.
- **خطوط ۹ تا ۱۰:** استان‌های اصلی را نیمه‌شفاف رسم می‌کنند تا نقاط جدید بهتر دیده شوند.
- **خطوط ۱۱ تا ۱۳:** centroidها را با علامت X رسم می‌کنند.
- **خطوط ۱۴ تا ۱۶:** نقاط جدید را با علامت ستاره رسم می‌کنند. `vmin` و `vmax` هماهنگی رنگ labelها را حفظ می‌کنند.
- **خط ۱۷:** با `zip` نام هر منطقه را با مختصات آن جفت و مختصات را در `(x, y)` باز می‌کند.
- **خط ۱۸:** نام منطقه را کمی کنار ستاره و به‌صورت bold می‌نویسد.
- **خطوط ۱۹ تا ۲۰:** نام محورها و عنوان نمودار را تنظیم می‌کنند.
- **خط ۲۱:** legend، layout و نمایش نمودار را انجام می‌دهد.

### ارزیابی پیش‌بینی‌ها

از نظر هندسی، پیش‌بینی‌ها با قانون نزدیک‌ترین centroid سازگار هستند. Region B اجارهٔ بالایی دارد و وارد گروه با اجارهٔ بالاتر می‌شود. Region C اجارهٔ پایین دارد و وارد خوشهٔ بزرگ‌تر می‌شود. Region A اجارهٔ پایین ولی درآمد نسبتاً بالایی دارد، پس عضو معمولی خوشهٔ خود نیست و باید با احتیاط تفسیر شود.

این ارزیابی accuracy یادگیری نظارت‌شده نیست، چون label واقعی نداریم. می‌توانیم دربارهٔ معقول‌بودن موقعیت، فاصله از centroid و کیفیت Silhouette صحبت کنیم، اما accuracy پیش‌بینی قابل محاسبه نیست.

---

## بخش ۱۲: تفسیرها و محدودیت‌های مهم

### توضیح مناسب برای TA

1. **K-means بدون ناظر است.** گروه‌ها را کشف می‌کند و از categoryهای ازقبل‌مشخص یاد نمی‌گیرد.
2. **شناسهٔ خوشه دلخواه است.** خوشهٔ ۰ ذاتاً بهتر، پایین‌تر یا مهم‌تر از خوشهٔ ۱ نیست.
3. **مقدار منتخب `k=2` است.** بیشترین Silhouette آزمایش‌شده حدود `0.6521` بود.
4. **الگوریتم از فاصلهٔ اقلیدسی استفاده می‌کند.** assignment و prediction هر دو به فاصله تا centroid وابسته‌اند.
5. **مقیاس اهمیت دارد.** بازهٔ عددی اجاره بزرگ‌تر است، پس در فاصلهٔ خام اثر بیشتری از درآمد دارد. برای پیروی از صورت آزمایش و استفادهٔ مستقیم از نقاط جدید، مقادیر خام استفاده شدند.
6. **Initialization اهمیت دارد.** K-means ممکن است با centroidهای اولیهٔ متفاوت به جواب محلی متفاوت برسد. seed ثابت این اجرا را تکرارپذیر می‌کند.
7. **تقسیم train/validation نداریم.** target واقعی وجود ندارد و Silhouette معیار درونی است، نه اثبات ground truth.
8. **داده کوچک است.** فقط ۲۱ استان از یک سال داریم؛ بنابراین نباید نتیجه را بیش از حد تعمیم دهیم.

---

## متن کوتاه آماده برای ارائه

> من ۲۱ استان سوئد را با استفاده از اجارهٔ سالانه به‌ازای هر مترمربع و میانگین درآمد سالانه خوشه‌بندی کردم. ابتدا داده را بررسی و نمودار پراکندگی آن را رسم کردم. سپس K-means را از ابتدا و در قالب یک کلاس پایتون پیاده‌سازی کردم. کلاس، نمونه‌های تصادفی را به‌عنوان centroid اولیه انتخاب می‌کند، هر نقطه را با فاصلهٔ اقلیدسی به نزدیک‌ترین centroid نسبت می‌دهد، centroid را با میانگین اعضای خوشه به‌روزرسانی می‌کند و این کار را تا همگرایی تکرار می‌کند.
>
> چون تعداد خوشه یک hyperparameter است، Silhouette را نیز از ابتدا پیاده‌سازی کردم. برای هر نقطه، میانگین فاصله در خوشهٔ خودش یعنی `a(i)` را با میانگین فاصله تا نزدیک‌ترین خوشهٔ دیگر یعنی `b(i)` مقایسه کردم. مقادیر `k` از ۱ تا ۱۰ آزمایش شدند. Silhouette برای یک خوشه تعریف نشده، بنابراین انتخاب بهترین مدل با مقادیر ۲ تا ۱۰ انجام شد.
>
> بهترین نتیجه دو خوشه با Silhouette حدود ۰٫۶۵۲۱ بود. یک خوشه شامل استان‌های با اجارهٔ بالاتر یعنی Stockholm، Uppsala و Skåne است و خوشهٔ دیگر بقیهٔ استان‌ها را شامل می‌شود. در پایان، سه منطقهٔ جدید را به نزدیک‌ترین centroid نسبت دادم و با ستاره در نمودار نشان دادم. Region A به‌دلیل اجارهٔ پایین و درآمد نسبتاً بالا عضو کم‌تر معمول خوشهٔ خود است.
>
> چون این یک مسئلهٔ بدون ناظر است، label واقعی و accuracy معمول نداریم. نتیجه را باید نوعی segmentation مفید از همین داده دانست. محدودیت مهم این است که مقیاس عددی بزرگ‌تر اجاره باعث می‌شود در فاصلهٔ اقلیدسی خام اثر بیشتری از درآمد داشته باشد.

---

## سؤال‌های احتمالی TA و پاسخ‌ها

### چرا Silhouette برای `k=1` تعریف نشده است؟

محاسبهٔ `b(i)` به یک خوشهٔ دیگر نیاز دارد. وقتی فقط یک خوشه داریم، خوشهٔ جایگزینی وجود ندارد و فرمول معنی ندارد.

### چرا `k` را با کمترین inertia انتخاب نمی‌کنیم؟

با افزایش تعداد خوشه‌ها، inertia تقریباً همیشه کم می‌شود و اگر هر نقطه خوشهٔ جدا داشته باشد می‌تواند صفر شود. بنابراین کمترین inertia به‌تنهایی معیار مناسبی برای انتخاب `k` نیست. Silhouette هم فشردگی و هم جدایی را در نظر می‌گیرد.

### چرا در محاسبهٔ فاصله `axis=2` داریم؟

بعد از broadcasting، آرایه سه بُعد دارد: مشاهده، centroid و ویژگی. بُعد آخر شامل اختلاف دو ویژگی است. `axis=2` این اختلاف‌های ویژگی را به یک فاصله برای هر جفت مشاهده-centroid تبدیل می‌کند.

### چرا centroid با میانگین به‌روزرسانی می‌شود؟

K-means مجموع مربع فاصله‌های اقلیدسی را کمینه می‌کند. برای اعضای ثابت یک خوشه، میانگین حسابی نقطه‌ای است که این مجموع را کمینه می‌کند.

### Inertia چیست؟

مجموع مربع فاصلهٔ هر مشاهده تا centroid خوشهٔ خودش است. فشردگی داخل خوشه را اندازه می‌گیرد، ولی جدایی خوشه‌های مختلف را مستقیماً اندازه نمی‌گیرد.

### آیا K-means واقعاً نقاط جدید را classify می‌کند؟

نقطهٔ جدید را به یکی از خوشه‌های کشف‌شده نسبت می‌دهد، ولی این کار supervised classification با class labelهای واقعی نیست. نوت‌بوک از واژهٔ صورت آزمایش استفاده می‌کند، ولی این تفاوت را روشن می‌سازد.

### چرا initialization می‌تواند جواب را تغییر دهد؟

تابع هدف K-means ممکن است چند minimum محلی داشته باشد. centroidهای شروع متفاوت می‌توانند الگوریتم را به جواب نهایی متفاوتی برسانند. seed ثابت نتیجه را قابل تکرار می‌کند و اجرای چند initialization می‌تواند extension مناسبی باشد.

### چرا استانداردسازی ویژگی‌ها ممکن است خوشه‌ها را تغییر دهد؟

استانداردسازی دو ویژگی را در مقیاس قابل مقایسه قرار می‌دهد. بدون آن، اختلاف ۱۰۰ واحد اجاره بسیار بیشتر از اختلاف ۱۰ واحد درآمد روی فاصله اثر دارد. با استانداردسازی، تغییر نسبی هر دو ویژگی سهم متعادل‌تری خواهد داشت.

---

## نکات نهایی برای حفظ‌کردن

- داده: ۲۱ استان سوئد از سال ۲۰۲۰.
- ویژگی‌ها: اجارهٔ سالانه به‌ازای مترمربع و میانگین درآمد سالانه برحسب KSEK.
- الگوریتم: K-means دست‌نویس با فاصلهٔ اقلیدسی.
- مرحلهٔ update: نسبت‌دادن نقطه به نزدیک‌ترین centroid و جایگزینی centroid با میانگین اعضا.
- شرط توقف: حرکت centroid کمتر یا مساوی tolerance شود یا iteration به حداکثر برسد.
- grid مربوط به hyperparameter: از `k=1` تا `k=10`.
- معیار انتخاب: میانگین Silhouette.
- بهترین نتیجه: `k=2` با Silhouette حدود `0.6521`.
- پیش‌بینی: Region A به خوشهٔ ۱، Region B به خوشهٔ ۰ و Region C به خوشهٔ ۱.
- محدودیت اصلی: به‌دلیل مقیاس خام، اجاره از درآمد اثر بیشتری روی فاصله دارد.

</div>
